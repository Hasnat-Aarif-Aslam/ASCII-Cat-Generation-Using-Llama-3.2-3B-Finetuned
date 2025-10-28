# gguf_base_model.py
# type: ignore

# As we want to run this on the Modal platform, we import its SDK.
# Modal lets us define a container image, schedule a function on their infra,
# and run our Python code in that container—clean, reproducible, and headless.
import modal

# We name our Modal app so we can see it in the Modal UI and reuse artifacts.
app = modal.App("convert_gguf_manual")

# --- Build the container image our function will run inside ---
# The story here: we need a tiny Linux image with just enough tools to:
#  1) download a Hugging Face model snapshot (config.json + weights + tokenizer),
#  2) run the official llama.cpp converter,
#  3) and upload the resulting .gguf back to Hugging Face.
#
# We start from a small Debian base, add a few system packages for building llama.cpp,
# then pip-install the Python libraries the converter depends on (Transformers,
# SentencePiece, etc.). Finally, we add CPU PyTorch—because the converter imports torch.
base_image = (
    modal.Image.debian_slim()
    # We install command-line tools and build essentials:
    # - git: to clone llama.cpp
    # - build-essential + cmake: to compile llama.cpp binaries (optional but handy)
    # - curl + libcurl: used by various build steps and utilities
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")
    # We install Python deps:
    # - huggingface_hub[hf_transfer]: fast, reliable downloads & uploads to HF
    # - transformers: required by convert_hf_to_gguf.py to parse configs/tokenizers
    # - sentencepiece: tokenizer backend for Llama-family models
    # - protobuf + safetensors + numpy: common model file formats/utilities
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.24.0",
        "transformers>=4.45.0",
        "sentencepiece>=0.1.99",
        "protobuf>=4.25.3",
        "safetensors>=0.4.5",
        "numpy",
    )
    # And now the crucial piece: CPU PyTorch.
    # The llama.cpp converter imports `torch` even for CPU conversion.
    # We don’t need CUDA here—CPU wheels are lighter and sufficient.
    .run_commands(
        "python -m pip install --upgrade pip",
        "pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.*"
    )
)

# We define a Modal function—the entry point that runs in our container.
# gpu=None because conversion is a CPU task; no GPU acceleration needed.
# image=base_image ensures our function uses the container we just defined.
# timeout is generous for large models. We also pass a secret so the code
# can read HF_ACCESS_TOKEN at runtime for gated model access and uploads.
@app.function(
    gpu=None,
    image=base_image,
    timeout=60 * 60 * 2,
    secrets=[modal.Secret.from_name("hugginface-secret")],  # must expose HF_ACCESS_TOKEN
)
def convert_base_model_gguf(hf_model_repo: str, save_model_repo: str, outtype: str) -> str:
    """
    We tell a simple story:
      - Bring the base model from Hugging Face into a local folder (ensuring config.json exists).
      - Fetch/build llama.cpp so we have the official converter.
      - Run convert_hf_to_gguf.py to produce a .gguf file (f16 or a quant like q4_k_m).
      - Upload the .gguf back to your Hugging Face repo.
    """
    import os, subprocess
    from huggingface_hub import snapshot_download, HfApi

    # Our Hugging Face token arrives through Modal secrets as an env variable.
    token = os.environ["HF_ACCESS_TOKEN"]

    # We keep all work under /root for simplicity in this ephemeral container.
    workdir = "/root"
    hf_dir = f"{workdir}/hf_model"

    # (1) Download a proper Hugging Face snapshot to a local directory.
    # This guarantees a real `config.json`, tokenizer files, and weights—exactly what the
    # converter expects. allow_patterns keeps the download lean while pulling essentials.
    snapshot_download(
        repo_id=hf_model_repo,
        local_dir=hf_dir,
        token=token,
        allow_patterns=["*.json","*.safetensors","*.model","*.txt","tokenizer*","*.py","*model*"],
    )
    # If the repo lacked config.json we’d fail early—better than a cryptic converter error later.
    assert os.path.exists(f"{hf_dir}/config.json"), "config.json not found after download"

    # Optional “preflight” check: if Transformers can parse config and tokenizer locally,
    # the converter will almost certainly succeed.
    from transformers import AutoConfig, AutoTokenizer
    AutoConfig.from_pretrained(hf_dir)
    AutoTokenizer.from_pretrained(hf_dir)

    # (2) Bring in llama.cpp. We clone the repo and build it.
    # The Python converter is what we need, but having the binaries nearby is useful
    # (e.g., for quantization or quick local tests).
    llama_dir = f"{workdir}/llama.cpp"
    if not os.path.exists(llama_dir):
        subprocess.run(["git","clone","https://github.com/ggerganov/llama.cpp", llama_dir], check=True)
        subprocess.run(["cmake", llama_dir, "-B", f"{llama_dir}/build"], check=True)
        subprocess.run(["cmake", "--build", f"{llama_dir}/build", "-j"], check=True)

    # (3) Convert to GGUF.
    # In llama.cpp’s vocabulary, “not_quantized” corresponds to “f16”.
    # We normalize that here so you can pass familiar words or converter-native ones.
    outtype = "f16" if outtype.lower() == "not_quantized" else outtype

    # We choose a clear output filename: <repo-name>-<OUTTYPE>.gguf
    outfile = f"{workdir}/{save_model_repo.split('/')[-1]}-{outtype.upper()}.gguf"

    # We call the official converter script. The --split-max-size guard helps when
    # HF snapshots are large; it avoids running out of memory during consolidation.
    subprocess.run(
        ["python", f"{llama_dir}/convert_hf_to_gguf.py",
         hf_dir, "--outfile", outfile, "--outtype", outtype, "--split-max-size", "50G"],
        check=True,
    )
    # If the converter didn’t produce the file, we fail loudly and clearly here.
    assert os.path.exists(outfile), "GGUF file missing after conversion"

    # (4) Upload the finished artifact to your Hugging Face repo so you can use it
    # from anywhere (Ollama, llama.cpp, LM Studio, etc.).
    api = HfApi(token=token)
    api.create_repo(repo_id=save_model_repo, exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=outfile, path_in_repo=os.path.basename(outfile), repo_id=save_model_repo)

    # Return the local path (mainly for logging); the real deliverable is in your HF repo.
    return outfile

# Finally, we define a tiny local entrypoint so you can run:
#   python -m modal run path/to/gguf_base_model.py
# It triggers our Modal function remotely with the arguments below.
@app.local_entrypoint()
def main():
    # The model we’re converting: a base Meta Llama 3.2 3B repo on Hugging Face.
    hf_model_repo = "meta-llama/Llama-3.2-3B"

    # Where we’ll upload the produced .gguf artifact. Create once; reuse forever.
    save_model_repo = "Hasnat5/Llama-3.2-3B-guide-GGUF"

    # Pick your format:
    #   - "f16" for no quantization (biggest file, best fidelity),
    #   - "q4_k_m"/"q5_k_m"/"q8_0" for popular, tested quantization levels.
    outtype = "f16"  # or "q4_k_m", "q5_k_m", "q8_0"

    # And now we ask Modal to run our remote function with those arguments.
    path = convert_base_model_gguf.remote(hf_model_repo, save_model_repo, outtype)

    # A small epilogue: print the local path in the container (for logs).
    # The actual artifact is already uploaded to save_model_repo.
    print("Uploaded GGUF:", path)
