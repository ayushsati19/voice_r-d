import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="ajajali09/indian-noise-dataset",
    repo_type="dataset",
    local_dir="./data/indian-noise-dataset",
    max_workers=8,
    # allow_patterns=["*.wav", "*.flac", "*.mp3", "*.json", "*.csv"],  # optional filter
)
print("Downloaded to:", path)