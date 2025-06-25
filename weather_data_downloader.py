from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="notadib/NASA-Power-Daily-Weather",
    repo_type="dataset",
    allow_patterns="csvs/*weekly*",
    local_dir="data/nasa_power",
)
