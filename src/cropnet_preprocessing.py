import os
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "CropNet/CropNet"
folder_prefix = "WRF-HRRR Computed Dataset/data/"  # adjust as needed

# Create the data/CropNet directory if it doesn't exist
local_dir = "data/CropNet"
os.makedirs(local_dir, exist_ok=True)

# List all files in the repo
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

# Filter to the desired folder
target_files = [f for f in all_files if f.startswith(folder_prefix)]
print(f"✅ Found {len(target_files)} files in {folder_prefix}")

# Download each file
for file in target_files:
    local_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename=file, local_dir=local_dir
    )
    print(f"Downloaded: {local_path}")
