import os
import time
from huggingface_hub import list_repo_files, hf_hub_download

repo_id = "CropNet/CropNet"
folder_prefixes = [
    "USDA Crop Dataset/",
    "WRF-HRRR Computed Dataset/data/",
]

# Target states to download
target_states = ["MS", "LA", "IA", "IL"]

# Create the data/CropNet directory if it doesn't exist
local_dir = "data/CropNet"
os.makedirs(local_dir, exist_ok=True)

print(f"🚀 Starting download from {repo_id}")
print(f"📂 Target directory: {local_dir}")
print(f"🏛️  Target states: {', '.join(target_states)}")
print(f"📋 Processing {len(folder_prefixes)} folders: {', '.join(folder_prefixes)}")

# List all files in the repo
print("\n🔍 Fetching repository file list...")
all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")
print(f"📊 Total files in repository: {len(all_files)}")

# Download files from both datasets
total_downloaded = 0
total_skipped = 0
total_failed = 0
total_filtered = 0
start_time = time.time()

for folder_idx, folder_prefix in enumerate(folder_prefixes, 1):
    print(
        f"\n📁 Processing folder {folder_idx}/{len(folder_prefixes)}: {folder_prefix}"
    )

    # Filter to the desired folder
    folder_files = [f for f in all_files if f.startswith(folder_prefix)]

    # Apply state filtering only to WRF-HRRR dataset, download all USDA Crop Dataset files
    target_files = []
    if "WRF-HRRR" in folder_prefix:
        # Filter by state for WRF-HRRR dataset
        for file in folder_files:
            if any(f"/{state}/" in file for state in target_states):
                target_files.append(file)
            else:
                total_filtered += 1
        print(
            f"✅ Found {len(target_files)} files for target states in {folder_prefix}"
        )
        print(
            f"🚫 Filtered out {len(folder_files) - len(target_files)} files from other states"
        )
    else:
        # Download all files for USDA Crop Dataset (no state filtering)
        target_files = folder_files
        print(
            f"✅ Found {len(target_files)} files in {folder_prefix} (no state filtering applied)"
        )

    # Download each file
    for file_idx, file in enumerate(target_files, 1):
        local_file_path = os.path.join(local_dir, file)

        # Check if file already exists and has reasonable size
        if os.path.exists(local_file_path) and os.path.getsize(local_file_path) > 0:
            print(
                f"⏭️  Skipping ({file_idx}/{len(target_files)}): {file} (already exists)"
            )
            total_skipped += 1
            continue

        try:
            print(f"⬇️  Downloading ({file_idx}/{len(target_files)}): {file}")
            local_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file,
                local_dir=local_dir,
                force_download=True,  # Force download to handle corrupted files
            )
            print(f"✅ Downloaded: {local_path}")
            total_downloaded += 1
        except Exception as e:
            print(f"❌ Failed to download {file}: {e}")
            total_failed += 1
            continue

elapsed_time = time.time() - start_time
print(f"\n🎉 Download process completed!")
print(f"📊 Summary:")
print(f"   ✅ Downloaded: {total_downloaded} files")
print(f"   ⏭️  Skipped: {total_skipped} files (already existed)")
print(f"   ❌ Failed: {total_failed} files")
print(f"   🚫 Filtered: {total_filtered} files (other states)")
print(f"   ⏱️  Total time: {elapsed_time:.1f} seconds")
print(f"📂 Files saved in: {local_dir}")
