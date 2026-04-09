import os
import sys
import argparse

def download_models(target_dir, repo_id="digital-avatar/ditto-talkinghead"):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ Error: huggingface_hub not installed. Please install it with 'pip install huggingface_hub'")
        sys.exit(1)

    print(f"🚀 Downloading Ditto checkpoints from {repo_id}...")
    print(f"📍 Target directory: {target_dir}")
    
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Checkpoints in the HF repo are usually in a 'checkpoints' folder
        # We download everything but filtered to what we need if possible
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            # We want the contents of the 'checkpoints' folder from the repo to go into target_dir
            # or we download everything and the user handles it.
            # Based on install_ditto.sh, it expects the contents to be in $DITTO_DIR/checkpoints
            # If the HF repo has a 'checkpoints' folder, snapshot_download will create it.
        )
        print("✅ Download completed successfully.")
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Ditto TalkingHead checkpoints")
    parser.add_argument("target_dir", help="Directory to save checkpoints")
    parser.add_argument("--repo", default="digital-avatar/ditto-talkinghead", help="Hugging Face repo ID")
    
    args = parser.parse_args()
    download_models(args.target_dir, args.repo)
