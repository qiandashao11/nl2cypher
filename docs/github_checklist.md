# GitHub Publishing Checklist

Before pushing this project publicly:

1. Revoke any Hugging Face token that was previously stored in source files.
2. Confirm `rg -n "hf_|sk-|api_key|password" --glob "*.py" .` has no real
   credentials.
3. Confirm `find . -type f -size +100M` only returns ignored model artifacts.
4. Run `git status --ignored` and make sure model/checkpoint folders are ignored.
5. Add a license if you want other people to reuse the code.
6. Keep LoRA adapters and base models in Hugging Face, Git LFS, or private
   storage, then add download instructions to the README.
