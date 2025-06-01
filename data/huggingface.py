import os
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./data",
    repo_id="Hyunoh430/ppe",
    repo_type="dataset",
)
