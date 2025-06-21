from huggingface_hub import HfApi
from MimeEval.utils.constants import PACKAGE_DIR

api = HfApi()

# make sure that these datasets have been created on HuggingFace 

api.upload_folder(
    folder_path=PACKAGE_DIR / "data" / "REAL" / "resized_videos",
    repo_id="wise-east/mime-real-resized",
    repo_type="dataset"
)

api.upload_folder(
    folder_path=PACKAGE_DIR / "data" / "REAL" / "videos",
    repo_id="wise-east/mime-real-original",
    repo_type="dataset"
)

api.upload_folder(
    folder_path=PACKAGE_DIR / "data" / "MIME" / "cropped_videos",
    repo_id="wise-east/mime-cropped",
    repo_type="dataset",
)

api.upload_folder(
    folder_path=PACKAGE_DIR / "data" / "MIME" / "videos",
    repo_id="wise-east/mime-original",
    repo_type="dataset",
)