import pandas as pd
import os 
from pathlib import Path
import json 
import re 
import hashlib 
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
from typing import List, Dict
import boto3
from loguru import logger
import requests
from concurrent.futures import ThreadPoolExecutor
from MimeEval.utils.constants import PACKAGE_DIR

# Initialize AWS client
aws_client = boto3.client("s3")

# Initialize sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_labels() -> Dict[str, List[str]]:
    """Load consolidated labels from jsonl file."""
    labels_data_path = PACKAGE_DIR / "data/consolidated_data.jsonl"
    with open(labels_data_path, "r") as f: 
        labels_data = [json.loads(line) for line in f]
    return {sample["action"]: sample["final_label"] for sample in labels_data}

def get_all_labels(action2label: Dict[str, List[str]]) -> List[str]:
    """Get unique list of all labels."""
    all_labels = []
    for labels in action2label.values():
        all_labels.extend(labels)
    return list(set(all_labels))

def sample_distractors(labels: List[str], all_labels: List[str], label_embeddings: np.ndarray, n: int) -> List[str]:
    """Sample n distractors from labels that are not too similar to the target label."""
    target_label = labels[0]
    
    # Remove the most similar 30 labels and random sample n labels from the rest 
    target_label_embedding = model.encode(target_label)
    similarities = np.dot(label_embeddings, target_label_embedding) / (np.linalg.norm(label_embeddings, axis=1) * np.linalg.norm(target_label_embedding))
    sorted_indices = np.argsort(similarities)
    
    most_similar_labels = [all_labels[i] for i in sorted_indices[-10:]]
    remaining_labels = [l for l in all_labels if l not in most_similar_labels] 
    
    assert target_label not in remaining_labels
    
    random.seed(42)
    return random.sample(remaining_labels, n)

def create_mime_data(mocap_info_data: pd.DataFrame, action2label: Dict[str, List[str]], all_labels: List[str], label_embeddings: np.ndarray) -> List[Dict]:
    """Create mime data entries from mocap info."""
    data = []
    for index, row in tqdm(mocap_info_data.iterrows(), total=len(mocap_info_data), desc="Processing mime data"):
        actor_name = "j" if "justin" in row["blender_name"] else "n"
        
        blender_name = row["blender_name"].split("/")[-1]
        action_name = Path(blender_name).stem 
        if action_name not in action2label:
            continue 
        
        labels = action2label[action_name]
        distractors = sample_distractors(labels, all_labels, label_embeddings, 3)
        
        configs = [
            ("man", "blank", "0", "base+blank@0"),
            ("spacesuit", "blank", "0", "adversarial+blank@0"),
            ("woman", "blank", "0", "woman+blank@0"),
            ("man", "aligned", "0", "base+aligned@0"),
            ("man", "misaligned", "0", "base+misaligned@0"),
            ("spacesuit", "aligned", "0", "adversarial+aligned@0"),
            ("spacesuit", "misaligned", "0", "adversarial+misaligned@0"),
            ("man", "blank", "90", "base+blank@90"),
            ("man", "blank", "180", "base+blank@180"),
            ("man", "blank", "270", "base+blank@270"),
        ]
        
        for config in configs:
            avatar, background_config, angle, config_name = config
            
            if background_config == "blank":
                background_name = "blank"
            elif background_config == "aligned":
                background_name = row["background_aligned"]
            elif background_config == "misaligned":
                background_name = row["background_misaligned"]
            else:
                raise ValueError(f"Invalid background config: {background_config}")
            
            video_name = f"{avatar}_{actor_name}_{action_name}_angle{angle}_{background_name}.mp4"
            s3_url = f"https://mime-understanding.s3.amazonaws.com/{video_name}"
            hash_id = hashlib.sha256(s3_url.encode()).hexdigest()[:8]
            data.append({
                "action": action_name,
                "angle": angle,
                "avatar": avatar,
                "background_config": background_config,
                "background_name": background_name,
                "distractors": '|'.join(distractors),
                "label": '|'.join(labels),
                "sample_id": hash_id,
                "s3_url": s3_url,
                "type": "MIME",
                "split": "train",
                "file_name": video_name,
                "config_name": config_name,
            })
    
    return data

def create_real_data(action2label: Dict[str, List[str]], mime_data: List[Dict]) -> List[Dict]:
    """Create real data entries from real videos."""
    # Get all existing videos in S3 using pagination
    videos_in_s3 = set()
    paginator = aws_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket='mime-understanding'):
        if 'Contents' in page:
            for obj in page['Contents']:
                videos_in_s3.add(obj['Key'])
    
    # Create mapping of action names to distractors from synthetic data
    action_name2distractors = {
        item["action"]: item["distractors"] for item in mime_data
    }
    
    data = []
    videos = Path(PACKAGE_DIR / "data" / "REAL" / "resized_videos" / "test").glob(pattern="*.mp4")
    
    for video in tqdm(videos, desc="Processing real data"):
        video_name = video.name
        action_name = video.stem.replace("real_", "")
        
        if action_name not in action2label:
            logger.error(f"Action {action_name} not in labels_dict for video {video_name}")
            continue 
        
        label = action2label[action_name]
        distractors = action_name2distractors[action_name]
        
        if video_name not in videos_in_s3:
            breakpoint() 
            try:
                aws_client.upload_file(video, "mime-understanding", f"{video_name}")
                logger.info(f"Uploaded video {video_name} to s3")
            except Exception as e:
                logger.error(f"Error uploading video {video_name}: {e}")
                continue
        else:
            logger.info(f"Video {video_name} already in s3")

        name_hash = hashlib.sha256(video_name.encode()).hexdigest()[:8]
        data.append({
            "action": action_name,
            "angle": "n/a", 
            "avatar": "n/a",
            "background_config": "n/a",
            "background_name": "n/a",
            "distractors": distractors,
            "label": '|'.join(label),
            "sample_id": name_hash,
            "s3_url": f"https://mime-understanding.s3.amazonaws.com/{video_name}",
            "type": "REAL", 
            "split": "test",
            "file_name": video_name,
            "config_name": "none",
        })
    
    return data

def check_s3_urls(data: List[Dict]) -> List[str]:
    """Check which S3 URLs are accessible."""
    missing_urls = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=10) as executor:
            def check_url(item):
                try:
                    response = session.head(item["s3_url"], timeout=5)
                    if response.status_code != 200:
                        return item["s3_url"]
                except:
                    return item["s3_url"]
                return None
                
            results = list(tqdm(executor.map(check_url, data), total=len(data), desc="Checking S3 URLs"))
            missing_urls = [url for url in results if url is not None]
    
    return missing_urls
    
def create_metadata_csv(data: List[Dict], video_dir: Path):
    """Create metadata.csv for this data for creating huggingface video dataset."""
    metadata_path = video_dir / "metadata.csv"
    with open(metadata_path, "w") as f:
        f.write(f"{','.join(data[0].keys())}\n")
        for item in data:
            try: 
                f.write(f"{','.join(item.values())}\n")
            except Exception as e:
                logger.error(f"Error writing item {item} to metadata.csv: {e}")
                breakpoint() 
            
    logger.info(f"Created metadata.csv at {metadata_path}")

def main():
    # Load labels
    action2label = load_labels()
    all_labels = get_all_labels(action2label)
    label_embeddings = model.encode(all_labels)
    
    # Load mocap info
    csv_path = PACKAGE_DIR / "data/filtered_mocap_info.csv"
    mocap_info_data = pd.read_csv(csv_path)
    
    # Create MIME data
    mime_data = create_mime_data(mocap_info_data, action2label, all_labels, label_embeddings)
    
    # save mime data to jsonl
    mime_data_path = PACKAGE_DIR / "data" / "mime_data_legacy.jsonl"
    with open(mime_data_path, "w") as f:
        for item in mime_data:
            f.write(json.dumps(item) + "\n")    
    logger.info(f"Saved mime data to {mime_data_path}")
    
    mime_cropped_video_dir = PACKAGE_DIR / "data" / "MIME" / "cropped_videos" / "test"
    mime_original_video_dir = PACKAGE_DIR / "data" / "MIME" / "videos" / "test"
    mime_cropped_video_dir.mkdir(parents=True, exist_ok=True)
    mime_original_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.csv for MIME data
    create_metadata_csv(mime_data, mime_cropped_video_dir)
    create_metadata_csv(mime_data, mime_original_video_dir)
    
    # Create REAL data based on live action footage from pexels
    real_data = create_real_data(action2label, mime_data)
    
    # save real data to jsonl
    real_data_path = PACKAGE_DIR / "data" / "real_data_legacy.jsonl"
    with open(real_data_path, "w") as f:
        for item in real_data:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Saved real data to {real_data_path}")
    
    real_resized_video_dir = PACKAGE_DIR / "data" / "REAL" / "resized_videos" / "test"
    real_original_video_dir = PACKAGE_DIR / "data" / "REAL" / "videos" / "test"
    real_resized_video_dir.mkdir(parents=True, exist_ok=True)
    real_original_video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata.csv for REAL data
    create_metadata_csv(real_data, real_resized_video_dir)
    create_metadata_csv(real_data, real_original_video_dir)
    
    # Combine all data
    all_data = mime_data + real_data
    
    # Save combined data
    output_path = PACKAGE_DIR / "data" / "data_combined_legacy.jsonl"
    with open(output_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
                
    # Check S3 URLs
    missing_urls = check_s3_urls(all_data)
    if missing_urls:
        logger.error(f"Found {len(missing_urls)} missing URLs:")
        for url in missing_urls:
            logger.info(url)
    else: 
        logger.info("All S3 URLs are accessible")
    
    # Print statistics
    logger.info(f"\nTotal number of samples: {len(all_data)}")
    logger.info(f"MIME samples: {len(mime_data)}")
    logger.info(f"REAL samples: {len(real_data)}")

if __name__ == "__main__":
    main() 