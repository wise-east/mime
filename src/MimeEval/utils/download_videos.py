import jsonlines
import os
from tqdm import tqdm
import argparse
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True, help="Path to mime dataset file")
parser.add_argument("--videos_dir", type=str, required=True, help="Path to mime dataset file") 
args = parser.parse_args()

f = jsonlines.open(args.dataset_path, 'r')
urls = []
for i, line in enumerate(f):
    urls.append(line['s3_url'])

dataset_dir = os.path.dirname(args.dataset_path)
videos_dir = os.path.join(args.videos_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)
urls = list(set(urls))
for url in tqdm(urls):
    video_name = url.split("/")[-1]
    if os.path.exists(os.path.join(videos_dir, video_name)):
        logger.debug(f"Skipping {url} because it already exists")
        continue
    command = f"wget {url} -P {videos_dir}"
    os.system(command)
    logger.debug(f"Downloaded {url}")