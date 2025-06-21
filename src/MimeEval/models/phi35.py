from typing import List, Optional, Union, Tuple, Any
import torch
import os
from tqdm import tqdm
import logging

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
from decord import VideoReader, cpu

from .vlms import VLM
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from vlms import VLM

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler if none exists
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

def encode_video(video_path, max_num_frames=10):
    """ convering from video to image lists """
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    frame_idx = list(range(0, len(vr), max(1, round(vr.get_avg_fps()))))

    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)

    frames = vr.get_batch(frame_idx).asnumpy()
    return [Image.fromarray(frame.astype('uint8')) for frame in frames]

def process_media_files(file_path, max_num_frames=10):
    """ process video & image files """
    images = []

    #full_path = os.path.join(video_dir, file_path)
    if is_video_file(file_path):
        frames = encode_video(file_path, max_num_frames=max_num_frames)
        images.extend(frames)  # 비디오 프레임을 이미지 리스트로 추가
    else:
        images.append(load_image(file_path))
    return images

class PhiVLM(VLM):

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3.5-vision-instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(model_name, device, torch_dtype, max_new_tokens, **kwargs)
        logger.info(f"Initializing Phi-3.5-vision-instruct with model: {model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
                #_attn_implementation="flash_attention_2",
            ).to(device).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
        )

    def generate_text_singleimageinput(self, image, text: str) -> str:
        image_inputs = process_media_files(image)
        placeholder = "".join([f"<|image_{i}|>\n" for i in range(1, len(image_inputs) + 1)])

        msg = {
            "role": "user",
            "content": placeholder + text,
        }
        # Preparation for inference
        text_inputs = self.processor.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                return_tensors="pt",
        ).to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def generate_text_videoinput(self, video_path, text: str) -> str:
        messages = [
                {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,                # video_path can be file path or URL
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        image_inputs = process_media_files(video_path, max_num_frames=1)
        placeholder = "".join([f"<|image_{i}|>\n" for i in range(1, len(image_inputs) + 1)])

        msg = {
            "role": "user",
            "content": placeholder + text,
        }
        # Preparation for inference
        text_inputs = self.processor.tokenizer.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                return_tensors="pt",
        ).to(self.model.device)

        generation_args = {
            "max_new_tokens": 1000,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **generation_args)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

if __name__ == "__main__":
    model_name = "microsoft/Phi-3.5-vision-instruct"
    model = PhiVLM(model_name)
    input_text = """"What action is the person doing in this image/video? Describe the action in a single phrase.

    You can think out the action in a chain of thought, but please reply on the final line of your response, a single short phrase (under 5 words).

    This action is being 'mimed' meaning backgrounds or objects that are relevant may not be present. Think about only the *action* taking place in the video, and give a response for what it looks like the character is "acting out" or doing "charades" of."""

    images = ["samples/kicking_initial.png", "samples/kicking_final.png"]
    num_correct = 0
    # for image_path in images:
    #     print("IMAGE PATH:", image_path)
    #     #input_text = "What action is the person in the image doing? Answer in a few words only."
    #     generated_text = model.generate_text_singleimageinput(image_path, input_text)
    #     #print("INPUT:", input_text)
    #     print("Qwen2VL ANSWER:", generated_text)

    # print("-"*100)

    for activity in ["fishing"]:
        video_path = f"samples/{activity}.mp4"
        print("VIDEO PATH:", video_path)
        generated_text = model.generate_text_videoinput(video_path, input_text)
        print("ANSWER:", generated_text)
    #import pdb; pdb.set_trace()
