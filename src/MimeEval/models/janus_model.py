from typing import List, Optional, Union, Tuple, Any
import torch
from PIL import Image
import cv2
import numpy as np
import logging
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from .vlms import VLM

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

"""
Requires separate installation of https://github.com/deepseek-ai/Janus as janus package
"""

class JanusVLM(VLM):
    def __init__(
        self,
        model_name: str = "deepseek-ai/Janus-Pro-7B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(model_name, device, torch_dtype, max_new_tokens, **kwargs)
        logger.info(f"Initializing JanusVLM with model: {model_name}")
        
        # Initialize processor and tokenizer
        self.processor = VLChatProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        
        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device).eval()

    def generate_text_singleimageinput(self, image: Any, text: str) -> str:
        logger.debug("Processing images")
        # Prepare conversation format
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{text}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        print("processing images")
        # Process images and prepare inputs
        pil_images = load_pil_images(conversation)  # Load from path if string
        logger.debug(f"PIL IMAGES: {pil_images}")
        
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.device)

        # Get input embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # Decode and return answer
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    def extract_video_frames(
        self,
        video_path: str,
        frame_skip: int = 60,  # Skip N frames between extractions
    ) -> Tuple[List[Image.Image], int]:
        """
        Extract frames from video, converting them to PIL Images.
        Returns tuple of (list of frames as PIL Images, total frame count)
        """
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info("\nVideo properties:")
        logger.info(f"- FPS: {fps}")
        logger.info(f"- Total frames: {total_frames}")
        logger.info(f"- Duration: {duration:.2f} seconds")
        logger.info(f"- Frame skip: {frame_skip} (sampling every {frame_skip/fps:.2f} seconds)")

        frames = []
        frame_count = 0
        processed_count = 0

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
                
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            processed_count += 1

            if processed_count <= 3:  # Print details for first few frames
                logger.debug(f"Frame {frame_count}: shape={frame.shape}, dtype={frame.dtype}")

        video.release()
        logger.info("Processing complete:")
        logger.info(f"- Processed {processed_count}/{frame_count} frames")
        logger.info(f"- Final frame list size: {len(frames)}")
        return frames, total_frames

    def generate_text_videoinput(
        self,
        video_path: str,
        text: str,
        frame_skip: int = 60,
        max_new_tokens: int = 512,
        do_sample: bool = False,
    ) -> str:
        """
        Generate text response for video input.
        Args:
            video_path: Path to video file
            text: Question or prompt about the video
            frame_skip: Number of frames to skip between extractions
            max_new_tokens: Max tokens to generate
            do_sample: Whether to use sampling for generation
        Returns:
            Generated text response
        """
        logger.info(f"Processing video: {video_path}")
        
        # Extract frames from video
        frames, total_frames = self.extract_video_frames(video_path, frame_skip)
        logger.info("Preparing model input:")
        logger.info(f"- Number of frames to process: {len(frames)}")
        logger.info(f"- First frame size: {frames[0].size if frames else 'N/A'}")

        # Generate image placeholders dynamically based on number of frames
        image_placeholders = "<image_placeholder>" * len(frames)
        
        # Prepare conversation format with multiple images
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{image_placeholders}\n{text}",
                "images": frames,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        logger.debug("Processing frames through model...")
        prepare_inputs = self.processor(
            conversations=conversation,
            images=frames,
            force_batchify=True
        ).to(self.device)

        logger.debug("Generating response...")
        # Get input embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        # Generate response
        with torch.no_grad():
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_cache=True,
            )

        # Decode and return answer
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

if __name__ == '__main__':
    from PIL import Image

    # Initialize the model
    model = JanusVLM(device="cpu")
    
    # Test video input
    video_path = "../../samples/fishing.mp4"
    input_text = """"What is happening in this video? Describe the action in a single phrase. 
    
    You can think out the action in a chain of thought, but please reply on the final line of your response, a single short phrase (under 5 words).
    
    This action is being 'mimed' meaning backgrounds or objects that are relevant may not be present. Think about only the *action* taking place in the video, and give a response for what it looks like the character is "acting out" or doing "charades" of."""
    
    generated_text = model.generate_text_videoinput(
        video_path,
        input_text,
        frame_skip=30  # Extract 1 frame per second for 30fps video
    )
    logger.info(f"ANSWER: {generated_text}")
    print()
        
            
