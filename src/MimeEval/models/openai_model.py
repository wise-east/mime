from typing import List, Any, Union, Tuple, Callable, Dict
import base64
import logging
import time
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from openai import OpenAI, RateLimitError, APIError, APITimeoutError
import os
from .vlms import VLM

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class OpenAIVLM(VLM):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        max_new_tokens: int = 200,
        api_key: str = None,
        max_retries: int = 5,
        initial_delay: float = 1.0,
        **kwargs,
    ):
        super().__init__(model_name, max_new_tokens, **kwargs)
        logger.info(f"Initializing OpenAIVLM with model: {model_name}")
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.image_size = 512
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def _call_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Helper method to call OpenAI API with exponential backoff."""
        delay = self.initial_delay
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RateLimitError, APIError, APITimeoutError) as e:
                last_exception = e
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                logger.warning(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                # For other exceptions, log and re-raise immediately
                logger.error(f"Unexpected error in API call: {str(e)}")
                raise

        # If we've exhausted all retries, raise the last exception
        logger.error(f"API call failed after {self.max_retries} attempts")
        raise last_exception

    def _convert_image_to_base64(self, image: Union[str, Image.Image]) -> str:
        """Convert image to base64 string and resize to standard size."""
        if isinstance(image, str):
            image = Image.open(image)
        
        # Resize image maintaining aspect ratio
        image.thumbnail((self.image_size, self.image_size))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_text_singleimageinput(self, image: Any, text: str) -> str:
        logger.debug("Processing single image input")
        
        # Convert image to base64
        base64_image = self._convert_image_to_base64(image)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ],
            }
        ]

        logger.debug("Sending request to OpenAI API")
        response = self._call_with_retry(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
        )

        return response.choices[0].message.content

    def extract_video_frames(
        self,
        video_path: str,
        frame_skip: int = 30,
    ) -> Tuple[List[Image.Image], int]:
        """Reuse the same frame extraction logic from JanusVLM"""
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info("Video properties:")
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

            if processed_count <= 3:
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
        frame_skip: int = 30,
        max_new_tokens: int = None,
    ) -> str:
        """
        Generate text response for video input using extracted frames.
        """
        logger.info(f"Processing video: {video_path}")
        
        # Extract frames from video
        frames, total_frames = self.extract_video_frames(video_path, frame_skip)
        logger.info(f"Processing {len(frames)} frames")

        # Convert all frames to base64
        base64_frames = [self._convert_image_to_base64(frame) for frame in frames]
        
        # Prepare the message with all frames
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low"
                        }
                    } for frame in base64_frames],
                ],
            }
        ]

        logger.debug("Sending request to OpenAI API")
        response = self._call_with_retry(
            self.client.chat.completions.create,
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens or self.max_new_tokens,
        )

        return response.choices[0].message.content

    def make_few_shot_prompt(self, prompt: str, examples: Dict, mcq: bool = True) -> List:
        """Create messages list for few-shot learning with OpenAI's format"""
        messages = []
        
        # System message to set context
        messages.append({
            "role": "system",
            "content": "You are an expert at analyzing mime actions in videos. You will be shown some example videos with their correct answers, followed by a new video to analyze."
        })
        
        # Add examples
        for i in [1, 2, 3]:
            example = examples[f'example_{i}']
            
            # Extract frames from example video
            frames, _ = self.extract_video_frames(example['video_path'])
            base64_frames = [self._convert_image_to_base64(frame) for frame in frames]
            
            if mcq:
                # Multiple choice format
                question = f"""What action is the person doing in this video? Choose from:
A. {example['option_a']}
B. {example['option_b']}
C. {example['option_c']}
D. {example['option_d']}"""
                answer = example['answer']
            else:
                # Free response format
                question = "What action is the person doing in this video? Describe the action in a single short phrase."
                answer = example['label']
            
            # Add the example frames and prompt
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low"
                        }
                    } for frame in base64_frames]
                ]
            })
            
            # Add the correct answer
            messages.append({
                "role": "assistant",
                "content": answer
            })
        
        return messages

    def predict_mcq(self, media_path, options, eval_type, use_video: bool = True, few_shot_examples: Dict = None):
        if eval_type == "few-shot":
            # Get messages with examples
            messages = self.make_few_shot_prompt(
                f"""What action is the person doing in this video? Choose from:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer with just a single letter (A, B, C, or D).""",
                few_shot_examples
            )
            
            # Extract frames from the test video
            frames, _ = self.extract_video_frames(media_path)
            base64_frames = [self._convert_image_to_base64(frame) for frame in frames]
            
            # Add the test video frames
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""What action is the person doing in this video? Choose from:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer with just a single letter (A, B, C, or D)."""},
                    *[{"type": "image_url", "image_url": f"data:image/jpeg;base64,{frame}"} for frame in base64_frames]
                ]
            })
            
            # Generate response
            response = self._call_with_retry(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
            )
            answer = response.choices[0].message.content.strip()[-1].upper()
            
        else:
            # Use existing zero-shot logic
            answer = super().predict_mcq(media_path, options, eval_type, use_video)
        
        return answer

    def predict_freeform(self, media_path, eval_type, use_video: bool = True, few_shot_examples: Dict = None):
        if eval_type == "few-shot":
            # Get messages with examples
            messages = self.make_few_shot_prompt(
                "What action is the person doing in this video? Describe the action in a single short phrase.",
                few_shot_examples,
                mcq=False
            )
            
            # Extract frames from the test video
            frames, _ = self.extract_video_frames(media_path)
            base64_frames = [self._convert_image_to_base64(frame) for frame in frames]
            
            # Add the test video frames
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "What action is the person doing in this video? Describe the action in a single short phrase."},
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{frame}",
                            "detail": "low"
                        }
                    } for frame in base64_frames]
                ]
            })
            
            # Generate response
            response = self._call_with_retry(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
            )
            answer = response.choices[0].message.content.strip()
            
        else:
            # Use existing zero-shot logic
            answer = super().predict_freeform(media_path, eval_type, use_video)
        
        return answer

if __name__ == '__main__':
    # Test the implementation
    model = OpenAIVLM()
    
    # Test video input
    video_path = "../../samples/fishing.mp4"
    input_text = "What is happening in this video? Describe the action."
    
    generated_text = model.generate_text_videoinput(
        video_path,
        input_text,
        frame_skip=30
    )
    logger.info(f"ANSWER: {generated_text}") 