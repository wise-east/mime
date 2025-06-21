from typing import List, Any, Union, Tuple, Callable, Dict
import google.generativeai as genai
import logging
from pathlib import Path
from ..models.vlms import VLM
from ..utils.retry import exponential_backoff
from ..models.fewshotutils import get_few_shot_examples, format_few_shot_prompt

logger = logging.getLogger(__name__)

class GeminiVLM(VLM):
    """Gemini Pro Vision model implementation"""
    
    def __init__(
        self,
        model_name: str = "gemini-pro-vision",
        api_key: str = None,
        max_new_tokens: int = 512,
        **kwargs
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens, **kwargs)
        
        # Configure API
        if api_key is None:
            # Try to get from environment
            import os
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key is None:
                raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable or pass api_key parameter")
        
        genai.configure(api_key=api_key)
        
        # Initialize model
        logger.info(f"Initializing Gemini with model: {model_name}")
        self.model = genai.GenerativeModel(model_name)
    
    def make_few_shot_prompt(self, prompt: str, examples: Dict, mcq: bool = True) -> List:
        """Make a few-shot prompt with interleaved video examples"""
        contents = []
        
        # Add examples
        for i in [1, 2, 3]:
            example = examples[f'example_{i}']
            with open(example['video_path'], 'rb') as f:
                video_data = f.read()
                
            if mcq:
                # Multiple choice format
                contents.extend([
                    {"mime_type": "video/mp4", "data": video_data},
                    f"""What action is the person doing in this video? Choose from:
A. {example['option_a']}
B. {example['option_b']}
C. {example['option_c']}
D. {example['option_d']}

Answer: {example['answer']}
""",
                ])
            else:
                # Free response format
                contents.extend([
                    {"mime_type": "video/mp4", "data": video_data},
                    f"""What action is the person doing in this video? Describe the action in a single short phrase.

Answer: {example['label']}
""",
                ])
        
        # Add the actual prompt at the end
        contents.append(prompt)
        
        return contents

    @exponential_backoff(max_delay=64, backoff_factor=2,max_attempts=10)
    def wrapped_generate_content(self, contents, generation_config=None):
        return self.model.generate_content(contents, generation_config=generation_config)

    def generate_text_videoinput(self, video_path: str, text: str) -> str:
        """Generate text from video using Gemini's native video support"""
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Read video file as bytes
        with open(video_path, 'rb') as f:
            video_data = f.read()
            
        # Generate response
        response = self.wrapped_generate_content(
            [
                text,  # The prompt
                {"mime_type": "video/mp4", "data": video_data}  # The video
            ],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_new_tokens,
            )
        )
        
        return response.text

    def generate_text_singleimageinput(self, image, text: str) -> str:
        """Generate text from image using Gemini"""
        # Handle both file paths and PIL Images
        if isinstance(image, str) or isinstance(image, Path):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            with open(image_path, 'rb') as f:
                image_data = f.read()
        else:
            # Convert PIL Image to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()
            
        # Generate response
        response = self.wrapped_generate_content(
            [
                text,  # The prompt
                {"mime_type": "image/png", "data": image_data}  # The image
            ],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=self.max_new_tokens,
            )
        )
        
        return response.text

    def predict_mcq(self, media_path, options, eval_type, use_video: bool = True, few_shot_examples: Dict = None):
        if eval_type == "few-shot":
            # Get few-shot prompt with examples
            contents = self.make_few_shot_prompt(
                f"""What action is the person doing in this video? Choose from:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer with just a single letter (A, B, C, or D).""",
                few_shot_examples
            )
            
            # Add the test video
            with open(media_path, 'rb') as f:
                video_data = f.read()
            contents.append({"mime_type": "video/mp4", "data": video_data})
            
            # Generate response
            response = self.wrapped_generate_content(contents)
            answer = response.text.strip()[-1].upper()
            
        else:
            # Use existing zero-shot logic
            answer = super().predict_mcq(media_path, options, eval_type, use_video)
        
        return answer

    def predict_freeform(self, media_path, eval_type, use_video: bool = True, few_shot_examples: Dict = None):
        if eval_type == "few-shot":
            # Get messages with examples
            contents = self.make_few_shot_prompt(
                "What action is the person doing in this video? Describe the action in a single short phrase.",
                few_shot_examples,
                mcq=False
            )
            
            # Add the test video
            with open(media_path, 'rb') as f:
                video_data = f.read()
            contents.append({"mime_type": "video/mp4", "data": video_data})
            
            # Generate response
            response = self.wrapped_generate_content(contents)
            answer = response.text.strip()
            
        else:
            # Use existing zero-shot logic
            answer = super().predict_freeform(media_path, eval_type, use_video)
        
        return answer

# Add to model registry
from ..models import MODEL2CLASS
MODEL2CLASS['gemini'] = GeminiVLM 