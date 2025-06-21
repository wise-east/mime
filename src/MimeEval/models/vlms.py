from PIL import Image
import os
import json
from tqdm import tqdm
from typing import List, Any, Dict
from abc import ABC, abstractmethod

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


class VLM(ABC):
    """Base class for Vision-Language Models"""

    def __init__(
        self, 
        model_name: str = "",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        api_key: str = "",
        dataset_path: str = ""
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.api_key = api_key
        self.dataset_path = dataset_path
        pass

    def generate_text_singleimageinput(self, image: Any, text: str) -> str:
        """Generate text from an image and prompt"""
        return ""

    def generate_text_multiimageinput(self, images: List[Any], text: str) -> str:
        """Generate text from multiple images and prompt"""
        return ""

    def generate_text_videoinput(self, video_path: str, text: str) -> str:
        """Generate text from a video and prompt"""
        return ""
    
    # needs to be separately implemented for each model depending on how it builds interleaved multi-image prompts
    def make_few_shot_prompt(self, prompt: str) -> str:
        """Make a few-shot prompt"""
        return ""

    def predict_mcq(self, media_path, options, eval_type, use_video: bool = True, few_shot_examples: Dict = None):
        """Run multiple choice prediction using either image or video generation"""
        answer = ""
        if eval_type == "zero-shot":
            prompt = f"""What action is the person doing in this image/video? Choose the most accurate description from the options below.

A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Respond with just a single letter (A, B, C, or D)."""

            while True:    
                # Use video or image generation based on parameter
                if use_video:
                    output = self.generate_text_videoinput(media_path, prompt)
                else:
                    output = self.generate_text_singleimageinput(media_path, prompt)

                # Extract the model's answer (last letter in the output)
                answer = output.strip().strip('.').upper()
                if answer not in ['A', 'B', 'C', 'D']:
                    if any(answer.startswith(letter) for letter in ['A', 'B', 'C', 'D']):
                        answer = answer[0]
                    elif any(answer.endswith(letter) for letter in ['A', 'B', 'C', 'D']):
                        answer = answer[-1]
                if answer in ['A', 'B', 'C', 'D']:
                    break            

        elif eval_type == 'few-shot':
            if few_shot_examples is None:
                raise ValueError("few_shot_examples must be provided for few-shot evaluation")
            if use_video:
                answer = self.generate_text_videoinput(media_path, self.make_few_shot_prompt(prompt, few_shot_examples))
            else:
                answer = self.generate_text_singleimageinput(media_path, self.make_few_shot_prompt(prompt, few_shot_examples))
        elif eval_type == 'cot':
            prompt = f"""What action is the person doing in this image/video? Choose the most accurate description from the options below.

A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

"""
            prompt += """Carefully think through the answer, by detailing the particular actions and movements that you see the person doing. Your output should contain your explanation, and then on a new line, a single letter corresponding to the answer you choose, with no punctuation. An example response is shown below:
'In the video, the person is moving a single arm back and forth, as if they are swinging a bat. This action is most accurately described by option B.\\nB'"""
                        # Use video or image generation based on parameter
            while True:
                if use_video:
                    output = self.generate_text_videoinput(media_path, prompt)
                else:
                    output = self.generate_text_singleimageinput(media_path, prompt)

                # Extract the model's answer (last letter in the output)
                lines = [line.strip() for line in output.replace('\\n', '\n').split('\n') if line.strip()]
                answer = lines[-1].strip('.').upper()
                if answer not in ['A', 'B', 'C', 'D']:
                    if any(answer.startswith(letter) for letter in ['A', 'B', 'C', 'D']):
                        answer = answer[0]
                    elif any(answer.endswith(letter) for letter in ['A', 'B', 'C', 'D']):
                        answer = answer[-1]
                if answer in ['A', 'B', 'C', 'D']:
                    break

        return answer

    def predict_freeform(self, media_path, eval_type, use_video: bool = True, few_shot_examples: Dict = None):
        """Run free-form prediction using either image or video generation"""

        answer = ""
        other_auxiliary_outputs = {}
        
        if eval_type == "zero-shot":
            prompt = """What action is the person doing in this image/video? Describe the action in a single short phrase (under 5 words). 

This action is being 'mimed' meaning backgrounds or objects that are relevant may not be present. Think about only the *action* taking place in the video, and give a response for what it looks like the character is "acting out" or doing "charades" of. Your answer shpould be a short phrase (under 5 words), with no punctuation or answer prefix such as 'Answer:'."""

            # Use video or image generation based on parameter
            if use_video:
                output = self.generate_text_videoinput(media_path, prompt)
            else:
                output = self.generate_text_singleimageinput(media_path, prompt)
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            answer = lines[-1]
            if "Answer:" in answer:
                answer = answer.replace("Answer:", "").strip()

        elif eval_type == 'few-shot':
            raise NotImplementedError("Few-shot evaluation must be implemented by each model class")

        elif eval_type == 'cot':
            prompt = """What action is the person doing in this image/video? Carefully think through the answer, by detailing the particular actions and movements that you see the person doing. 

This action is being 'mimed' meaning backgrounds or objects that are relevant may not be present. Think about only the *action* taking place in the video, and give a response for what it looks like the character is "acting out" or doing "charades" of. Your output should contain your explanation, and then on a new line, a short phrase (under 5 words) corresponding to your answer, with no punctuation or answer prefix such as 'Answer:'."""

            # Use video or image-based generation based on parameter
            if use_video:
                output = self.generate_text_videoinput(media_path, prompt)
            else:
                output = self.generate_text_singleimageinput(media_path, prompt)
            lines = [line.strip() for line in output.split('\n') if line.strip()]
            chain_of_thought = lines[:-1]
            answer = lines[-1]
            if "Answer:" in answer:
                answer = answer.replace("Answer:", "").strip()
            other_auxiliary_outputs["chain_of_thought"] = chain_of_thought
            
        return answer, other_auxiliary_outputs

class Molmo(VLM):

    def __init__(self, model_name: str, max_new_tokens: int = 200):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )
        self.generation_config = GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>")

    def generate_text_singleimageinput(self, image, text: str) -> str:
        inputs = self.processor.process(
            images=[image],
            text=text
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        output = self.model.generate_from_batch(
            inputs,
            generation_config=self.generation_config,
            tokenizer=self.processor.tokenizer
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0,inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return generated_text  
    

if __name__ == '__main__':
    pass