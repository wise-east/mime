from typing import List, Optional, Union, Tuple, Any, Dict
import torch
import os
from tqdm import tqdm
import logging

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

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

class Qwen2VL(VLM):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        **kwargs,
    ):
        super().__init__(model_name, device, torch_dtype, max_new_tokens, **kwargs)
        logger.info(f"Initializing Qwen2-VL with model: {model_name}")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            #attn_implementation="flash_attention_2",
            torch_dtype="auto", 
            device_map="auto",
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name,
        )
    
    def generate_text_singleimageinput(self, image, text: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

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
        

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def make_few_shot_prompt(self, prompt: str, examples: Dict) -> List:
        """Make a few-shot prompt with interleaved video examples"""
        messages = []
        
        # Add examples
        for i in [1, 2, 3]:
            example = examples[f'example_{i}']
            
            # Add example video and prompt
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": example['video_path'],
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": f"""What action is the person doing in this video? Choose from:
A. {example['option_a']}
B. {example['option_b']}
C. {example['option_c']}
D. {example['option_d']}"""}
                ]
            })
            
            # Add the correct answer
            messages.append({
                "role": "assistant",
                "content": example['answer']
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
            
            # Add the test video and prompt
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": media_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": f"""What action is the person doing in this video? Choose from:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Answer with just a single letter (A, B, C, or D)."""}
                ]
            })
            
            # Process the messages
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            answer = output[0].strip()[-1].upper()
            
        else:
            # Use existing zero-shot logic
            answer = super().predict_mcq(media_path, options, eval_type, use_video)
        
        return answer

if __name__ == "__main__":
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VL(model_name)
    input_text = """"What action is the person doing in this image/video? Describe the action in a single phrase. 
    
    You can think out the action in a chain of thought, but please reply on the final line of your response, a single short phrase (under 5 words).
    
    This action is being 'mimed' meaning backgrounds or objects that are relevant may not be present. Think about only the *action* taking place in the video, and give a response for what it looks like the character is "acting out" or doing "charades" of."""

    images = ["samples/kicking_initial.png", "samples/kicking_final.png"]
    num_correct = 0
    for image_path in images:
        print("IMAGE PATH:", image_path)
        #input_text = "What action is the person in the image doing? Answer in a few words only."
        generated_text = model.generate_text_singleimageinput(image_path, input_text)
        #print("INPUT:", input_text)
        print("Qwen2VL ANSWER:", generated_text)
        if 'kick' in generated_text.lower():
            num_correct += 1
    print("NUM CORRECT:", num_correct)
    print("-"*100)

    for activity in ["fishing", "kicking"]:
        video_path = f"samples/{activity}.mp4"
        print("VIDEO PATH:", video_path)
        generated_text = model.generate_text_videoinput(video_path, input_text)
        print("ANSWER:", generated_text)
    import pdb; pdb.set_trace()