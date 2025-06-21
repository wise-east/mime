import json
import jsonlines
import random
from pathlib import Path
from typing import Dict, List, Union

BASE_FEW_SHOT_PROMPT = """
What action is the person doing in this image/video? Choose the most accurate description from the options below.

EXAMPLES:
--------------------------------
Example 1: {example_1_prompt}

A. {example_1_option_a}
B. {example_1_option_b}
C. {example_1_option_c}
D. {example_1_option_d}

{example_1_visual_tokens}

Answer: D
--------------------------------
Example 2: {example_2_prompt}

A. {example_2_option_a}
B. {example_2_option_b}
C. {example_2_option_c}
D. {example_2_option_d}

{example_2_visual_tokens}

Answer: C
--------------------------------
Example 3: {example_3_prompt}

A. {example_3_option_a}
B. {example_3_option_b}
C. {example_3_option_c}
D. {example_3_option_d}

{example_3_visual_tokens}

Answer: B

--------------------------------
END EXAMPLES
--------------------------------
"""

# collect the three videos, example option tesxt, and answers from data/data.jsonl (or any path)
# extract those from three lines of the jsonl, by default it's going to be 0 5 and 10.
# return them as a dict of dicts, ie example_1, example_1.video_path, example_1.option_a, example_1.option_b, example_1.option_c, example_1.option_d, example_1.answer, etc

def get_few_shot_examples(
    dataset_path: Union[str, Path], 
    example_indices: List[int] = [0, 5, 10]
) -> Dict:
    """
    Extract few-shot examples from dataset file.
    
    Args:
        dataset_path: Path to the JSONL dataset file
        example_indices: Which examples to use from the dataset (default: [0, 5, 10])
        
    Returns:
        Dictionary containing the examples formatted for the prompt template
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
    examples = {}
    
    with jsonlines.open(dataset_path) as f:
        # Load all lines into memory
        lines = list(f)
        
        # Ensure indices are valid
        max_idx = len(lines) - 1
        for idx in example_indices:
            if idx > max_idx:
                raise ValueError(f"Example index {idx} exceeds dataset size {max_idx}")
        
        # Process each example
        for i, idx in enumerate(example_indices, 1):
            line = lines[idx]
            
            # Randomly select one correct answer if multiple are provided
            label = random.choice(line['label']) if isinstance(line['label'], list) else line['label']
            
            # Create list of options with correct answer and distractors
            options = line['distractors'] + [label]
            # Shuffle options and keep track of correct answer
            random.shuffle(options)
            correct_idx = options.index(label)
            correct_letter = chr(65 + correct_idx)  # Convert 0,1,2,3 to A,B,C,D
            
            # Store example data
            example_key = f"example_{i}"
            examples[example_key] = {
                'video_path': line['video_path'] if 'video_path' in line else line['s3_url'].split('/')[-1],
                'prompt': "What action is the person doing in this video?",
                'option_a': options[0],
                'option_b': options[1],
                'option_c': options[2],
                'option_d': options[3],
                'answer': correct_letter,
                'visual_tokens': '<image>'  # This will be replaced by model-specific tokens
            }
    
    return examples

def format_few_shot_prompt(examples: Dict, prompt : str = BASE_FEW_SHOT_PROMPT) -> str:
    """
    Format the examples into the BASE_FEW_SHOT_PROMPT template.
    
    Args:
        examples: Dictionary of examples from get_few_shot_examples()
        prompt: Prompt template to use (default: BASE_FEW_SHOT_PROMPT)
    Returns:
        Formatted prompt string
    """
    return prompt.format(
        example_1_prompt=examples['example_1']['prompt'],
        example_1_option_a=examples['example_1']['option_a'],
        example_1_option_b=examples['example_1']['option_b'],
        example_1_option_c=examples['example_1']['option_c'],
        example_1_option_d=examples['example_1']['option_d'],
        example_1_visual_tokens=examples['example_1']['visual_tokens'],
        
        example_2_prompt=examples['example_2']['prompt'],
        example_2_option_a=examples['example_2']['option_a'],
        example_2_option_b=examples['example_2']['option_b'],
        example_2_option_c=examples['example_2']['option_c'],
        example_2_option_d=examples['example_2']['option_d'],
        example_2_visual_tokens=examples['example_2']['visual_tokens'],
        
        example_3_prompt=examples['example_3']['prompt'],
        example_3_option_a=examples['example_3']['option_a'],
        example_3_option_b=examples['example_3']['option_b'],
        example_3_option_c=examples['example_3']['option_c'],
        example_3_option_d=examples['example_3']['option_d'],
        example_3_visual_tokens=examples['example_3']['visual_tokens']
    )






