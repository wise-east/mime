import json
import random
import logging
import jsonlines
from tqdm import tqdm
from ..models import MODEL2CLASS
import argparse
from pathlib import Path
from datasets import Video, load_dataset

def prepare_few_shot_examples(dataset, example_actions=["Soccer003", "Fishing001", "Violin002"]):
    """Prepare few-shot examples from dataset"""
    examples = {}
    
    for i, action in enumerate(example_actions, 1):
        line = dataset.filter(lambda x: x['action'] == action)[0]
        
        # Randomly select one correct answer if multiple are provided
        label = random.choice(line['label']) if isinstance(line['label'], list) else line['label']
        
        # Create list of options with correct answer and distractors
        options = line['distractors'].split('|') + [label]
        random.shuffle(options)
        correct_idx = options.index(label)  
        correct_letter = chr(65 + correct_idx)  # Convert 0,1,2,3 to A,B,C,D
        
        # Store example data with full video path
        example_key = f"example_{i}"
        examples[example_key] = {
            'video_path': line['video']['path'],  # Include videos_dir in path
            'prompt': "What action is the person doing in this video?",
            'option_a': options[0],
            'option_b': options[1],
            'option_c': options[2],
            'option_d': options[3],
            'answer': correct_letter,
            'sample_id': line['sample_id']
        }
    
    return examples

def main(args):
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Load model
    logging.info(f"Loading model {args.model}...")
    model_class = MODEL2CLASS[args.model]
    model = model_class(model_name=args.model_name, api_key=args.api_key)

    # Load HuggingFace dataset
    logging.info(f"Loading dataset from {args.dataset_name}...")
    dataset = load_dataset(args.dataset_name, split="test")
    
    # automatically set variant to "none" for REAL dataset
    if "real" in args.dataset_name:
        args.variant = "none"
    
    if args.variant != "all":
        dataset = dataset.filter(lambda x: x['config_name'] == args.variant)
    dataset = dataset.cast_column("video", Video(decode=False))

    if args.test:
        logging.info("Running on a small subset of data for testing...")
        dataset = dataset.select(range(5))

    logging.info(f"Dataset size: {len(dataset)} samples")

    # Prepare few-shot examples if needed
    few_shot_examples = None
    if args.eval_type == 'few-shot':
        example_actions = ["Soccer003", "Fishing001", "Violin002"]
        few_shot_examples = prepare_few_shot_examples(dataset, example_actions)
        # remove those chosen for few shot examples from the dataset
        dataset = dataset.filter(lambda x: x['action'] not in example_actions)

    model_outputs = []
    correct_count = 0
    
    for item in tqdm(dataset):
        instance_id = item['sample_id']
        video_path = str(item['video']['path'])
        logging.info(f"Predicting for video {video_path}")

        label = item['label']
        # Create list of all options and shuffle them
        options = item['distractors'].split('|') + [label]
        random.shuffle(options)

        # Find the correct answer index
        correct_answer_idx = options.index(label)
        correct_letter = chr(65 + correct_answer_idx)  # Convert 0,1,2,3 to A,B,C,D

        # Get model's answer - convert Path to string
        model_answer = model.predict_mcq(
            video_path, 
            options, 
            args.eval_type,
            few_shot_examples=few_shot_examples if args.eval_type == 'few-shot' else None
        )
        is_correct = model_answer == correct_letter

        if is_correct:
            correct_count += 1

        if args.verbose:
            status = "✓" if is_correct else "✗"
            logging.info(f"Options: A: {options[0]}, B: {options[1]}, C: {options[2]}, D: {options[3]}")
            logging.info(f"Video {item['video']['path']} - Model answer: {model_answer}, Correct answer: {correct_letter} {status}")
            
        model_outputs.append({
            'instance_id': instance_id,
            'model_answer': model_answer,
            'correct_answer': correct_letter,
            'is_correct': is_correct,
            'video_path': video_path,
            'options': options
        })
    
    # Calculate final accuracy
    accuracy = correct_count / len(dataset)
    
    # Add accuracy to the output
    final_output = {
        'accuracy': accuracy,
        'predictions': model_outputs
    }
    
    # Save results
    logging.info(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(final_output, f, indent=4)
    
    # Save binary scores to CSV
    if args.score_file:
        logging.info(f"Saving scores to {args.score_file}...")
        with open(args.score_file, 'w') as f:
            f.write("instance_id,is_correct\n")
            for output in model_outputs:
                f.write(f"{output['instance_id']},{1 if output['is_correct'] else 0}\n")
    
    logging.info(f"Evaluation complete!")
    print(f"\nFinal Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run multiple choice evaluation on mime videos')
    parser.add_argument('--model', type=str, required=True,
                      help='Model type to use for evaluation')
    parser.add_argument('--model-name', type=str,
                      help='Optional model name (e.g. HuggingFace model key)')
    parser.add_argument('--eval_type', type=str, choices=['zero-shot', 'few-shot', 'cot'],
                      help='Evaluation type (zero-shot, few-shot, cot)')
    parser.add_argument('--dataset-name', type=str, required=True,
                      help='Huggingface dataset name')
    parser.add_argument("--variant", type=str, default="base+blank@0",
                      help='Variant to run evaluation on (e.g., base+blank@0, adversarial+blank@0, woman+blank@0, base+aligned@0, base+misaligned@0, adversarial+aligned@0, adversarial+misaligned@0, base+blank@90, base+blank@180, base+blank@270) Use "all" to run on all variants. Automatically uses "none" for REAL.')
    parser.add_argument('--output-file', type=Path,
                      help='Path to save model outputs JSON file. Defaults to results/[dataset_name]/raw_mcq_[model].json')
    parser.add_argument('--score-file', type=Path,
                         help='Path to save score CSV file. Defaults to results/[dataset_name]/score_mcq_[model].csv')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--api-key', type=str, default=None,
                         help='API key for cloud models (can also use environment variables)')
    parser.add_argument('--test', action='store_true',
                      help='Run on a small subset of data for testing.')
    
    args = parser.parse_args()
    main(args)