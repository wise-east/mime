import json
import logging
import jsonlines
from tqdm import tqdm
from pathlib import Path
from ..models import MODEL2CLASS
import argparse
from datasets import Video, load_dataset

def prepare_few_shot_examples(dataset, example_actions=["Soccer003", "Fishing001", "Violin002"]):
    """Prepare few-shot examples from dataset"""
    examples = {}
    
    for i, action in enumerate(example_actions, 1):
        line = dataset.filter(lambda x: x['action'] == action)[0]
        
        # Get the label (use first one if multiple)
        label = line['label'][0] if isinstance(line['label'], list) else line['label']
        
        # Store example data with full video path
        example_key = f"example_{i}"
        examples[example_key] = {
            'video_path': line['video']['path'],  # Include videos_dir in path
            'prompt': "What action is the person doing in this video?",
            'label': label,  # Store the correct action label
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

    # Prepare few-shot examples if needed
    few_shot_examples = None
    if args.eval_type == 'few-shot':
        example_actions = ["Soccer003", "Fishing001", "Violin002"]
        few_shot_examples = prepare_few_shot_examples(dataset, example_actions)
        # remove those chosen for few shot examples from the dataset
        dataset = dataset.filter(lambda x: x['action'] not in example_actions)
    
    model_outputs = []
    for item in tqdm(dataset):
        instance_id = item['sample_id']
        video_path = str(item['video']['path'])
        output_text, other_auxiliary_outputs = model.predict_freeform(
            video_path,
            args.eval_type,
            few_shot_examples=few_shot_examples if args.eval_type == 'few-shot' else None
        )
        
        model_outputs.append({
            'instance_id': instance_id,
            'output_text': output_text, 
            'video_path': video_path,
            'action': item['label'],
            **other_auxiliary_outputs
        })
    
    # Save results
    logging.info(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(model_outputs, f, indent=4)

    print(f"Processed {len(model_outputs)} videos")
    print("\nTo score these results, run:")
    print(f"mimeeval score --model {args.model} "
                f"--predictions {args.output_file} "
                f"--dataset-name {args.dataset_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run free-form evaluation on mime videos')
    parser.add_argument('--model', type=str, required=True,
                      help='Model type to use for evaluation')
    parser.add_argument('--model-name', type=str,
                      help='Optional model name (e.g. HuggingFace model key)')
    parser.add_argument('--eval_type', type=str, choices=['zero-shot', 'few-shot', 'cot'],
                      help='Evaluation type (zero-shot, few-shot, cot)')
    parser.add_argument('--dataset-name', type=str, required=True,
                      help='Huggingface dataset name')
    parser.add_argument('--output-file', type=Path,
                      help='Path to save model outputs JSON file')
    parser.add_argument("--variant", type=str, default="base+blank@0",
                      help='Variant to run evaluation on (e.g., base+blank@0, adversarial+blank@0, woman+blank@0, base+aligned@0, base+misaligned@0, adversarial+aligned@0, adversarial+misaligned@0, base+blank@90, base+blank@180, base+blank@270) Use "all" to run on all variants. Automatically uses "none" for REAL.')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    parser.add_argument('--api-key', type=str,
                      help='API key for cloud models (can also use environment variables)')
    parser.add_argument('--test', action='store_true',
                      help='Run on a small subset of data for testing.')
    
    args = parser.parse_args()
    main(args)