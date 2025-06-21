#import argparse
import argparse
import sys
import os
from pathlib import Path

def add_common_args(parser):
    """Add arguments common to all non-utility commands"""
    parser.add_argument('--model', type=str, required=True,
                      help='Model type to use for evaluation')
    parser.add_argument('--model-name', type=str,
                      help='Optional model name (e.g. HuggingFace model key)')
    parser.add_argument('--eval-type', type=str, choices=['zero-shot', 'few-shot', 'cot'],
                        help='Evaluation type (zero-shot, few-shot, cot)')
    parser.add_argument('--variant', type=str, choices=['all', 'none', 'base+blank@0', 'adversarial+blank@0', 'woman+blank@0', 'base+aligned@0', 'base+misaligned@0', 'adversarial+aligned@0', 'adversarial+misaligned@0', 'base+blank@90', 'base+blank@180', 'base+blank@270'],
                        help='Variant to use for evaluation (default: base+blank@0)')
    parser.add_argument('--dataset-name', type=str, required=True,
                      help='Huggingface dataset name (e.g., wise-east/mime)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')

def get_default_output_path(command_type, dataset_name, variant, eval_type, model, model_name=None):
    """Generate default output path based on command type and model"""
    results_dir = Path(f"results")
    results_dir.mkdir(exist_ok=True)
    
    model_suffix = f"{model}"
    if model_name:
        model_suffix += f"_{model_name}"
    
    if command_type == 'mcq':
        return {
            'raw': results_dir / dataset_name / eval_type / variant / f"raw_{command_type}_{model_suffix}.json",
            'score': results_dir / dataset_name / eval_type / variant / f"score_{command_type}_{model_suffix}.csv"
        }
    else:  # freeform
        return {
            'raw': results_dir / dataset_name / eval_type / variant / f"raw_{command_type}_{model_suffix}.json"
        }


def create_parser():
    
    parser = argparse.ArgumentParser(
        description='MimeEval: Tools for evaluating mime video understanding',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Utils command and its subcommands
    utils_parser = subparsers.add_parser('utils', help='Utility functions')
    utils_subparsers = utils_parser.add_subparsers(dest='utils_type', required=True)
    
    # Crop subcommand under utils
    crop_parser = utils_subparsers.add_parser('crop', help='Crop videos to standard format')
    crop_parser.add_argument('--input-dir', type=Path,
                           help='Directory containing input videos')
    crop_parser.add_argument('--output-dir', type=Path,
                           help='Directory to save cropped videos')
    crop_parser.add_argument('--dataset-name', type=str, required=True,
                           help='Huggingface dataset name')
    
    # Run command and its subcommands
    run_parser = subparsers.add_parser('run', help='Run model evaluation')
    run_subparsers = run_parser.add_subparsers(dest='run_type', required=True)
    
    # Common arguments for run subcommands
    run_common_args = {
        '--api-key': {
            'type': str,
            'required': False,
            'help': 'API key for cloud models (can also use environment variables)'
        },
        '--output-file': {
            'type': Path,
            'required': False,
            'help': 'Path to save model outputs JSON file (default: results/raw_[type]_[model].json)'
        },
        '--test': {
            'action': 'store_true',
            'help': 'Run on a small subset of data for testing.'
        }
    }
    
    # MCQ subcommand
    mcq_parser = run_subparsers.add_parser('mcq', help='Run multiple choice evaluation')
    add_common_args(mcq_parser)
    for arg, kwargs in run_common_args.items():
        mcq_parser.add_argument(arg, **kwargs)
    mcq_parser.add_argument('--score-file', type=Path,
                         help='Path to save score CSV file (default: results/score_mcq_[model].csv)')
    
    # Free-form subcommand
    ff_parser = run_subparsers.add_parser('ff', help='Run free-form evaluation')
    add_common_args(ff_parser)
    for arg, kwargs in run_common_args.items():
        ff_parser.add_argument(arg, **kwargs)
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score model outputs')
    score_parser.add_argument('--predictions', type=Path, required=True,
                           help='Path to predictions JSON file')
    score_parser.add_argument('--dataset-name', type=str, required=True,
                           help='Huggingface dataset name (e.g., wise-east/mime)')
    score_parser.add_argument('--threshold', type=float, default=0.75,
                           help='Similarity threshold above which predictions are considered correct')
    score_parser.add_argument('--output', type=Path, default=None,
                           help='Path to save detailed results CSV (optional)')
    score_parser.add_argument('--verbose', action='store_true',
                           help='Enable verbose logging')
    
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    # Set default output paths if not provided
    if args.command == 'run':
        default_paths = get_default_output_path(
            args.run_type, 
            args.dataset_name,
            args.variant,
            args.eval_type,
            args.model, 
            args.model_name.replace('/', '_') if args.model_name else None,
        )
        if not args.output_file:
            args.output_file = default_paths['raw']
        if args.run_type == 'mcq' and not args.score_file:
            args.score_file = default_paths['score']

    if args.command == 'utils':
        if args.utils_type == 'crop':
            from MimeEval.utils.crop_videos import main as crop_main
            crop_main(args)
    elif args.command == 'run':
        if args.run_type == 'mcq':
            from MimeEval.run.mimevideo_mcq import main as mcq_main
            mcq_main(args)
        elif args.run_type == 'ff':
            from MimeEval.run.mimevideo_freeform import main as ff_main
            ff_main(args)
    elif args.command == 'score':
        from MimeEval.run.score_outputs import main as score_main
        score_main(args)

if __name__ == '__main__':
    main() 