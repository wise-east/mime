import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from pathlib import Path
from MimeEval.run.score_outputs import compute_similarity, compute_metrics
import click
from loguru import logger

PACKAGE_DIR = Path(__file__).parent.parent.parent.parent

def load_jsonl(file_path):
    data = []
    seen_pairs = set()  # Track (annotator, sample_id) pairs
    
    with open(file_path, 'r') as f:
        for line in f:
            try: 
                item = json.loads(line)
                
                # Extract annotator and sample_id
                annotator = item['url_data']['annotator']
                sample_id = item['question']['sample_id']
                pair = (annotator, sample_id)
                
                # Only add if we haven't seen this pair before
                if pair not in seen_pairs:
                        data.append(item)
                        seen_pairs.add(pair)
            except Exception as e:
                logger.error(f"Error loading line: {line} due to {e}")
                continue
            
    logger.info(f"Loaded {len(data)} unique entries after filtering duplicates")
    return data

def calculate_accuracies_by_groups(fp, similarity_threshold=0.5, cache_file=None, first_only=False):
    
    data = load_jsonl(fp)
    if cache_file is None: 
        cache_file = str(fp).replace(".jsonl", "_similarities_cache.json")
    
    # Initialize the model once
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info(f"Loaded sentence transformer model")
    
    # Try to load cached similarities
    all_similarities = None
    if Path(cache_file).exists():
        logger.info("Loading cached similarities...")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            # Verify cache matches current data
            if len(cached_data) == len(data):
                cache_valid = all(
                    cached_data[i]['output'] == item['user_answer'] and 
                    cached_data[i]['truth'] == item['question']['label']
                    for i, item in enumerate(data)
                )
                if cache_valid:
                    all_similarities = np.array([item['similarity'] for item in cached_data])
                    logger.info("Using cached similarities")
    
    # Compute similarities if not cached or cache invalid
    if all_similarities is None:
        logger.info("Computing similarities...")
        all_similarities = []
        cache_data = []
        
        for item in tqdm(data):
            # Create row structure expected by compute_similarity
            row = {
                'output_text': item['user_answer'],
                'ground_truth': item['question']['label']
            }
            similarity = compute_similarity(row, model)
            all_similarities.append(similarity)
            
            # Store data for caching
            cache_data.append({
                'output': item['user_answer'],
                'truth': item['question']['label'],
                'similarity': float(similarity)  # Convert to float for JSON serialization
            })
        
        all_similarities = np.array(all_similarities)
        
        # Save cache if cache_file specified
        if cache_file:
            logger.info("Saving similarities to cache...")
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
    
    # Compute metrics using imported function
    precision, recall, f1, accuracy = compute_metrics(all_similarities, similarity_threshold)
    
    # Initialize results structure
    results = {
        'mcq': {},
        'free_form': {},
        'annotators': {}  # Add new dictionary for annotator stats
    }
    
    actor_results = {}
    
    # add 'actor' to the data, which is each item's ['question']['s3_url'].split('_')[1]
    for item in data:
        try: 
            item['actor'] = Path(item['question']['s3_url']).stem.split('_')[1]
        except Exception as e:
            item['actor'] = 'unknown' # error handling for the REAL dataset that doesn't have actor in the s3 url
    
    # Process each item
    for idx, item in enumerate(data):
        
        if first_only: 
            if item["question_i"] != 0:
                continue  
        
        # Extract grouping variables
        avatar = item['question'].get('avatar', 'unknown')
        angle = item['question'].get('angle', 'unknown')
        background = item['question'].get('background_config', 'unknown') if 'background_config' in item['question'] else item['question'].get('background', 'unknown')
        annotator = item['url_data']['annotator']  # Extract annotator
        actor = item['actor']
        
        # Initialize annotator stats if needed
        if annotator not in results['annotators']:
            results['annotators'][annotator] = {
                'mcq': {'correct': 0, 'total': 0},
                'free_form': {'correct': 0, 'total': 0, 'similarities': []}
            }
            
        if actor not in actor_results:
            actor_results[actor] = {
                'mcq': {'correct': 0, 'total': 0},
                'free_form': {'correct': 0, 'total': 0, 'similarities': []}
            }
        
        # Initialize nested dictionaries if needed
        for metric in ['mcq', 'free_form']:
            if avatar not in results[metric]:
                results[metric][avatar] = {}
            if angle not in results[metric][avatar]:
                results[metric][avatar][angle] = {}
            if background not in results[metric][avatar][angle]:
                results[metric][avatar][angle][background] = {
                    'correct': 0, 
                    'total': 0,
                    'similarities': []
                }
        
        # Update MCQ counts for both group and annotator
        results['mcq'][avatar][angle][background]['total'] += 1
        results['annotators'][annotator]['mcq']['total'] += 1
        if item['user_is_correct']:
            results['mcq'][avatar][angle][background]['correct'] += 1
            results['annotators'][annotator]['mcq']['correct'] += 1
        
        # Update free-form counts
        results['free_form'][avatar][angle][background]['total'] += 1
        results['annotators'][annotator]['free_form']['total'] += 1
        results['free_form'][avatar][angle][background]['similarities'].append(all_similarities[idx])
        results['annotators'][annotator]['free_form']['similarities'].append(all_similarities[idx])
        
        # Determine correctness based on threshold and distribution
        threshold_negative = np.percentile(all_similarities, 25)
        prediction = all_similarities[idx] >= similarity_threshold
        ground_truth = all_similarities[idx] > threshold_negative
        if prediction == ground_truth:
            results['free_form'][avatar][angle][background]['correct'] += 1
            results['annotators'][annotator]['free_form']['correct'] += 1
    
    
        # calculate accuracy for each actor 
        actor_results[actor]['mcq']['total'] += 1
        actor_results[actor]['free_form']['total'] += 1
        if item['user_is_correct']:
            actor_results[actor]['mcq']['correct'] += 1
        if prediction == ground_truth:
            actor_results[actor]['free_form']['correct'] += 1
    
    # Calculate accuracies for annotators
    for annotator in results['annotators']:
        for metric_type in ['mcq', 'free_form']:
            stats = results['annotators'][annotator][metric_type]
            stats['accuracy'] = round(stats['correct'] / stats['total'] * 100, 2) if stats['total'] > 0 else 0
            
            if metric_type == 'free_form':
                similarities = np.array(stats['similarities'])
                stats['mean_similarity'] = np.mean(similarities)
                stats['median_similarity'] = np.median(similarities)
                stats['std_similarity'] = np.std(similarities)
                del stats['similarities']
    
    # Calculate accuracies and add similarity statistics
    for metric in ['mcq', 'free_form']:
        for avatar in results[metric]:
            for angle in results[metric][avatar]:
                for background in results[metric][avatar][angle]:
                    stats = results[metric][avatar][angle][background]
                    total = stats['total']
                    correct = stats['correct']
                    
                    # Calculate accuracy
                    stats['accuracy'] = round(correct / total * 100, 2) if total > 0 else 0
                    
                    # Add similarity statistics for free-form
                    if metric == 'free_form':
                        similarities = np.array(stats['similarities'])
                        stats['mean_similarity'] = np.mean(similarities)
                        stats['median_similarity'] = np.median(similarities)
                        stats['std_similarity'] = np.std(similarities)
                        del stats['similarities']  # Remove raw similarities after computing stats
    
    results['actor_results'] = actor_results
    
    return results, {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}

def print_results(results):

    # Print grouped results
    for metric in ['mcq', 'free_form']:
        print(f"\n{metric.upper()} Results:")
        print("-" * 100)
        if metric == 'mcq':
            print(f"{'Avatar':<15} {'Angle':<8} {'Background':<15} {'Total':<8} {'Correct':<8} {'Accuracy':<8}")
            print("-" * 100)
        else:
            print(f"{'Avatar':<15} {'Angle':<8} {'Background':<15} {'Total':<8} {'Correct':<8} {'Accuracy':<8} {'Mean Sim':<10} {'Median Sim':<10} {'Std Sim':<10}")
            print("-" * 100)
        
        for avatar in sorted(results[metric].keys()):
            for angle in sorted(results[metric][avatar].keys()):
                for background in sorted(results[metric][avatar][angle].keys()):
                    stats = results[metric][avatar][angle][background]
                    if metric == 'mcq':
                        print(f"{avatar:<15} {angle:<8} {background:<15} "
                              f"{stats['total']:<8} {stats['correct']:<8} "
                              f"{stats['accuracy']}%")
                    else:
                        print(f"{avatar:<15} {angle:<8} {background:<15} "
                              f"{stats['total']:<8} {stats['correct']:<8} "
                              f"{stats['accuracy']}% "
                              f"{stats['mean_similarity']:.3f}     "
                              f"{stats['median_similarity']:.3f}     "
                              f"{stats['std_similarity']:.3f}")

    # Print annotator results
    print("\nIndividual Annotator Performance:")
    print("-" * 100)
    print("MCQ/FF Results:")
    print(f"{'Annotator':<15} {'Total':<8} {'Correct':<8} {'Accuracy':<10} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
    print("-" * 100)
    # change all names to lower case 
    results['annotators'] = {k.lower(): v for k, v in results['annotators'].items()}
    for annotator in sorted(results['annotators'].keys()):
        mcq_stats = results['annotators'][annotator]['mcq']
        ff_stats = results['annotators'][annotator]['free_form']
        mcq_accuracy = f"{mcq_stats['accuracy']}%"
        ff_accuracy = f"{ff_stats['accuracy']}%"
        
        print(f"{annotator:<15} {mcq_stats['total']:>8} {mcq_stats['correct']:>8} {mcq_accuracy:>10} {ff_stats['total']:>8} {ff_stats['correct']:>8} {ff_accuracy:>10}")
    

@click.command()
@click.option("-st", "--similarity-threshold", type=float, default=0.5)
@click.option("-f", "--fp", type=str, default=PACKAGE_DIR / "results" / "mime-cropped" / "human_eval.jsonl")
@click.option("-p", "--should-print-results", type=bool, default=True)
@click.option("-fo", "--first-only", type=bool, default=False, help="Only compute accuracies based on the first samples for each annotator for measuring proxy of effect of label leakage via mcq")
def compute_human_eval_results(similarity_threshold, fp, should_print_results, first_only): 
    
    # Calculate accuracies by groups with caching
    results, overall_metrics = calculate_accuracies_by_groups(
        fp, 
        similarity_threshold=similarity_threshold, 
        first_only=first_only
    )
    
    # Print results
    if should_print_results:
        print_results(results)

    return results 

if __name__ == "__main__":
    compute_human_eval_results()