import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import jsonlines
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from datasets import load_dataset

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def compute_similarity(row, model):
    #print(row["output_text"])
    #print(row["ground_truth"][0])
    pred_embedding = model.encode(row['output_text'], convert_to_tensor=True)
    similarities = []
    for label in row['ground_truth']:
        gt_embedding = model.encode(label, convert_to_tensor=True)
        # Compute cosine similarity
        similarity = util.pytorch_cos_sim(pred_embedding, gt_embedding).item()
        similarities.append(similarity)
    return max(similarities)

def compute_metrics(similarities, threshold):
    # Convert similarities to binary predictions based on threshold
    predictions = (similarities >= threshold).astype(int)
    # Return -1 for precision, recall, and f1 score as we don't have a ground truth
    # Return the mean of predictions as the accuracy
    return -1, -1, -1, np.mean(predictions)

def get_default_output_path(predictions_path: Path) -> Path:
    """Generate default output path from predictions path"""
    # Replace 'raw' with 'score' in the filename
    filename = predictions_path.name.replace('raw_', 'score_').replace('.json', '.csv')
    return predictions_path.parent / filename

def main(args):
    # Set default output path if not provided
    if args.output is None:
        args.output = get_default_output_path(args.predictions)

    # Load the model
    print(f"Loading model {SENTENCE_TRANSFORMER_MODEL}...")
    model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

    # Load the output predictions
    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, 'r') as f:
        predictions = json.load(f)
    
    # For MCQ format, predictions might be nested
    if isinstance(predictions, dict) and 'predictions' in predictions:
        predictions = predictions['predictions']

    # Load ground truth data
    print(f"Loading ground truth from {args.dataset_name}...")
    data = load_dataset(args.dataset_name)
    ground_truth = data['test']['label']

    # Create DataFrames
    pred_df = pd.DataFrame(predictions)
    gt_df = pd.DataFrame(ground_truth)

    # Ensure we have matching instance_ids
    assert len(pred_df) == len(gt_df), "Prediction and ground truth lengths don't match"

    # Add ground truth column to predictions
    pred_df['ground_truth'] = gt_df['label']
    print(pred_df.head())
    
    # Calculate similarities
    print("Computing similarities...")
    similarities = []
    for _, row in pred_df.iterrows():
        similarity = compute_similarity(row, model)
        similarities.append(similarity)
    pred_df['similarity_score'] = similarities

    # Calculate statistics
    mean_similarity = pred_df['similarity_score'].mean()
    std_similarity = pred_df['similarity_score'].std()
    median_similarity = pred_df['similarity_score'].median()

    # Calculate metrics using threshold
    precision, recall, f1, accuracy = compute_metrics(pred_df['similarity_score'].values, args.threshold)

    # Count predictions above threshold
    correct_count = (pred_df['similarity_score'] >= args.threshold).sum()
    total_count = len(pred_df)
    print(f"\nPredictions above threshold: {correct_count}/{total_count} ({(correct_count/total_count)*100:.1f}%)")

    # Save detailed results if output path provided
    if args.output:
        results_df = pred_df[['instance_id', 'output_text', 'ground_truth', 'similarity_score']]
        results_df['correct'] = results_df['similarity_score'] >= args.threshold
        results_df.to_csv(args.output, index=False)
        print(f"\nDetailed results saved to {args.output}")

    # Print worst and best matches
    print("\nWorst Matches:")
    print(pred_df.nsmallest(5, 'similarity_score')[['output_text', 'ground_truth', 'similarity_score']])

    print("\nBest Matches:")
    print(pred_df.nlargest(5, 'similarity_score')[['output_text', 'ground_truth', 'similarity_score']])

    print("\nResults:")
    print(f"Mean Similarity Score: {mean_similarity:.3f}")
    print(f"Median Similarity Score: {median_similarity:.3f}")
    print(f"Standard Deviation: {std_similarity:.3f}")
    print(f"\nMetrics (threshold = {args.threshold}):")
    print(f"Accuracy: {accuracy:.3f}")