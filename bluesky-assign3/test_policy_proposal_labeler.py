"""Script for testing the automated job scam labeler using labeled money posts data"""

import argparse
import csv
import os
from typing import List, Dict, Any

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from pylabel.policy_proposal_labeler import PolicyProposalLabeler

def main():
    """
    Main function for the test script
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="Path to CSV file with labeled posts (labeled_money_posts.csv)")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file for results")
    parser.add_argument("errors_csv", type=str, help="Path to output CSV file for incorrectly labeled posts")
 
    args = parser.parse_args()

    # Initialize job scam detector
    detector = PolicyProposalLabeler()

    # Load input data
    posts_df = pd.read_csv(args.input_csv)
    
    # Prepare results list
    results: List[Dict[str, Any]] = []
    
    # Lists to store true and predicted labels for metrics calculation
    y_true = []
    y_pred = []
    
    # Lists to store incorrectly labeled posts for debugging
    incorrect_predictions = []
    
    num_correct = 0
    total = posts_df.shape[0]
    
    for index, row in posts_df.iterrows():
        # Extract data from the row
        post_id = row["post_id"]
        text = row["text"]
        creator = row.get("creator", "unknown")
        expected_label = row["is_scam"]  # "scam" or "not_scam"
        
        # Create post data dictionary from the CSV row
        post_data = {
            'uri': f"at://{creator}/app.bsky.feed.post/{post_id}",
            'text': text,
            'image_urls': [],  # No images in this test
            'url': f"https://bsky.app/profile/{creator}/post/{post_id}",
            'author': creator,
            'hashtag': row.get("hashtag", "")
        }
        
        # Analyze post
        print(f"Analyzing post {index+1}/{total}: {post_id}")
        result = detector.predict(post_data)
        
        # Determine predicted label (convert to match the format in the CSV)
        predicted_label = result.get("is_job_scam")
        
        # Check if prediction matches expected label
        is_correct = predicted_label == expected_label
        if is_correct:
            num_correct += 1
        else:
            print(f"For post {post_id}, detector produced '{predicted_label}', expected '{expected_label}'")
            print(f"Text: {text[:100]}...")
            # Add to incorrect predictions list
            incorrect_predictions.append({
                "post_id": post_id,
                "text": text,
                "creator": creator,
                "expected_label": expected_label,
                "predicted_label": predicted_label,
                "explanation": result.get("explanation", "No explanation provided"),
                "method": result.get("method", "unknown"),
                "confidence": row.get("confidence", 0),
                "url": f"https://bsky.app/profile/{creator}/post/{post_id}"
            })
        
        # Add to results
        results.append({
            "post_id": post_id,
            "text": text[:50] + "..." if len(text) > 50 else text,  # Truncate long text for readability
            "creator": creator,
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "correct": is_correct,
            "explanation": result.get("explanation", "No explanation provided"),
            "method": result.get("method", "unknown"),
            "confidence": row.get("confidence", 0)  # Include original confidence if available
        })
        
        # Store labels for metrics calculation (convert string labels to binary)
        y_true.append(1 if expected_label == "scam" else 0)
        y_pred.append(1 if predicted_label == "scam" else 0)
    
    # Write results to CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["post_id", "text", "creator", "expected_label", "predicted_label", 
                      "correct", "explanation", "method", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Write incorrect predictions to a separate CSV if specified
    if args.errors_csv and incorrect_predictions:
        with open(args.errors_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ["post_id", "text", "creator", "expected_label", "predicted_label", 
                         "explanation", "method", "confidence", "url"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in incorrect_predictions:
                writer.writerow(result)
        print(f"Saved {len(incorrect_predictions)} incorrectly labeled posts to {args.errors_csv}")
    
    # Calculate metrics
    accuracy = num_correct / total if total > 0 else 0
    
    # Calculate confusion matrix
    if len(y_true) > 0 and len(y_pred) > 0:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Calculate precision, recall, and F1 score
        precision = precision_score(y_true, y_pred) if sum(y_pred) > 0 else 0
        recall = recall_score(y_true, y_pred) if sum(y_true) > 0 else 0
        f1 = f1_score(y_true, y_pred) if (precision + recall) > 0 else 0
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print("                  Predicted   Predicted")
        print("                  Not Scam    Scam")
        print(f"Actual Not Scam    {tn}          {fp}")
        print(f"Actual Scam         {fn}          {tp}")
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
    else:
        print("\nNot enough data to calculate metrics")
    
    # Print summary
    print(f"\nThe detector produced {num_correct} correct label assignments out of {total}")
    print(f"Overall accuracy: {accuracy:.4f}")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()