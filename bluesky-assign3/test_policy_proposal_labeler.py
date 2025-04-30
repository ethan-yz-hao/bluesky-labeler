"""Script for testing the automated hate speech labeler"""

import argparse
import csv
import os
from typing import List, Dict, Any

import pandas as pd
from atproto import Client
from dotenv import load_dotenv

from pylabel.policy_proposal_labeler import HateSpeechDetector, extract_post_data

load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

def main():
    """
    Main function for the test script
    """
    client = Client()
    labeler_client = None
    client.login(USERNAME, PW)

    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=str, help="Path to CSV file with post URLs and expected labels")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file for results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Hate speech probability threshold")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--emit_labels", action="store_true", help="Apply labels to posts")
    args = parser.parse_args()

    # Initialize hate speech detector
    detector = HateSpeechDetector(model_name=args.model)

    # Load input data
    urls_df = pd.read_csv(args.input_csv)
    
    # Prepare results list
    results: List[Dict[str, Any]] = []
    
    num_correct = 0
    total = urls_df.shape[0]
    
    for index, row in urls_df.iterrows():
        url = row["post_url"]
        expected_label = row["label"]
        
        # Extract post data
        post_data = extract_post_data(client, url)
        
        if not post_data:
            print(f"Failed to extract data from post URL: {url}")
            results.append({
                "post_url": url,
                "expected_label": expected_label,
                "predicted_label": "Error",
                "hate_probability": 0.0,
                "correct": False,
                "explanation": "Failed to extract post data"
            })
            continue
        
        # Analyze post
        print(f"Analyzing post {index+1}/{total}: {url}")
        result = detector.predict(post_data, threshold=args.threshold)
        
        # Determine predicted label
        predicted_label = "Hate Speech" if result.get("is_hate_speech", False) else "Not Hate Speech"
        
        # Check if prediction matches expected label
        is_correct = predicted_label == expected_label
        if is_correct:
            num_correct += 1
        else:
            print(f"For {url}, detector produced '{predicted_label}', expected '{expected_label}'")
        
        # Add to results
        results.append({
            "post_url": url,
            "expected_label": expected_label,
            "predicted_label": predicted_label,
            "hate_probability": result.get("hate_probability", 0.0),
            "correct": is_correct,
            "explanation": result.get("explanation", "No explanation provided"),
            "method": result.get("method", "unknown")
        })
    
    # Write results to CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ["post_url", "expected_label", "predicted_label", "hate_probability", 
                      "correct", "explanation", "method"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Print summary
    print(f"The detector produced {num_correct} correct label assignments out of {total}")
    print(f"Overall accuracy: {num_correct/total:.4f}")
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()