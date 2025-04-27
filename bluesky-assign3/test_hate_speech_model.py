#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import json
from typing import List, Dict, Tuple, Union
import pandas as pd

def load_model(model_name: str = "Hate-speech-CNERG/dehatebert-mono-english"):
    """
    Load the hate speech detection model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def predict_hate_speech(texts: List[str], tokenizer, model, batch_size: int = 8) -> List[Dict[str, float]]:
    """
    Predict hate speech probability for a list of texts.
    
    Args:
        texts: List of text strings to classify
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        batch_size: Batch size for processing
        
    Returns:
        List of dictionaries with hate speech probabilities
    """
    results = []
    model.eval()
    
    # Process in batches to avoid OOM errors
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize and prepare inputs
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract probabilities
        for j, prob in enumerate(probs):
            # For binary classification: [not_hate, hate]
            hate_prob = prob[1].item()
            results.append({
                "text": batch_texts[j],
                "hate_probability": hate_prob,
                "is_hate_speech": hate_prob > 0.5  # Using 0.5 as default threshold
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test hate speech detection model')
    parser.add_argument('--input', type=str, help='Input file (JSON or CSV)')
    parser.add_argument('--text-field', type=str, default='text', help='Field name containing text in input file')
    parser.add_argument('--output', type=str, default='hate_speech_results.json', help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for hate speech')
    parser.add_argument('--sample-text', action='store_true', help='Test with sample texts instead of file')
    args = parser.parse_args()
    
    # Load model and tokenizer
    tokenizer, model = load_model()
    
    if args.sample_text:
        # Test with sample texts
        sample_texts = [
            "I love all people regardless of their background.",
            "You are stupid and worthless because of your race.",
            "This is a neutral statement about the weather.",
            "I hate people like you, you don't deserve to live.",
            "Everyone deserves equal treatment and respect."
        ]
        
        results = predict_hate_speech(sample_texts, tokenizer, model)
        
        # Print results
        print("\nSample Text Results:")
        print("-" * 80)
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Hate Probability: {result['hate_probability']:.4f}")
            print(f"Classification: {'HATE SPEECH' if result['is_hate_speech'] else 'NOT HATE SPEECH'}")
            print("-" * 80)
            
        # Save results
        with open("sample_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Sample results saved to sample_results.json")
        
    elif args.input:
        # Load input data
        if args.input.endswith('.json'):
            with open(args.input, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    texts = [item.get(args.text_field, "") for item in data]
                else:
                    texts = [data.get(args.text_field, "")]
        elif args.input.endswith('.csv'):
            df = pd.read_csv(args.input)
            texts = df[args.text_field].tolist()
        else:
            raise ValueError("Input file must be JSON or CSV")
        
        # Process texts
        results = predict_hate_speech(texts, tokenizer, model)
        
        # Update results with custom threshold
        if args.threshold != 0.5:
            for result in results:
                result["is_hate_speech"] = result["hate_probability"] > args.threshold
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Processed {len(results)} texts")
        print(f"Results saved to {args.output}")
        
        # Print summary
        hate_count = sum(1 for r in results if r["is_hate_speech"])
        print(f"Hate speech detected: {hate_count}/{len(results)} ({hate_count/len(results)*100:.1f}%)")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 