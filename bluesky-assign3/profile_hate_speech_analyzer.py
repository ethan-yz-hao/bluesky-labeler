#!/usr/bin/env python3
import os
import time
import argparse
import json
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from atproto import Client
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

class HateSpeechDetector:
    def __init__(self, model_name: str = "Hate-speech-CNERG/dehatebert-mono-english"):
        """Initialize the hate speech detector with the specified model."""
        print(f"Loading hate speech model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        print(f"Model loaded and running on {self.device}")
    
    def predict(self, texts: List[str], threshold: float = 0.5, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Predict hate speech probability for a list of texts.
        
        Args:
            texts: List of text strings to classify
            threshold: Probability threshold for classifying as hate speech
            batch_size: Batch size for processing
            
        Returns:
            List of dictionaries with hate speech probabilities and classifications
        """
        results = []
        self.model.eval()
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and prepare inputs
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Extract probabilities
            for j, prob in enumerate(probs):
                hate_prob = prob[1].item()
                results.append({
                    "text": batch_texts[j],
                    "hate_probability": hate_prob,
                    "is_hate_speech": hate_prob > threshold
                })
        
        return results

def fetch_profile_posts(client: Client, handle: str, limit: int = 100) -> List[Dict]:
    """Fetch posts from a specific profile."""
    print(f"Fetching up to {limit} posts from @{handle}...")
    
    try:
        # Resolve handle to DID
        resp = client.com.atproto.identity.resolve_handle({"handle": handle})
        did = resp.did
        
        # Fetch author feed
        feed = client.app.bsky.feed.get_author_feed({"actor": did, "limit": limit})
        
        posts = []
        for item in feed.feed:
            post = item.post
            text = getattr(post.record, 'text', '')
            
            # Include external link if present
            if getattr(post, 'embed', None) and getattr(post.embed, 'external', None):
                text += ' ' + post.embed.external.uri
            
            posts.append({
                'uri': post.uri,
                'cid': post.cid,
                'author': post.author.handle,
                'text': text,
                'url': f"https://bsky.app/profile/{post.author.handle}/post/{post.uri.split('/')[-1]}"
            })
        
        print(f"Fetched {len(posts)} posts from @{handle}")
        return posts
    
    except Exception as e:
        print(f"Error fetching posts for {handle}: {e}")
        return []

def analyze_profile(
    client: Client,
    detector: HateSpeechDetector,
    handle: str,
    post_limit: int = 50,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze a profile for hate speech content.
    
    Args:
        client: ATProto client
        detector: Hate speech detector
        handle: Bluesky handle to analyze
        post_limit: Maximum number of posts to analyze
        threshold: Probability threshold for classifying as hate speech
        
    Returns:
        Dictionary with analysis results
    """
    # Fetch posts
    posts = fetch_profile_posts(client, handle, limit=post_limit)
    
    if not posts:
        return {
            "handle": handle,
            "posts_analyzed": 0,
            "hate_speech_count": 0,
            "hate_speech_ratio": 0.0,
            "hate_speech_posts": [],
            "analysis_complete": False,
            "error": "Failed to fetch posts"
        }
    
    # Extract text for analysis
    texts = [post["text"] for post in posts]
    
    # Analyze texts
    print(f"Analyzing {len(texts)} posts for hate speech...")
    results = detector.predict(texts, threshold=threshold)
    
    # Match results back to posts
    hate_speech_posts = []
    hate_speech_count = 0
    
    for i, result in enumerate(results):
        if result["is_hate_speech"]:
            hate_speech_count += 1
            hate_speech_posts.append({
                **posts[i],
                "hate_probability": result["hate_probability"]
            })
    
    # Compile results
    hate_speech_ratio = hate_speech_count / len(posts) if posts else 0
    
    return {
        "handle": handle,
        "posts_analyzed": len(posts),
        "hate_speech_count": hate_speech_count,
        "hate_speech_ratio": hate_speech_ratio,
        "hate_speech_threshold": threshold,
        "hate_speech_posts": hate_speech_posts,
        "analysis_complete": True
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze Bluesky profiles for hate speech')
    parser.add_argument('--handles', type=str, nargs='+', help='Bluesky handles to analyze')
    parser.add_argument('--handles-file', type=str, help='File with Bluesky handles (one per line)')
    parser.add_argument('--post-limit', type=int, default=50, help='Maximum posts to analyze per profile')
    parser.add_argument('--threshold', type=float, default=0.5, help='Hate speech probability threshold')
    parser.add_argument('--output', type=str, default='profile_analysis.json', help='Output JSON file')
    args = parser.parse_args()
    
    # Load handles
    handles = []
    if args.handles:
        handles.extend(args.handles)
    
    if args.handles_file:
        with open(args.handles_file, 'r') as f:
            file_handles = [line.strip() for line in f if line.strip()]
            handles.extend(file_handles)
    
    if not handles:
        parser.error("No handles provided. Use --handles or --handles-file")
    
    # Initialize client
    client = Client()
    client.login(USERNAME, PW)
    
    # Initialize hate speech detector
    detector = HateSpeechDetector()
    
    # Analyze profiles
    results = []
    for handle in handles:
        print(f"\nAnalyzing profile: @{handle}")
        profile_result = analyze_profile(
            client, 
            detector, 
            handle, 
            post_limit=args.post_limit,
            threshold=args.threshold
        )
        results.append(profile_result)
        
        # Print summary
        if profile_result["analysis_complete"]:
            print(f"Results for @{handle}:")
            print(f"  Posts analyzed: {profile_result['posts_analyzed']}")
            print(f"  Hate speech posts: {profile_result['hate_speech_count']} " +
                  f"({profile_result['hate_speech_ratio']*100:.1f}%)")
        else:
            print(f"Analysis failed for @{handle}: {profile_result.get('error', 'Unknown error')}")
        
        # Avoid rate limiting
        time.sleep(1)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {args.output}")
    
    # Print overall summary
    analyzed_count = sum(1 for r in results if r["analysis_complete"])
    hate_profiles = sum(1 for r in results if r["analysis_complete"] and r["hate_speech_ratio"] > 0.1)
    
    print(f"\nOverall Summary:")
    print(f"  Profiles analyzed: {analyzed_count}/{len(handles)}")
    print(f"  Profiles with >10% hate speech: {hate_profiles}/{analyzed_count}")

if __name__ == '__main__':
    main() 