#!/usr/bin/env python3
import os
import argparse
import json
from atproto import Client
from dotenv import load_dotenv
from profile_hate_speech_analyzer import HateSpeechDetector
from label import post_from_url

# Load credentials from .env
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

def extract_post_data(client, url):
    """Extract a post from a Bluesky URL"""
    try:
        # Use the helper function from label.py to get the post
        post = post_from_url(client, url)
        print(post)
        
        # Extract text
        text = getattr(post.value, 'text', '')
        
        # Extract image URLs
        image_urls = []
        if (hasattr(post.value, 'embed') and 
            hasattr(post.value.embed, 'images')):
            
            for img in post.value.embed.images:
                if hasattr(img, 'image') and hasattr(img.image, 'ref'):
                    # Construct CDN URL from image reference
                    did = post.uri.split('/')[2]
                    img_ref = img.image.ref.link
                    img_url = f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{img_ref}@jpeg"
                    image_urls.append(img_url)
        
        # Include external link if present
        if (hasattr(post.value, 'embed') and 
            hasattr(post.value.embed, 'external')):
            text += ' ' + post.value.embed.external.uri
        
        return {
            'uri': post.uri,
            'cid': post.cid,
            'author': post.uri.split('/')[2],  # Extract author DID from URI
            'text': text,
            'image_urls': image_urls,
            'url': url
        }
    
    except Exception as e:
        print(f"Error fetching post from URL {url}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test hate speech detection on a single Bluesky post')
    parser.add_argument('--url', type=str, required=True, help='URL of the Bluesky post to analyze')
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model to use')
    parser.add_argument('--threshold', type=float, default=0.5, help='Hate speech probability threshold')
    args = parser.parse_args()
    
    # Initialize client
    client = Client()
    client.login(USERNAME, PW)
    
    # Initialize hate speech detector
    detector = HateSpeechDetector(model_name=args.model)
    
    # Fetch post from URL
    post = extract_post_data(client, args.url)
    
    if not post:
        print(f"Failed to fetch post from URL: {args.url}")
        return
    
    print(f"Analyzing post: {post['url']}")
    print(f"Text: {post['text']}")
    print(f"Images: {len(post['image_urls'])}")
    
    # Analyze post
    results = detector.predict([post], threshold=args.threshold)
    
    if not results:
        print("Analysis failed")
        return
    
    result = results[0]
    
    # Print results
    print("\nAnalysis Results:")
    print("-" * 80)
    print(f"Hate Speech Probability: {result['hate_probability']:.4f}")
    print(f"Classification: {'HATE SPEECH' if result['is_hate_speech'] else 'NOT HATE SPEECH'}")
    print(f"Explanation: {result.get('explanation', 'No explanation provided')}")
    print("-" * 80)
    
    # Save detailed results to file
    output_file = "single_post_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'post': post,
            'analysis': result
        }, f, indent=2)
    
    print(f"Detailed results saved to {output_file}")

if __name__ == "__main__":
    main() 