#!/usr/bin/env python3
import os
import time
import argparse
import json
import requests
import base64
from typing import List, Dict, Any
import openai
from atproto import Client
from dotenv import load_dotenv
from label import did_from_handle

# Load credentials from .env
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

class HateSpeechDetector:
    def __init__(self, model_name: str = "gpt-4o"):
        """Initialize the hate speech detector with OpenAI."""
        print(f"Using OpenAI model: {model_name}")
        self.model_name = model_name
        
        # Verify API key is set
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    def _download_image_as_base64(self, image_url: str) -> str:
        """Download an image and convert it to base64 for API submission."""
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to download image: {image_url}")
                return None
            
            # Convert to base64
            image_data = base64.b64encode(response.content).decode('utf-8')
            return image_data
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None
    
    def predict(self, posts: List[Dict], threshold: float = 0.5, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Predict hate speech probability for a list of posts using OpenAI.
        
        Args:
            posts: List of post dictionaries containing text and image_urls
            threshold: Probability threshold for classifying as hate speech
            batch_size: Batch size for processing (used for rate limiting)
            
        Returns:
            List of dictionaries with hate speech probabilities and classifications
        """
        results = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(posts), batch_size):
            batch_posts = posts[i:i+batch_size]
            
            for post in batch_posts:
                text = post.get("text", "")
                image_urls = post.get("image_urls", [])
                
                if not text.strip() and not image_urls:
                    # Skip empty posts
                    results.append({
                        "text": text,
                        "hate_probability": 0.0,
                        "is_hate_speech": False
                    })
                    continue
                
                try:
                    # Prepare messages for OpenAI
                    messages = [
                        {"role": "system", "content": "You are an AI trained to detect hate speech in text and images."}
                    ]
                    
                    # Create prompt for hate speech detection
                    prompt = f"""
                    Analyze the following content for hate speech. Hate speech is defined as content that expresses hate, encourages violence, 
                    threatens, or harasses individuals or groups based on attributes like race, 
                    ethnicity, gender, religion, sexual orientation, or disability.
                    
                    Post text: "{text}"
                    """
                    
                    # Add information about images if present
                    if image_urls:
                        prompt += "\n\nThis post also contains images which are provided for analysis."
                    
                    prompt += """
                    
                    Provide a score from 0.0 to 1.0 indicating the probability that this is hate speech, 
                    where 0.0 means definitely not hate speech and 1.0 means definitely hate speech.
                    
                    Return only a JSON object with the following format:
                    {
                        "hate_probability": [score between 0.0 and 1.0],
                        "explanation": [brief explanation of your reasoning]
                    }
                    """
                    
                    messages.append({"role": "user", "content": prompt})
                    
                    # Add images if present
                    image_contents = []
                    for img_url in image_urls:
                        img_data = self._download_image_as_base64(img_url)
                        if img_data:
                            image_contents.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_data}"
                                }
                            })
                    
                    # If we have images, add them to the user message
                    if image_contents:
                        messages[-1]["content"] = [
                            {"type": "text", "text": prompt},
                            *image_contents
                        ]
                    
                    # Call OpenAI API
                    response = openai.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.0,  # Use deterministic output
                        response_format={"type": "json_object"}
                    )
                    
                    # Parse response
                    response_content = response.choices[0].message.content
                    response_json = json.loads(response_content)
                    
                    hate_prob = float(response_json.get("hate_probability", 0.0))
                    explanation = response_json.get("explanation", "No explanation provided")
                    
                    results.append({
                        "text": text,
                        "image_count": len(image_urls),
                        "hate_probability": hate_prob,
                        "is_hate_speech": hate_prob > threshold,
                        "explanation": explanation
                    })
                    
                except Exception as e:
                    print(f"Error analyzing post: {e}")
                    results.append({
                        "text": text,
                        "image_count": len(image_urls),
                        "hate_probability": 0.0,
                        "is_hate_speech": False,
                        "error": str(e)
                    })
                
                # Sleep to avoid rate limits
                time.sleep(0.5)
        
        return results

def fetch_profile_posts(client: Client, handle: str, limit: int = 100) -> List[Dict]:
    """Fetch posts from a specific profile using the helper function."""
    print(f"Fetching up to {limit} posts from @{handle}...")
    
    try:
        # Use the helper function to resolve handle to DID
        did = did_from_handle(handle)
        
        # Fetch author feed
        feed = client.app.bsky.feed.get_author_feed({"actor": did, "limit": limit})
        
        posts = []
        for item in feed.feed:
            post = item.post
            text = getattr(post.record, 'text', '')
            
            # Extract image URLs
            image_urls = []
            if (hasattr(post, 'embed') and 
                hasattr(post.embed, 'images')):
                
                for img in post.embed.images:
                    if hasattr(img, 'fullsize'):
                        image_urls.append(img.fullsize)
            
            # Include external link if present
            if getattr(post, 'embed', None) and getattr(post.embed, 'external', None):
                text += ' ' + post.embed.external.uri
            
            posts.append({
                'uri': post.uri,
                'cid': post.cid,
                'author': post.author.handle,
                'text': text,
                'image_urls': image_urls,
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
    
    # Analyze posts
    print(f"Analyzing {len(posts)} posts for hate speech...")
    results = detector.predict(posts, threshold=threshold)
    
    # Match results back to posts
    hate_speech_posts = []
    hate_speech_count = 0
    
    for i, result in enumerate(results):
        if result["is_hate_speech"]:
            hate_speech_count += 1
            hate_speech_posts.append({
                **posts[i],
                "hate_probability": result["hate_probability"],
                "explanation": result.get("explanation", "No explanation provided")
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
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model to use')
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
    detector = HateSpeechDetector(model_name=args.model)
    
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