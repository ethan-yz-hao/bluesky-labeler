#!/usr/bin/env python3
import os
import re
import argparse
import json
from typing import Dict, Any
from atproto import Client
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

GENERAL_JOB_KEYWORDS = [
    "hiring", "job opening", "career opportunity", "join our team",
    "we are recruiting", "software engineer role", "remote developer",
    "internship", "graduate program", "talent acquisition",
]

SCAM_PATTERNS = {
    # A. pay-to-earn / social-task grifts
    "social_tasks": [
        r"earn\s+\$?\d+\s*(/|per)?\s*(day|week|hour|hr|month)\b",
        r"make\s+\$?\d+\s*(/|per)?\s*(day|week|hour|hr|month)\b",
        r"\$?\d+\s+per\s+(day|week|hour|hr|month)\b",
        r"get paid\s+\$?\d+\s+(daily|weekly|hourly)\b",
        r"like (tweets|posts) (and|&) get paid",
        r"get paid to (retweet|like|share)",
    ],

    # B. zero-barrier promises
    "no_barrier": [
        r"no experience (needed|required)",
        r"work from home.*start (today|now)",
        r"quick cash", r"easy money", r"instant payout",
        r"daily payout", r"same-day pay",
    ],

    "funnel": [
        r"telegram\.me/", r"t\.me/", r"join our telegram",
        r"whatsapp.*\+?\d{6,}",
    ],
}

CTA_KEYWORDS = [
    "apply now", "apply today", "sign up", "sign-up",
    "join our telegram", "join telegram", "join our whatsapp", "join whatsapp",
    "dm me", "dm us", "message me", "contact us", "inbox me",
    "register here", "click the link", "start today",
]
CTA_REGEX = re.compile("|".join(CTA_KEYWORDS), re.IGNORECASE)

def is_scam_by_regex(text: str) -> bool:
    """True if text matches *any* scam pattern."""
    tl = text.lower()
    for bucket in SCAM_PATTERNS.values():
        for pattern in bucket:
            if re.search(pattern, tl, flags=re.IGNORECASE):
                return True
    return False

def post_from_url(client: Client, url: str):
    """
    Retrieve a Bluesky post from its URL
    """
    parts = url.split("/")
    rkey = parts[-1]
    handle = parts[-3]
    return client.get_post(rkey, handle)

def extract_post_data(client: Client, url: str) -> Dict:
    """Extract data from a post URL for analysis"""
    try:
        # Use the helper function to get the post
        post = post_from_url(client, url)
        
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

class JobScamDetector:
    def __init__(self):
        """Initialize the job scam detector with regex patterns."""
        pass
    
    def predict(self, post: Dict, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict if a post is a job scam using regex patterns.
        
        Args:
            post: Post dictionary containing text
            threshold: Not used in this implementation but kept for API consistency
            
        Returns:
            Dictionary with scam probability and classification
        """
        text = post.get("text", "")
        
        if not text.strip():
            # Skip empty posts
            return {
                "text": text,
                "scam_probability": 0.0,
                "is_job_scam": False,
                "explanation": "Empty post",
                "method": "empty_check"
            }
        
        try:
            # Check if text contains job-related keywords
            is_job_related = any(kw.lower() in text.lower() for kw in GENERAL_JOB_KEYWORDS)
            
            # Check for scam patterns
            money_bait = is_scam_by_regex(text)
            has_cta = CTA_REGEX.search(text) is not None
            
            # Determine which patterns matched
            matched_patterns = []
            for category, patterns in SCAM_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, text, flags=re.IGNORECASE):
                        matched_patterns.append(f"{category}: {pattern}")
            
            # Calculate probability based on our heuristics
            scam_probability = 0.0
            
            if money_bait and has_cta:
                scam_probability = 0.9  # High confidence
            elif money_bait:
                scam_probability = 0.7  # Medium-high confidence
            elif has_cta and not is_job_related:
                scam_probability = 0.4  # Medium confidence
            elif is_job_related:
                scam_probability = 0.1  # Likely legitimate job post
            
            # Generate explanation
            explanation = []
            if money_bait:
                explanation.append("Contains suspicious earnings claims")
            if has_cta:
                explanation.append("Contains call-to-action phrases")
            if matched_patterns:
                explanation.append(f"Matched patterns: {', '.join(matched_patterns[:3])}")
            if not explanation:
                explanation.append("No suspicious patterns detected")
            
            return {
                "text": text,
                "image_count": len(post.get("image_urls", [])),
                "scam_probability": scam_probability,
                "is_job_scam": scam_probability > threshold,
                "explanation": ". ".join(explanation),
                "post_url": post.get("url", ""),
                "uri": post.get("uri", ""),
                "method": "regex_patterns"
            }
            
        except Exception as e:
            print(f"Error analyzing post: {e}")
            return {
                "text": text,
                "image_count": len(post.get("image_urls", [])),
                "scam_probability": 0.0,
                "is_job_scam": False,
                "error": str(e),
                "post_url": post.get("url", ""),
                "uri": post.get("uri", ""),
                "method": "error"
            }

def main():
    parser = argparse.ArgumentParser(description='Analyze and label job scams in a Bluesky post')
    parser.add_argument('--url', type=str, required=True, help='URL of the Bluesky post to analyze')
    parser.add_argument('--threshold', type=float, default=0.5, help='Job scam probability threshold')
    parser.add_argument('--output', type=str, default='job_scam_analysis.json', help='Output JSON file')
    parser.add_argument('--label', type=str, default='job-scam', help='Label to apply to job scam posts')
    parser.add_argument('--no-label', action='store_true', help='Only analyze post, do not apply label')
    args = parser.parse_args()
    
    # Initialize client
    client = Client()
    client.login(USERNAME, PW)
    
    # Initialize job scam detector
    detector = JobScamDetector()
    
    # Extract post data
    post_data = extract_post_data(client, args.url)
    
    if not post_data:
        print(f"Failed to extract data from post URL: {args.url}")
        return
    
    # Analyze post
    print(f"Analyzing post: {args.url}")
    print(f"Text: {post_data['text']}")
    print(f"Images: {len(post_data['image_urls'])}")
    
    result = detector.predict(post_data, threshold=args.threshold)
    
    # Print analysis results
    print("\nAnalysis Results:")
    print("-" * 80)
    print(f"Detection Method: {result.get('method', 'unknown')}")
    print(f"Job Scam Probability: {result['scam_probability']:.4f}")
    print(f"Classification: {'JOB SCAM' if result['is_job_scam'] else 'NOT A JOB SCAM'}")
    print(f"Explanation: {result.get('explanation', 'No explanation provided')}")
    print("-" * 80)
    
    # Save results
    output_data = {
        "post_url": args.url,
        "analysis": result,
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main() 