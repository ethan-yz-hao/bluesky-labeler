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

SCAM_PATTERNS = {
    # A. pay-to-earn / social-task grifts
    "social_tasks": [
        # original …
        r"earn\s+\$?\d+\s*(/|per)?\s*(day|week|hour|hr|month)\b",
        r"make\s+\$?\d+\s*(/|per)?\s*(day|week|hour|hr|month)\b",
        r"\$?\d+\s+per\s+(day|week|hour|hr|month)\b",
        r"get paid\s+\$?\d+\s+(daily|weekly|hourly)\b",
        r"like (tweets|posts) (and|&) get paid",
        r"get paid to (retweet|like|share)",
        # new …
        r"(like|click ads|surveys|data entry)\s*\w*\s*(get paid|earn|\$\d+)",
        r"work\s+\d+\s*hours?.*?\b(day|home)\b",          # “work 2 hours a day / work 3 hours home”
    ],

    # B. zero-barrier promises & “too-good-to-be-true” hooks
    "no_barrier": [
        # original …
        r"no experience (needed|required)",
        r"work from home.*start (today|now)",
        r"quick cash", r"easy money", r"instant payout",
        r"daily payout", r"same-day pay",
        # new …
        r"earn.*?(no experience|no degree|basic knowledge)",
        r"easy money.*?(work from home|online)",
        r"quick cash.*?(today|sign up)",
    ],

    # C. funnels that push victims off-platform
    "funnel": [
        # original …
        r"telegram\.me/", r"t\.me/", r"join our telegram",
        r"whatsapp.*\+?\d{6,}",
        # new …
        r"DM\s+for\s+details",
        r"whatsapp.*earn",
        r"telegram.*earn",
    ],

    # D. bait hashtags (single pattern for efficiency)
    "hashtags": [
        r"#(?:earnmoneyonline|earnfromhome|quickcash|easymoney|fastcash|getpaid|"
        r"paidinstantly|passiveincome|reviewandearn|likeandgetpaid|payperlike|"
        r"clicktask|paysurvey|dmforinfo|linkinbio|telegramlink|whatsappme|"
        r"100aday|500aweek|5kmonth)\b",
    ],
}

CTA_KEYWORDS = [
    "apply now", "apply today", "sign up", "sign-up",
    "join our telegram", "join telegram", "join our whatsapp", "join whatsapp",
    "dm me", "dm us", "message me", "contact us", "inbox me",
    "register here", "click the link", "start today",
]
CTA_REGEX = re.compile("|".join(CTA_KEYWORDS), re.IGNORECASE)


# ----------------------------------------------------------------------
# helpers ──────────────────────────────────────────────────────────────
# ----------------------------------------------------------------------

def _matched_patterns(text: str):
    """
    Return two booleans and a list:
        has_other  – any match outside the 'hashtags' bucket
        has_hash   – any match in the 'hashtags' bucket
        matches    – list[str] of "<bucket>: <regex>" for explanation
    """
    has_other, has_hash = False, False
    matches = []

    tl = text.lower()

    # Pass 1: look at every bucket except hashtags -------------
    for bucket, patterns in SCAM_PATTERNS.items():
        if bucket == "hashtags":
            continue
        for pat in patterns:
            if re.search(pat, tl, re.IGNORECASE):
                has_other = True
                matches.append(f"{bucket}: {pat}")

    for pat in SCAM_PATTERNS["hashtags"]:
        if re.search(pat, tl, re.IGNORECASE):
            has_hash = True
            matches.append(f"hashtags: {pat}")

    return has_other, has_hash, matches


def is_scam_by_regex(text: str):
    """
    True  → at least one *non-hashtag* pattern matched  
    False → either nothing matched or only hashtags matched
    """
    has_other, has_hash, _ = _matched_patterns(text)
    return has_other, has_hash

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

class PolicyProposalLabeler:
    def __init__(self):
        """Initialize the job scam detector with regex patterns."""
        pass
    
    def predict(self, post: Dict) -> Dict[str, Any]:
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
            
            # Check for scam patterns
            has_other, has_hash = is_scam_by_regex(text)
            has_cta    = CTA_REGEX.search(text) is not None
            
            isScam = "not_scam"

            if has_other or has_hash:
                isScam = "potential_scam"


            
            return {
                "text": text,
                "image_count": len(post.get("image_urls", [])),
                "is_job_scam": isScam,
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
                "is_job_scam": "",
                "error": str(e),
                "post_url": post.get("url", ""),
                "uri": post.get("uri", ""),
                "method": "error"
            }

def main():
    parser = argparse.ArgumentParser(description='Analyze and label job scams in a Bluesky post')
    parser.add_argument('--url', type=str, required=True, help='URL of the Bluesky post to analyze')
    parser.add_argument('--output', type=str, default='job_scam_analysis.json', help='Output JSON file')
    parser.add_argument('--label', type=str, default='job-scam', help='Label to apply to job scam posts')
    parser.add_argument('--no-label', action='store_true', help='Only analyze post, do not apply label')
    args = parser.parse_args()
    
    # Initialize client
    client = Client()
    client.login(USERNAME, PW)
    
    # Initialize job scam detector
    detector = PolicyProposalLabeler()
    
    # Extract post data
    post_data = extract_post_data(client, args.url)
    
    if not post_data:
        print(f"Failed to extract data from post URL: {args.url}")
        return
    
    # Analyze post
    print(f"Analyzing post: {args.url}")
    print(f"Text: {post_data['text']}")
    print(f"Images: {len(post_data['image_urls'])}")
    
    result = detector.predict(post_data)
    
    # Print analysis results
    print("\nAnalysis Results:")
    print("-" * 80)
    print(f"Detection Method: {result.get('method', 'unknown')}")
    print(f"Classification: {'Potential Job Scam' if result['is_job_scam'] == 'potential_scam' else 'Not a Job Scam'}")
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