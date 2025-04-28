#!/usr/bin/env python3
import os
import time
import argparse
import json
import requests
import base64
from typing import Dict, Any
import openai
from atproto import Client
from dotenv import load_dotenv
from label import did_from_handle, label_post, post_from_url

# Load credentials from .env
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

class HateSpeechDetector:
    def __init__(self, model_name: str = "gpt-4o", hurtlex_threshold: float = 0.05):
        """Initialize the hate speech detector with HurtLex and OpenAI."""
        print(f"Using OpenAI model: {model_name}")
        self.model_name = model_name
        self.hurtlex_threshold = hurtlex_threshold
        
        # Verify API key is set
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        # Load HurtLex dictionary
        self.hurtlex_terms = self._load_hurtlex()
        print(f"Loaded {len(self.hurtlex_terms)} HurtLex terms")
    
    def _load_hurtlex(self):
        """Load HurtLex dictionary for English language."""
        try:
            # Path to HurtLex file - adjust as needed
            hurtlex_path = os.path.join(os.path.dirname(__file__), "data", "hurtlex_EN.tsv")
            
            # If file doesn't exist, download it
            if not os.path.exists(hurtlex_path):
                os.makedirs(os.path.dirname(hurtlex_path), exist_ok=True)
                print("Downloading HurtLex dictionary...")
                # URL for English HurtLex
                url = "https://github.com/valeriobasile/hurtlex/raw/master/lexica/EN/1.2/hurtlex_EN.tsv"
                response = requests.get(url)
                with open(hurtlex_path, 'wb') as f:
                    f.write(response.content)
            
            # Load terms from file
            terms = set()
            with open(hurtlex_path, 'r', encoding='utf-8') as f:
                # Skip header
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:  # Make sure we have enough columns
                        # Add the lemma (term) - it's in column 5 (index 4)
                        lemma = parts[4].lower()
                        # Handle multi-word terms by adding both the full term and individual words
                        terms.add(lemma)
                        # For multi-word terms, add individual words too
                        if ' ' in lemma:
                            for word in lemma.split():
                                if len(word) > 2:  # Skip very short words
                                    terms.add(word)
            
            print(f"Loaded {len(terms)} HurtLex terms")
            if len(terms) < 100:  # Sanity check
                print(f"Warning: Only loaded {len(terms)} terms, which seems low")
                print(f"Sample terms: {list(terms)[:20]}")
            
            return terms
            
        except Exception as e:
            print(f"Error loading HurtLex: {e}")
            # Return empty set if loading fails
            return set()
    
    def _check_hurtlex(self, text: str) -> Dict[str, Any]:
        """
        Check text against HurtLex dictionary.
        
        Returns:
            Dictionary with hate speech probability based on HurtLex matches
        """
        if not text or not self.hurtlex_terms:
            print("No text or HurtLex terms available")
            return {"hate_probability": 0.0, "matched_terms": []}
        
        # Normalize text
        text = text.lower()
        # Split on whitespace and remove punctuation
        words = [word.strip(".,!?;:\"'()[]{}") for word in text.split()]
        
        # Find matches - check both individual words and n-grams
        matched_terms = []
        
        # Check individual words
        for word in words:
            if word in self.hurtlex_terms:
                matched_terms.append(word)
        
        # Check bigrams (pairs of words)
        for i in range(len(words) - 1):
            bigram = words[i] + " " + words[i+1]
            if bigram in self.hurtlex_terms:
                matched_terms.append(bigram)
        
        # Check trigrams (triplets of words)
        for i in range(len(words) - 2):
            trigram = words[i] + " " + words[i+1] + " " + words[i+2]
            if trigram in self.hurtlex_terms:
                matched_terms.append(trigram)
        
        # Calculate probability based on proportion of matched terms
        if not words:
            return {"hate_probability": 0.0, "matched_terms": []}
            
        # Calculate hate probability based on proportion of matched terms
        # Use a more sensitive formula - even one match could be significant
        if matched_terms:
            # Base probability on number of matches and their proportion in the text
            hate_probability = min(1.0, (len(matched_terms) / len(words) * 3) + 0.2)
        else:
            hate_probability = 0.0
        
        print(f"HurtLex check: Found {len(matched_terms)} matches in {len(words)} words")
        if matched_terms:
            print(f"Matched terms: {matched_terms}")
        print(f"HurtLex hate probability: {hate_probability}")
        
        return {
            "hate_probability": hate_probability,
            "matched_terms": matched_terms
        }
    
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
    
    def predict(self, post: Dict, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict hate speech probability for a single post using a staged approach:
        1. Check against HurtLex dictionary
        2. If HurtLex doesn't identify as hate speech, use OpenAI for more complex analysis
        
        Args:
            post: Post dictionary containing text and image_urls
            threshold: Probability threshold for classifying as hate speech
            
        Returns:
            Dictionary with hate speech probability and classification
        """
        text = post.get("text", "")
        image_urls = post.get("image_urls", [])
        
        if not text.strip() and not image_urls:
            # Skip empty posts
            return {
                "text": text,
                "hate_probability": 0.0,
                "is_hate_speech": False,
                "explanation": "Empty post",
                "method": "empty_check"
            }
        
        try:
            # Stage 1: Check against HurtLex
            print(f"Checking HurtLex for post: {post.get('url', '')}")
            hurtlex_result = self._check_hurtlex(text)
            hate_prob = hurtlex_result["hate_probability"]
            matched_terms = hurtlex_result["matched_terms"]
                      
            # If HurtLex score is very low, we can skip OpenAI
            if hate_prob < self.hurtlex_threshold:
                print(f"HurtLex identified as not hate speech: {hate_prob}")
                return {
                    "text": text,
                    "image_count": 0,
                    "hate_probability": hate_prob,
                    "is_hate_speech": False,
                    "explanation": "No hate terms detected",
                    "post_url": post.get("url", ""),
                    "uri": post.get("uri", ""),
                    "method": "hurtlex"
                }
            
            # Stage 2: Use OpenAI for more complex analysis
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
            
            # Add information about HurtLex matches if any
            if matched_terms:
                prompt += f"\n\nNote: The text contains potentially harmful terms: {', '.join(matched_terms)}"
            
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
            
            return {
                "text": text,
                "image_count": len(image_urls),
                "hate_probability": hate_prob,
                "is_hate_speech": hate_prob > threshold,
                "explanation": explanation,
                "post_url": post.get("url", ""),
                "uri": post.get("uri", ""),
                "method": "openai"
            }
            
        except Exception as e:
            print(f"Error analyzing post: {e}")
            return {
                "text": text,
                "image_count": len(image_urls),
                "hate_probability": 0.0,
                "is_hate_speech": False,
                "error": str(e),
                "post_url": post.get("url", ""),
                "uri": post.get("uri", ""),
                "method": "error"
            }

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

def label_hate_speech_post(client: Client, post_result: Dict, label_value: str = "hate-speech") -> Dict:
    """
    Apply a label to a post identified as hate speech.
    
    Args:
        client: ATProto client
        post_result: Post analysis result
        label_value: Label to apply to hate speech post
        
    Returns:
        Dictionary with labeling result
    """
    if not post_result.get("is_hate_speech", False):
        return {
            "post_url": post_result.get("post_url", ""),
            "hate_probability": post_result.get("hate_probability", 0.0),
            "explanation": post_result.get("explanation", ""),
            "labeling_success": False,
            "error": "Post not classified as hate speech"
        }
    
    post_url = post_result.get("post_url")
    if not post_url:
        return {
            "hate_probability": post_result.get("hate_probability", 0.0),
            "explanation": post_result.get("explanation", ""),
            "labeling_success": False,
            "error": "No post URL provided"
        }
    
    try:
        # Create a labeler client
        did = did_from_handle(USERNAME)
        labeler_client = client.with_proxy("atproto_labeler", did)
        
        # Apply label to post
        result = label_post(client, labeler_client, post_url, [label_value])
        
        return {
            "post_url": post_url,
            "hate_probability": post_result.get("hate_probability", 0.0),
            "explanation": post_result.get("explanation", ""),
            "labeling_success": True,
            "label_applied": label_value
        }
        
    except Exception as e:
        print(f"Error labeling post {post_url}: {e}")
        return {
            "post_url": post_url,
            "hate_probability": post_result.get("hate_probability", 0.0),
            "explanation": post_result.get("explanation", ""),
            "labeling_success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Analyze and label hate speech in a Bluesky post')
    parser.add_argument('--url', type=str, required=True, help='URL of the Bluesky post to analyze')
    parser.add_argument('--threshold', type=float, default=0.5, help='Hate speech probability threshold')
    parser.add_argument('--output', type=str, default='hate_speech_analysis.json', help='Output JSON file')
    parser.add_argument('--model', type=str, default='gpt-4o', help='OpenAI model to use')
    parser.add_argument('--label', type=str, default='hate-speech', help='Label to apply to hate speech posts')
    parser.add_argument('--no-label', action='store_false', help='Only analyze post, do not apply label')
    parser.add_argument('--hurtlex-threshold', type=float, default=0.05, 
                        help='Threshold below which HurtLex results are considered negative')
    args = parser.parse_args()
    
    # Initialize client
    client = Client()
    client.login(USERNAME, PW)
    
    # Initialize hate speech detector
    detector = HateSpeechDetector(model_name=args.model, hurtlex_threshold=args.hurtlex_threshold)
    
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
    print(f"Hate Speech Probability: {result['hate_probability']:.4f}")
    print(f"Classification: {'HATE SPEECH' if result['is_hate_speech'] else 'NOT HATE SPEECH'}")
    print(f"Explanation: {result.get('explanation', 'No explanation provided')}")
    print("-" * 80)
    
    # Apply label if requested and post is classified as hate speech
    labeling_result = None
    if not args.no_label and result.get("is_hate_speech", False):
        print(f"\nApplying '{args.label}' label to post...")
        labeling_result = label_hate_speech_post(client, result, label_value=args.label)
        
        if labeling_result.get("labeling_success", False):
            print(f"Successfully applied '{args.label}' label to post")
        else:
            print(f"Failed to apply label: {labeling_result.get('error', 'Unknown error')}")
    
    # Save results
    output_data = {
        "post_url": args.url,
        "analysis": result,
        "labeling_result": labeling_result if not args.no_label and result.get("is_hate_speech", False) else None
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main() 