import os
import json
import time
import csv
import random
import concurrent.futures
from dotenv import load_dotenv
import openai
import pandas as pd
from tqdm import tqdm

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Configuration
INPUT_FILE = "bluesky_money_posts.json"
OUTPUT_FILE = "labeled_money_posts.csv"
MAX_WORKERS = 5
MAX_RETRIES = 5  # Maximum number of retries for rate-limited requests

def load_posts():
    """Load posts from the JSON file"""
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_post(post):
    """
    Use ChatGPT to classify if a post is a money-making scam or not.
    Returns the post with added classification.
    """
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            # Create prompt for ChatGPT
            prompt = f"""
            Analyze the following social media post and determine if it's a money-making scam that promises easy income with minimal effort.
            
            POST: {post['text']}
            HASHTAG: #{post['hashtag']}
            SEARCH QUERY: {post.get('search_query', 'N/A')}
            
            Focus SPECIFICALLY on identifying posts that claim people can make money easily with minimal effort or time investment.
            Common signs include:
            - Promises of high/easy earnings with minimal work
            - "Get rich quick" schemes
            - Vague descriptions of how money is actually earned
            - Requests to contact via external messaging apps (WhatsApp, Telegram) for money-making opportunities
            - Unrealistic income claims
            
            DO NOT classify as scams:
            - Regular job postings (even if suspicious)
            - NSFW content or adult services
            - General spam that doesn't specifically promise easy money
            - Cryptocurrency discussions that don't promise easy earnings
            
            Classify as either "scam" or "not_scam" and provide a brief explanation.
            Return your answer in JSON format:
            {{"classification": "scam" or "not_scam", "confidence": 0-100, "explanation": "your reasoning"}}
            """
            
            # Call ChatGPT API
            response = client.chat.completions.create(
                model="gpt-4o",  # or "gpt-3.5-turbo" for a cheaper option
                messages=[
                    {"role": "system", "content": "You are an expert at identifying money-making scams that promise easy income with minimal effort. Only classify posts as scams if they explicitly promise easy money."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent results
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Add classification to post
            post_with_label = post.copy()
            post_with_label["is_scam"] = result["classification"]
            post_with_label["confidence"] = result["confidence"]
            post_with_label["explanation"] = result["explanation"]
            
            return post_with_label
        
        except openai.RateLimitError as e:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"Max retries exceeded for post {post['post_id']}: {e}")
                break
                
            # Exponential backoff with jitter
            wait_time = (2 ** retries) + random.uniform(0, 1)
            print(f"Rate limit hit, waiting {wait_time:.2f}s before retry {retries}/{MAX_RETRIES}...")
            time.sleep(wait_time)
            
        except Exception as e:
            print(f"Error classifying post {post['post_id']}: {e}")
            break
    
    # If we get here with retries exceeded or other error, return post with error info
    post_with_label = post.copy()
    post_with_label["is_scam"] = "error"
    post_with_label["confidence"] = 0
    post_with_label["explanation"] = f"Error: Rate limit or API error after {retries} retries"
    return post_with_label

def process_batch(batch):
    """Process a batch of posts"""
    results = []
    for post in batch:
        # Add significant delay between requests to avoid rate limiting
        delay = 0.5 + random.uniform(0.5, 1.0)  # 0.5-1.5 seconds between requests
        time.sleep(delay)
        results.append(classify_post(post))
    return results

def main():
    # Load posts
    print(f"Loading posts from {INPUT_FILE}...")
    posts = load_posts()
    total_posts = len(posts)
    print(f"Loaded {total_posts} posts")
    
    # Check for existing progress
    existing_labeled = []
    if os.path.exists(f"intermediate_{OUTPUT_FILE}"):
        try:
            existing_df = pd.read_csv(f"intermediate_{OUTPUT_FILE}")
            existing_labeled = existing_df.to_dict('records')
            existing_ids = set(item['post_id'] for item in existing_labeled)
            posts = [post for post in posts if post['post_id'] not in existing_ids]
            print(f"Resuming from intermediate file. {len(existing_labeled)} posts already processed, {len(posts)} remaining.")
        except Exception as e:
            print(f"Could not load intermediate results: {e}")
    
    # Process posts in parallel
    labeled_posts = existing_labeled.copy()
    
    # Calculate batch size based on total posts and workers
    batch_size = max(1, min(10, len(posts) // MAX_WORKERS))  # Smaller batches, max 10 posts per batch
    batches = [posts[i:i + batch_size] for i in range(0, len(posts), batch_size)]
    
    print(f"Processing {len(posts)} posts with {MAX_WORKERS} workers in {len(batches)} batches...")
    
    # Create progress bar for the entire process
    pbar = tqdm(total=len(posts), desc="Classifying posts")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all batches to the executor
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                batch_results = future.result()
                labeled_posts.extend(batch_results)
                
                # Update progress bar with the batch size
                pbar.update(len(batch))
                
                # Save intermediate results after each batch
                df = pd.DataFrame(labeled_posts)
                df.to_csv(f"intermediate_{OUTPUT_FILE}", index=False)
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
    
    # Close the progress bar
    pbar.close()
    
    # Save final results
    print(f"Saving {len(labeled_posts)} labeled posts to {OUTPUT_FILE}...")
    df = pd.DataFrame(labeled_posts)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # Print statistics
    scam_count = sum(1 for post in labeled_posts if post.get("is_scam") == "scam")
    not_scam_count = sum(1 for post in labeled_posts if post.get("is_scam") == "not_scam")
    error_count = sum(1 for post in labeled_posts if post.get("is_scam") == "error")
    
    print("\n--- Classification Results ---")
    print(f"Total posts: {len(labeled_posts)}")
    print(f"Scams: {scam_count} ({scam_count/len(labeled_posts)*100:.1f}%)")
    print(f"Not scams: {not_scam_count} ({not_scam_count/len(labeled_posts)*100:.1f}%)")
    print(f"Errors: {error_count} ({error_count/len(labeled_posts)*100:.1f}%)")
    print("-----------------------------")

if __name__ == "__main__":
    main() 