import os
import csv
import time
import json
import random
from dotenv import load_dotenv
from atproto import Client

# Load environment variables
load_dotenv()
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

# Initialize Bluesky client
client = Client()
client.login(USERNAME, PW)

# Target hashtags to search for
HASHTAGS = [
    "earnmoney", "makemoney", "earnfromhome", "quickcash", "easymoney", 
    "fastcash", "getpaid", "paidinstantly", "passiveincome", "reviewandearn", 
    "likeandgetpaid", "payperlike", "clicktask", "paysurvey", "dmforinfo", 
    "linkinbio", "telegramlink", "whatsappme", "100aday", "500aweek", "5kmonth"
]
TARGET_POSTS = 300  # Total posts to collect
POSTS_PER_HASHTAG = 100  # Target posts per hashtag

def search_posts_by_hashtag(hashtag, max_posts):
    """
    Collect up to max_posts that contain the specified hashtag using multiple search strategies.
    """
    collected = []
    
    # Strategy 1: Direct hashtag search
    direct_posts = search_with_strategy(hashtag, f"#{hashtag}", max_posts)
    collected.extend(direct_posts)
    print(f"  - Direct search: {len(direct_posts)} posts")
    
    # If we didn't get enough posts, try additional strategies
    if len(collected) < max_posts:
        remaining = max_posts - len(collected)
        
        # Strategy 2: Search with related terms
        related_terms = [
            "money", "income", "earn", "cash", "payment", "work", 
            "opportunity", "job", "business", "online"
        ]
        random.shuffle(related_terms)  # Randomize to get different results each run
        
        for term in related_terms[:3]:  # Use up to 3 related terms
            if len(collected) >= max_posts:
                break
                
            combo_query = f"#{hashtag} {term}"
            combo_posts = search_with_strategy(hashtag, combo_query, remaining)
            
            # Add only new posts
            for post in combo_posts:
                if not any(p["post_id"] == post["post_id"] for p in collected):
                    collected.append(post)
                    
            print(f"  - '{combo_query}' search: {len(combo_posts)} posts")
            remaining = max_posts - len(collected)
    
        # Strategy 3: Search for posts containing the hashtag text (without #)
        if len(collected) < max_posts:
            remaining = max_posts - len(collected)
            text_posts = search_with_strategy(hashtag, hashtag, remaining)
            
            # Add only new posts
            for post in text_posts:
                if not any(p["post_id"] == post["post_id"] for p in collected):
                    collected.append(post)
                    
            print(f"  - Text search: {len(text_posts)} posts")
    
    return collected[:max_posts]  # Ensure we don't exceed max_posts

def search_with_strategy(hashtag, query, max_posts):
    """
    Search for posts using a specific query strategy.
    """
    collected = []
    cursor = None
    
    while len(collected) < max_posts:
        try:
            # Search for posts with the query
            res = client.app.bsky.feed.search_posts({
                "q": query, 
                "limit": 100,  # Maximum allowed by API
                "cursor": cursor
            })
            
            if not res.posts:
                break
                
            for p in res.posts:
                rec = getattr(p, "record", None)
                if not rec or not hasattr(rec, "text"):
                    continue
                
                # Extract post data
                text = rec.text
                
                # Skip if the post doesn't actually contain the hashtag
                # (for text search strategy, ensure it's relevant)
                if query == hashtag and f"#{hashtag}" not in text.lower() and hashtag not in text.lower():
                    continue
                
                # Extract links if available
                links = []
                if getattr(rec, "facets", None):
                    for facet in rec.facets:
                        for feat in facet.features:
                            if hasattr(feat, "uri"):
                                links.append(feat.uri)
                
                # Create post entry
                post_data = {
                    "post_id": p.uri.split("/")[-1],
                    "text": text,
                    "creator": p.author.handle,
                    "likes": getattr(p, "likeCount", 0),
                    "reposts": getattr(p, "repostCount", 0),
                    "replies": getattr(p, "replyCount", 0),
                    "links": ", ".join(links),
                    "hashtag": hashtag,
                    "indexed_at": getattr(p, "indexedAt", ""),
                    "search_query": query,
                }
                
                # Only add if not already in the collection
                if not any(post["post_id"] == post_data["post_id"] for post in collected):
                    collected.append(post_data)
                
                if len(collected) >= max_posts:
                    break
            
            # Get cursor for pagination
            cursor = getattr(res, "cursor", None)
            if not cursor:
                break
                
            # Respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"  - Error: {e}")
            time.sleep(5)  # Wait longer on error
    
    return collected

def main():
    all_posts = []
    
    for hashtag in HASHTAGS:
        # Check if we've already reached the overall target
        if len(all_posts) >= TARGET_POSTS:
            print(f"\n🎯 Reached overall target of {TARGET_POSTS} posts. Stopping collection.")
            break
            
        # Calculate how many more posts we need
        remaining_posts = TARGET_POSTS - len(all_posts)
        posts_to_collect = min(POSTS_PER_HASHTAG, remaining_posts)
        
        print(f"\n🔎 Collecting posts with #{hashtag}... (need {posts_to_collect} more to reach target)")
        posts = search_posts_by_hashtag(hashtag, posts_to_collect)
        all_posts.extend(posts)
        print(f"✅ Collected {len(posts)} posts with #{hashtag}, total: {len(all_posts)}/{TARGET_POSTS}")
    
    # Remove duplicate posts based on identical text content
    print("\n🔍 Removing duplicate posts...")
    unique_posts = []
    seen_texts = set()
    
    for post in all_posts:
        if post["text"] not in seen_texts:
            seen_texts.add(post["text"])
            unique_posts.append(post)
    
    duplicate_count = len(all_posts) - len(unique_posts)
    print(f"✅ Removed {duplicate_count} duplicate posts")
    
    # Save to CSV
    csv_filename = "bluesky_money_posts.csv"
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "post_id", "text", "creator", "likes", "reposts", 
                "replies", "links", "hashtag", "indexed_at", "search_query"
            ],
        )
        writer.writeheader()
        writer.writerows(unique_posts)
    
    # Also save as JSON for easier processing
    json_filename = "bluesky_money_posts.json"
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(unique_posts, f, indent=2)
    
    print(f"\n📄 Collected {len(unique_posts)} unique posts from {len(all_posts)} total posts")
    print(f"📄 Data saved to {csv_filename} and {json_filename}")

if __name__ == "__main__":
    main() 