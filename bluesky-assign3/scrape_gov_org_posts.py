#!/usr/bin/env python3
import os
import re
import time
import argparse
import json
from typing import List, Dict

from atproto import Client
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv(override=True)
USERNAME = os.getenv("USERNAME")
PW = os.getenv("PW")

# Regex to extract domains
URL_PATTERN = re.compile(r'https?://(?:www\.)?([\w.-]+\.[a-zA-Z]{2,})')

def extract_domains(text: str) -> List[str]:
    """Return all domains found in text."""
    return URL_PATTERN.findall(text.lower())


def is_gov_or_org(domain: str) -> bool:
    """Check for .gov or .org suffix."""
    return domain.endswith('cdc.gov') or domain.endswith('who.int')


def resolve_did(client: Client, handle: str) -> str:
    """Resolve a Bluesky handle to its DID via the identity API."""
    resp = client.com.atproto.identity.resolve_handle({"handle": handle})
    return resp.did


def list_feed_generators(client: Client, actor_did: str) -> List[str]:
    """Get a list of feed-generator URIs for a given actor DID."""
    # Retrieve up to 100 generators
    resp = client.app.bsky.feed.get_actor_feeds({"actor": actor_did, "limit": 100})
    return [gen.uri for gen in resp.feeds]


def scrape_posts(
    client: Client,
    feeds: List[str],
    target: int = 100,
    max_attempts: int = 1000
) -> List[Dict]:
    """Collect posts containing .gov or .org domains."""
    collected: List[Dict] = []
    attempts = 0
    idx = 0

    print(f"Collecting up to {target} posts...")
    while len(collected) < target and attempts < max_attempts:
        attempts += 1
        try:
            if feeds:
                feed_uri = feeds[idx % len(feeds)]
                idx += 1
                print(f"[Attempt {attempts}] Fetching feed: {feed_uri}")
                batch = client.app.bsky.feed.get_feed({"feed": feed_uri}).feed
            else:
                print(f"[Attempt {attempts}] Fetching home timeline...")
                batch = client.app.bsky.feed.get_timeline({"limit": 50}).feed

            for view in batch:
                post = view.post
                text = getattr(post.record, 'text', '')
                if getattr(post, 'embed', None) and getattr(post.embed, 'external', None):
                    text += ' ' + post.embed.external.uri

                domains = extract_domains(text)
                gov_orgs = [d for d in domains if is_gov_or_org(d)]
                if gov_orgs:
                    entry = {
                        'uri': post.uri,
                        'cid': post.cid,
                        'author': post.author.handle,
                        'text': text,
                        'domains': gov_orgs,
                        'url': f"https://bsky.app/profile/{post.author.handle}/post/{post.uri.split('/')[-1]}"
                    }
                    if entry['uri'] not in {p['uri'] for p in collected}:
                        collected.append(entry)
                        print(f"Collected {len(collected)}/{target}: {entry['url']}")
                        if len(collected) >= target:
                            break

            time.sleep(1)
        except Exception as e:
            print(f"Error attempt {attempts}: {e}")
            time.sleep(1)

    print(f"Done: {len(collected)} posts in {attempts} attempts.")
    return collected


def main():
    parser = argparse.ArgumentParser(description='Scrape Bluesky posts with .gov/.org links')
    parser.add_argument('--count', type=int, default=10, help='Number of posts to collect')
    parser.add_argument('--output', type=str, default='health_posts.json', help='Output JSON path')
    args = parser.parse_args()

    client = Client()
    client.login(USERNAME, PW)

    profiles = ['allyann.bsky.social', 'aendra.com', 'itsonelouder.com', 'newsbeacon.co']
    feeds: List[str] = []

    for prof in profiles:
        try:
            did = resolve_did(client, prof)
            gens = list_feed_generators(client, did)
            print(f"Generators for {prof} ({did}): {gens}")
            feeds.extend(gens)
        except Exception as err:
            print(f"Failed to list generators for {prof}: {err}")

    if not feeds:
        print('No generators found; falling back to home timeline.')

    posts = scrape_posts(client, feeds, target=args.count)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2)
    print(f"Saved {len(posts)} posts to {args.output}")


if __name__ == '__main__':
    main()
