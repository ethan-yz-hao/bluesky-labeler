"""Implementation of automated moderator"""

import os
import pandas as pd
from typing import List, Dict
from atproto import Client
import re
import requests
from PIL import Image
from io import BytesIO
from perception.hashers import PHash
import numpy as np

T_AND_S_LABEL = "t-and-s"
DOG_LABEL = "dog"
THRESH = 0.3

class AutomatedLabeler:
    """Automated labeler implementation"""

    def __init__(self, client: Client, input_dir):
        self.client = client
        self.input_dir = input_dir
        
        # Load T&S words and domains from CSV files
        self.ts_words = self._load_ts_words()
        self.ts_domains = self._load_ts_domains()
        
        # Load news domains and their sources
        self.news_domains = self._load_news_domains()
        
        # Initialize PHash for image comparison
        self.hasher = PHash()
        
        # Load dog image hashes
        self.dog_hashes = self._load_dog_hashes()
   
    def _load_ts_words(self) -> List[str]:
        """Load Trust and Safety words from CSV file"""
        words_path = os.path.join(self.input_dir, "t-and-s-words.csv")
        if os.path.exists(words_path):
            df = pd.read_csv(words_path)
            return [word.lower() for word in df["Word"].tolist()]
        return []
    
    def _load_ts_domains(self) -> List[str]:
        """Load Trust and Safety domains from CSV file"""
        domains_path = os.path.join(self.input_dir, "t-and-s-domains.csv")
        if os.path.exists(domains_path):
            df = pd.read_csv(domains_path)
            return [domain.lower() for domain in df["Domain"].tolist()]
        return []
    
    def _load_news_domains(self) -> Dict[str, str]:
        """Load news domains and their sources from CSV file"""
        domains_path = os.path.join(self.input_dir, "news-domains.csv")
        news_domains = {}
        if os.path.exists(domains_path):
            df = pd.read_csv(domains_path)
            for _, row in df.iterrows():
                news_domains[row["Domain"].lower()] = row["Source"].lower()
        return news_domains
    
    def _load_dog_hashes(self) -> List[np.ndarray]:
        """Load and compute hashes for dog images"""
        dog_images_dir = os.path.join(self.input_dir, "dog-list-images")
        dog_hashes = []
        
        if os.path.exists(dog_images_dir):
            for filename in os.listdir(dog_images_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(dog_images_dir, filename)
                    try:
                        hash_value = self.hasher.compute(image_path)
                        dog_hashes.append(hash_value)
                    except Exception as e:
                        print(f"Error computing hash for {filename}: {e}")
        
        return dog_hashes
    
    def _post_from_url(self, url: str):
        """Retrieve a Bluesky post from its URL"""
        parts = url.split("/")
        rkey = parts[-1]
        handle = parts[-3]
        return self.client.get_post(rkey, handle)
    
    def _contains_ts_word(self, text: str) -> bool:
        """Check if text contains any Trust and Safety words"""
        if not text or not self.ts_words:
            return False
        
        text_lower = text.lower()
        for word in self.ts_words:
            if word.lower() in text_lower:
                return True
        return False
    
    def _contains_ts_domain(self, text: str) -> bool:
        """Check if text contains any Trust and Safety domains"""
        if not text or not self.ts_domains:
            return False
        
        text_lower = text.lower()
        for domain in self.ts_domains:
            if domain.lower() in text_lower:
                return True
        return False
    
    def _find_news_sources(self, text: str) -> List[str]:
        """Find news sources in the text based on domain matches"""
        if not text or not self.news_domains:
            return []
        
        # Use a set to avoid duplicate sources
        sources = set()
        
        # Simple URL pattern to extract domains
        url_pattern = re.compile(r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')
        matches = url_pattern.findall(text.lower())
        
        for domain in matches:
            # Check if the domain or any subdomain matches our news domains
            for news_domain, source in self.news_domains.items():
                if domain == news_domain or domain.endswith('.' + news_domain):
                    sources.add(source)
        
        return list(sources)
    
    def _extract_image_urls(self, post) -> List[str]:
        """Extract image URLs from a post"""
        image_urls = []
        
        # Check if post has embedded images
        if (hasattr(post, 'value') and 
            hasattr(post.value, 'embed') and 
            hasattr(post.value.embed, 'images')):
            
            for img in post.value.embed.images:
                if hasattr(img, 'image') and hasattr(img.image, 'ref'):
                    # Construct CDN URL from image reference
                    did = post.uri.split('/')[2]
                    img_ref = img.image.ref.link
                    img_url = f"https://cdn.bsky.app/img/feed_thumbnail/plain/{did}/{img_ref}@jpeg"
                    image_urls.append(img_url)
        
        return image_urls
    
    def _is_dog_image(self, image_url: str) -> bool:
        """Check if an image matches any dog image in our dataset"""
        try:
            # Download the image
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                return False
            
            # Open the image and compute its hash
            img = Image.open(BytesIO(response.content))
            img_hash = self.hasher.compute(img)
            
            # Compare with dog hashes
            for dog_hash in self.dog_hashes:
                distance = self.hasher.compute_distance(img_hash, dog_hash)
                if distance <= THRESH:
                    return True
            
            return False
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            return False
    
    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given url
        """
        try:
            post = self._post_from_url(url)
            
            labels = []
            
            # Check for T&S words or domains in text
            if hasattr(post, 'value') and hasattr(post.value, 'text'):
                text = post.value.text
                
                # Check for T&S words or domains
                if self._contains_ts_word(text) or self._contains_ts_domain(text):
                    labels.append(T_AND_S_LABEL)
                
                # Check for news sources
                news_sources = self._find_news_sources(text)
                labels.extend(news_sources)
            
            # Check for dog images
            image_urls = self._extract_image_urls(post)
            for img_url in image_urls:
                if self._is_dog_image(img_url):
                    labels.append(DOG_LABEL)
                    break  # Only need to add the label once
            
            return labels
        except Exception as e:
            print(f"Error moderating post {url}: {e}")
            return []