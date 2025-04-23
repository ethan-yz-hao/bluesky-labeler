"""Implementation of automated moderator"""

import os
import pandas as pd
from typing import List, Dict
from atproto import Client
import re

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
    
    def moderate_post(self, url: str) -> List[str]:
        """
        Apply moderation to the post specified by the given url
        """
        try:
            post = self._post_from_url(url)
            
            # Check if post text exists - handle the nested structure
            if hasattr(post, 'value') and hasattr(post.value, 'text'):
                text = post.value.text
            else:
                return []
            
            labels = []
            
            # Check for T&S words or domains
            if self._contains_ts_word(text) or self._contains_ts_domain(text):
                labels.append(T_AND_S_LABEL)
            
            # Check for news sources
            news_sources = self._find_news_sources(text)
            labels.extend(news_sources)
            
            return labels
        except Exception as e:
            print(f"Error moderating post {url}: {e}")
            return []