import re
from typing import List, Dict
from urllib.parse import urlparse
import pandas as pd
from tld import get_fld  # pip install tld
from app.core.logger import logger


class CitationExtractor:
    """Robust automatic citation domain classification without hardcoding"""

    SOCIAL_KEYWORDS = [
        'reddit', 'twitter', 'facebook', 'linkedin',
        'instagram', 'youtube', 'quora', 'medium', 'tiktok',
        'pinterest', 'snapchat',
    ]

    @staticmethod
    def extract_citations_from_response(response: str) -> List[Dict[str, str]]:
        """
        Extract citations from response string.

        Supports:
        - Markdown style links: [text](url)
        - Footnote citations: [1]: https://...

        Returns list of dicts with keys: 'text', 'url', 'domain'
        """
        if not response or pd.isna(response):
            return []

        citations = []

        md_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        md_matches = re.findall(md_pattern, response)

        for text, url in md_matches:
            url = url.strip()
            domain = CitationExtractor._extract_domain(url)
            citations.append({
                'text': text.strip(),
                'url': url,
                'domain': domain,
            })

        footnote_pattern = r'\[\d+\]:\s*(https?://[^\s]+)'
        footnote_matches = re.findall(footnote_pattern, response)

        for url in footnote_matches:
            url = url.strip()
            domain = CitationExtractor._extract_domain(url)
            citations.append({
                'text': url,
                'url': url,
                'domain': domain,
            })

        # Deduplicate by url
        unique_citations = {}
        for c in citations:
            if c['url'] not in unique_citations:
                unique_citations[c['url']] = c

        return list(unique_citations.values())

    @staticmethod
    def _extract_domain(url: str) -> str:
        """
        Extract domain from url, fallback to parsed netloc if exception
        """
        try:
            # Use public suffix aware domain extractor
            fld = get_fld(url, fix_protocol=True)
            return fld.lower()
        except Exception as e:
            logger.warning(f"Failed extracting domain using tld for '{url}', falling back to urlparse: {e}")
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith('www.'):
                    domain = domain[4:]
                return domain
            except:
                return 'unknown'

    @staticmethod
    def classify_citation(domain: str, marketplace: str) -> str:
        """
        Classify citation domain into Owned, Competition, Social, or Earned using dynamic heuristic

        Algorithm:
         - Owned if domain contains marketplace root keyword ignoring suffixes (e.g. amazon in amazon.com)
         - Social if domain contains known social keywords
         - Competition if domain contains any other known marketplace root keyword, excluding own
         - Otherwise Earned
        """

        domain_root = CitationExtractor._root_domain(domain)
        marketplace_lower = marketplace.lower()
        marketplace_root = CitationExtractor._root_domain(marketplace_lower)

        # Social domain check
        if any(social_kw in domain_root for social_kw in CitationExtractor.SOCIAL_KEYWORDS):
            return 'Social'

        # Owned domain check - domain root contains marketplace root or vice versa
        if marketplace_root in domain_root or domain_root in marketplace_root:
            return 'Owned'

        # Load all known marketplaces dynamically from DB or config cache (simulate here)
        known_marketplaces = CitationExtractor._get_known_marketplace_roots()
        known_marketplaces = [mk for mk in known_marketplaces if mk != marketplace_root]

        # Check if domain root matches any competitor marketplace root
        if any(comp_mk in domain_root or domain_root in comp_mk for comp_mk in known_marketplaces):
            return 'Competition'

        # Otherwise earned
        return 'Earned'

    @staticmethod
    def _root_domain(domain: str) -> str:
        """
        Extract root domain (e.g. amazon from amazon.com, amazon.co.in)
        """
        if not domain:
            return ''
        parts = domain.split('.')
        if len(parts) < 2:
            return domain.lower()
        # Handle ccTLDs (co.in, com.au)
        # Using public suffix list approach (simple heuristic)
        if parts[-2] in ("co", "com", "net", "org") and len(parts) >=3:
            return parts[-3].lower()
        else:
            return parts[-2].lower()

    @staticmethod
    def _get_known_marketplace_roots() -> List[str]:
        """
        Retrieve all known marketplace root keywords dynamically.

        This function should load unique marketplace column entries from the database,
        normalize their root domains and cache them.

        Since DB access is async and complex here,
        simulate with a static list OR in your application replace with actual DB integration.
        """

        # Example static list for illustration. Replace or integrate with DB.
        example_marketplaces = [
            'Amazon', 'Flipkart', 'Myntra', 'Snapdeal', 'ShopClues', 'Ajio', 'TataCliq'
        ]

        roots = []
        for mk in example_marketplaces:
            mk_root = CitationExtractor._root_domain(mk.lower())
            roots.append(mk_root)
        return roots

    @staticmethod
    def get_citation_position(response: str, citation_url: str) -> int:
        """
        Determine citation order position in response (1-based)
        """
        if not response or pd.isna(response):
            return -1

        citations = CitationExtractor.extract_citations_from_response(response)

        for idx, cit in enumerate(citations, 1):
            if cit['url'] == citation_url:
                return idx
        return -1
