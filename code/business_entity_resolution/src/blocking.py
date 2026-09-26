"""
Multi-Pass Inverted Index Blocking Engine for Candidate Generation.
"""

from typing import List, Tuple, Set, Dict, Any

try:
    from .utils import normalize_text, get_name_tokens, get_address_components
    from .config import config
except (ImportError, ValueError):
    from utils import normalize_text, get_name_tokens, get_address_components
    from config import config


class BlockingEngine:
    def __init__(
        self,
        name_stop_words: Set[str] = None,
        addr_stop_words: Set[str] = None,
        max_postings: int = None,
        top_k: int = None
    ):
        self.name_stop_words = name_stop_words if name_stop_words is not None else config.NAME_STOP_WORDS
        self.addr_stop_words = addr_stop_words if addr_stop_words is not None else config.ADDR_STOP_WORDS
        self.max_postings = max_postings if max_postings is not None else config.MAX_POSTINGS_PER_KEY
        self.top_k = top_k if top_k is not None else config.TOP_K_CANDIDATES
        self.index: Dict[Tuple, List[str]] = {}

    def extract_keys(self, name: str, address: str, country: str) -> List[Tuple]:
        """
        Extract high-recall multi-pass blocking keys for an entity:
        - Name unigrams, bigrams, 4-char prefixes
        - Informative address tokens & distinct word pairs
        - Postal code + street numbers
        - Street number + street words
        - Short acronym + address postal code / street number
        """
        keys = []
        name_clean = normalize_text(name, clean_domains=True)
        addr_clean = normalize_text(address, clean_domains=False)
        
        # 1. Informative Name Tokens & Short Tokens
        tokens, short_tokens = get_name_tokens(name_clean, self.name_stop_words, config.MIN_TOKEN_LEN)
        for t in tokens:
            keys.append(('tok', country, t))
            
        # 2. Adjacent Name Token Bigrams
        for i in range(len(tokens) - 1):
            keys.append(('bi', country, f"{tokens[i]}_{tokens[i+1]}"))
            
        # 3. 4-character prefix of first token
        if tokens and len(tokens[0]) >= config.PREFIX_LEN:
            keys.append(('pref4', country, tokens[0][:config.PREFIX_LEN]))
            
        # 4. Address Components
        nums, zip_code, street_nums, words = get_address_components(addr_clean, self.addr_stop_words)
        
        # Informative single address tokens (len >= 4, filtered by addr stopwords)
        for w in words[:4]:
            if len(w) >= 4:
                keys.append(('addr_tok', country, w))
        
        # Zip + Street number
        if zip_code and street_nums:
            for sn in street_nums[:3]:
                keys.append(('zip_num', country, zip_code, sn))
                    
        # Street number + Street word
        if street_nums and words:
            for sn in street_nums[:2]:
                for w in words[:4]:
                    keys.append(('sn_w', country, sn, w))
                
        # Zip + Street word
        if zip_code and words:
            for w in words[:3]:
                keys.append(('zip_w', country, zip_code, w))
                
        # Distinct informative address word pairs
        for i in range(min(len(words), 4)):
            for j in range(i + 1, min(len(words), 4)):
                w1, w2 = sorted([words[i], words[j]])
                keys.append(('addr_pair', country, w1, w2))
                
        # Short token + address number or zip (handles acronyms e.g. LU Infra, KM Thirty)
        if short_tokens and (street_nums or zip_code):
            st = short_tokens[0]
            if zip_code:
                keys.append(('st_zip', country, st, zip_code))
            for sn in street_nums[:2]:
                keys.append(('st_sn', country, st, sn))
            
        return keys

    def build_index_from_arrays(self, ids, names, addrs, countries):
        """
        Build inverted index directly from numpy/pandas column arrays with zero intermediate tuple allocation.
        """
        self.index.clear()
        cap = self.max_postings + 1
        n = len(ids)
        for i in range(n):
            keys = self.extract_keys(names[i], addrs[i], countries[i])
            eid = ids[i]
            for k in keys:
                p = self.index.get(k)
                if p is None:
                    self.index[k] = [eid]
                elif len(p) < cap:
                    p.append(eid)

    def build_index(self, records: List[Tuple[str, str, str, str]]):
        """
        Build inverted index from list of tuples: (entity_id, business_name, business_address, country).
        Caps posting list lengths at max_postings + 1 to eliminate memory waste on common tokens.
        """
        self.index.clear()
        cap = self.max_postings + 1
        for entity_id, name, addr, country in records:
            keys = self.extract_keys(name, addr, country)
            for k in keys:
                postings = self.index.get(k)
                if postings is None:
                    self.index[k] = [entity_id]
                elif len(postings) < cap:
                    postings.append(entity_id)



    def query(self, entity_id: str, name: str, address: str, country: str) -> List[str]:
        """
        Retrieve candidate matching IDs for a single Source 1 entity.
        Filters posting lists exceeding max_postings to prevent Cartesian explosions.
        """
        keys = self.extract_keys(name, address, country)
        candidates = set()
        for k in keys:
            postings = self.index.get(k, [])
            if 0 < len(postings) <= self.max_postings:
                candidates.update(postings)
        return list(candidates)
