"""
Medallion Architecture - Gold Layer: High-Precision Entity Resolution Engine.
Combines high-recall multi-pass blocking, dual-similarity candidate ranking,
dense pairwise feature extraction (24 features), and multi-tier quality verification gates.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Any
from rapidfuzz import fuzz

try:
    from .config import config
    from .silver import normalize_silver_entity
except (ImportError, ValueError):
    from config import config
    from silver import normalize_silver_entity


class GoldResolutionEngine:
    def __init__(
        self,
        name_stopwords: Set[str] = None,
        addr_stopwords: Set[str] = None,
        max_postings: int = None,
        top_k: int = None
    ):
        self.name_stopwords = name_stopwords if name_stopwords is not None else config.NAME_STOP_WORDS
        self.addr_stopwords = addr_stopwords if addr_stopwords is not None else config.ADDR_STOP_WORDS
        self.max_postings = max_postings if max_postings is not None else config.MAX_POSTINGS_PER_KEY
        self.top_k = top_k if top_k is not None else config.TOP_K_CANDIDATES
        self.index: Dict[Any, List[int]] = {}
        self.entity_ids: Optional[np.ndarray] = None

    def extract_keys(self, silver_entity: dict, country: str) -> List[Tuple]:
        """Extract multi-pass blocking keys from Silver entity representation."""
        keys = []
        brand = silver_entity['brand']
        words = silver_entity['words']
        nums = list(silver_entity['nums'])
        zip_code = silver_entity['zip']

        tokens = [t for t in brand.split() if len(t) >= 3 and t not in self.name_stopwords]
        short_toks = [t for t in brand.split() if len(t) == 2 and t not in self.name_stopwords]

        # 1. Informative Name Tokens & Bigrams
        for t in tokens:
            keys.append(('tok', country, t))
        for i in range(len(tokens) - 1):
            keys.append(('bi', country, f"{tokens[i]}_{tokens[i+1]}"))
        if tokens and len(tokens[0]) >= 4:
            keys.append(('pref4', country, tokens[0][:4]))

        # 2. Informative Address Tokens & Word Pairs
        filtered_words = [w for w in words if w not in self.addr_stopwords]
        for w in filtered_words[:4]:
            if len(w) >= 4:
                keys.append(('addr_tok', country, w))

        for i in range(min(len(filtered_words), 4)):
            for j in range(i + 1, min(len(filtered_words), 4)):
                w1, w2 = sorted([filtered_words[i], filtered_words[j]])
                keys.append(('addr_pair', country, w1, w2))

        # 3. Numeric & Postal Combinations
        if zip_code and nums:
            for sn in nums[:3]:
                keys.append(('zip_num', country, zip_code, sn))

        if nums and filtered_words:
            for sn in nums[:2]:
                for w in filtered_words[:4]:
                    keys.append(('sn_w', country, sn, w))

        if zip_code and filtered_words:
            for w in filtered_words[:3]:
                keys.append(('zip_w', country, zip_code, w))

        # 4. Short Acronyms + Numeric Anchors
        if short_toks and (nums or zip_code):
            st = short_toks[0]
            if zip_code:
                keys.append(('st_zip', country, st, zip_code))
            for sn in nums[:2]:
                keys.append(('st_sn', country, st, sn))

        # 5. Brand + Street Number / Postal Anchors (High Recall & Precision)
        if tokens and nums:
            for sn in nums[:2]:
                keys.append(('bn_sn', country, tokens[0], sn))

        if tokens and zip_code:
            keys.append(('bn_zip', country, tokens[0], zip_code))

        # 6. Consonant Skeleton (bridges typos and phonetic transliterations)
        if tokens and len(tokens[0]) >= 4:
            t0 = tokens[0]
            skel = t0[0] + ''.join(c for c in t0[1:] if c not in 'aeiouy')
            if len(skel) >= 3:
                collapsed = [skel[0]]
                for ch in skel[1:]:
                    if ch != collapsed[-1]:
                        collapsed.append(ch)
                sk = ''.join(collapsed)[:6]
                if len(sk) >= 3:
                    keys.append(('skel', country, sk))

        return keys

    def build_index(self, entity_ids: np.ndarray, silver_entities: Any, countries: np.ndarray):
        """Build compact inverted index directly from arrays with posting length capping."""
        self.index.clear()
        self.entity_ids = entity_ids
        cap = self.max_postings + 1
        n = len(entity_ids)
        is_dict = isinstance(silver_entities, dict)
        for i in range(n):
            ent = silver_entities[entity_ids[i]] if is_dict else silver_entities[i]
            c_keys = self.extract_keys(ent, countries[i])
            for k in c_keys:
                p = self.index.get(k)
                if p is None:
                    self.index[k] = [i]
                elif len(p) < cap:
                    p.append(i)

    def query_and_rank_candidates(
        self,
        s1_entity: dict,
        country: str,
        silver_pool_lookup: Dict[str, dict]
    ) -> Tuple[List[str], List[List[float]], List[dict]]:
        """
        Queries index, applies dual-similarity ranking, prunes to Top-K,
        and extracts 24 dense features for candidate pairs.
        """
        q_keys = self.extract_keys(s1_entity, country)
        cand_indices = set()
        for k in q_keys:
            postings = self.index.get(k)
            if postings and len(postings) <= self.max_postings:
                cand_indices.update(postings)

        if not cand_indices:
            return [], [], []

        s1_b = s1_entity['brand']
        s1_a = s1_entity['addr']
        s1_nums = s1_entity['nums']
        s1_zip = s1_entity['zip']
        s1_has_a = s1_entity['has_addr']

        scored = []
        for idx in cand_indices:
            cid = str(self.entity_ids[idx]) if self.entity_ids is not None else str(idx)
            c_data = silver_pool_lookup.get(cid)
            if not c_data:
                continue

            n_sim = fuzz.token_set_ratio(s1_b, c_data['brand'])
            a_sim = fuzz.token_set_ratio(s1_a, c_data['addr']) if s1_has_a and c_data['has_addr'] else 0.0
            num_m = bool(s1_nums.intersection(c_data['nums'])) if s1_nums and c_data['nums'] else False
            zip_m = bool(s1_zip and c_data['zip'] and s1_zip == c_data['zip'])

            # Composite ranking score: preserves transliterated & aliased records
            c_score = (
                max(n_sim, a_sim, int(0.6 * n_sim + 0.4 * a_sim))
                + (20 if n_sim >= 50 and a_sim >= 40 else 0)
                + (10 if zip_m or num_m else 0)
            )
            scored.append((c_score, cid, c_data, n_sim, a_sim, num_m, zip_m))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored[:self.top_k]

        cand_ids = [x[1] for x in top_candidates]
        features = []
        meta_list = []

        for c_score, cid, c_data, n_sim, a_sim, num_m, zip_m in top_candidates:
            c_b = c_data['brand']
            c_a = c_data['addr']
            c_has_a = c_data['has_addr']

            n_ratio = fuzz.ratio(s1_b, c_b) / 100.0
            n_partial = fuzz.partial_ratio(s1_b, c_b) / 100.0
            n_tsort = fuzz.token_sort_ratio(s1_b, c_b) / 100.0
            n_tset = n_sim / 100.0
            n_wratio = fuzz.WRatio(s1_b, c_b) / 100.0

            if s1_has_a and c_has_a:
                a_ratio = fuzz.ratio(s1_a, c_a) / 100.0
                a_partial = fuzz.partial_ratio(s1_a, c_a) / 100.0
                a_tsort = fuzz.token_sort_ratio(s1_a, c_a) / 100.0
                a_tset = a_sim / 100.0
            else:
                a_ratio, a_partial, a_tsort, a_tset = 0.0, 0.0, 0.0, 0.0

            max_sim = max(n_tset, a_tset)
            comb_sim = 0.6 * n_tset + 0.4 * a_tset
            dual_high = 1.0 if (n_tset >= 0.6 and a_tset >= 0.5) else 0.0

            num_overlap = len(s1_nums.intersection(c_data['nums'])) / max(len(s1_nums), len(c_data['nums']), 1)
            num_match = 1.0 if num_m else 0.0
            num_conflict = 1.0 if (s1_nums and c_data['nums'] and not num_m) else 0.0
            zip_conflict = 1.0 if (s1_zip and c_data['zip'] and not zip_m) else 0.0
            addr_missing = 1.0 if (not s1_has_a or not c_has_a) else 0.0

            s1_l = s1_entity['legal']
            c_l = c_data['legal']
            legal_match = 1.0 if (s1_l and c_l and s1_l == c_l) else 0.0
            legal_conflict = 1.0 if (s1_l and c_l and s1_l != c_l and not (s1_l == 'pvtltd' and c_l in ('ltd', 'inc', 'corp'))) else 0.0

            name_exact = 1.0 if (s1_b and c_b and s1_b == c_b) else 0.0
            brand_len = min(len(s1_b), len(c_b)) / 30.0
            word_cnt = min(len(s1_b.split()), len(c_b.split())) / 5.0
            has_translit = 1.0 if (s1_entity.get('is_translit', False) or c_data.get('is_translit', False)) else 0.0

            feat = [
                n_ratio, n_partial, n_tsort, n_tset, n_wratio,
                a_ratio, a_partial, a_tsort, a_tset,
                max_sim, comb_sim, dual_high,
                num_overlap, num_match, num_conflict,
                1.0 if zip_m else 0.0, zip_conflict,
                addr_missing, legal_match, legal_conflict,
                name_exact, brand_len, word_cnt, has_translit
            ]
            features.append(feat)

            meta = {
                'cid': cid,
                'num_conflict': num_conflict,
                'zip_conflict': zip_conflict,
                'a_tset': a_tset,
                'n_tset': n_tset,
                'addr_missing': addr_missing,
                'legal_match': legal_match,
                'legal_conflict': legal_conflict,
                's1_legal': s1_l,
                'c_legal': c_l,
                'name_exact': name_exact,
                'is_translit': bool(has_translit)
            }
            meta_list.append(meta)

        return cand_ids, features, meta_list

    @staticmethod
    def apply_precision_gate(meta: dict, prob: float, threshold: float) -> bool:
        """
        Medallion Gold Quality Gate: Enforces multi-tier precision verification.
        Suppresses false positives caused by distractor chain stores, legal form conflicts,
        and commercial address cluster distractors.
        """
        # Gate 1: Hard negative numeric anchor conflict (unless names are virtually identical)
        if meta['num_conflict'] and meta['a_tset'] < 0.60 and meta['n_tset'] < 0.85:
            return False

        # Gate 2: Postal code conflict (unless names are virtually identical)
        if meta['zip_conflict'] and meta['a_tset'] < 0.50 and meta['n_tset'] < 0.85:
            return False

        # Gate 3: Corporate legal form mismatch
        s1_l, c_l = meta['s1_legal'], meta['c_legal']
        if s1_l and c_l and s1_l != c_l:
            if not (s1_l == 'pvtltd' and c_l in ('ltd', 'inc', 'corp')):
                if meta['a_tset'] < 0.70 and meta['n_tset'] < 0.85:
                    return False

        # Gate 4: Address missing requires high name confidence
        if meta['addr_missing'] and meta['n_tset'] < 0.80:
            return False

        # Gate 5: Commercial plaza / building distractor (unrelated names sharing address)
        if not meta.get('is_translit', False) and meta['n_tset'] < 0.35 and meta['a_tset'] < 0.90:
            return False

        # Gate 6: Calibrated ML probability threshold
        return prob >= threshold
