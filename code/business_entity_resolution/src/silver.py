"""
Medallion Architecture - Silver Layer: Entity Normalization and Canonicalization.
Standardizes business names, isolates brand cores, extracts legal forms,
expands address abbreviations, transliterates non-Latin scripts (Devanagari/Hindi),
and isolates discrete numeric/postal anchors.
"""

import re
import unicodedata
from typing import Dict, Tuple, Set, List, Optional, Any

# Street standardizations
STREET_ABBRS = {
    r'\bst\b': 'street',
    r'\brd\b': 'road',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bln\b': 'lane',
    r'\bdr\b': 'drive',
    r'\bct\b': 'court',
    r'\bhwy\b': 'highway',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bbldg\b': 'building',
    r'\bfl\b': 'floor',
    r'\bpkwy\b': 'parkway',
    r'\bpl\b': 'place',
    r'\bsq\b': 'square',
    r'\bter\b': 'terrace',
    r'\bno\b': 'number',
    r'\bsec\b': 'sector',
    r'\bblk\b': 'block',
}

# Legal corporate forms (English + Hindi Devanagari)
LEGAL_FORMS = [
    (r'\b(private\s+limited|pvt\s+ltd|pvt\.\s*ltd|pvt\s+limited|private\s+ltd)\b', 'pvtltd'),
    (r'\b(public\s+limited|public\s+ltd)\b', 'publicltd'),
    (r'(पर\s*इवट\s*ल\s*म\s*टड|पाइवट\s*लिमिटेड|प्राइवेट\s*लिमिटेड)', 'pvtltd'),
    (r'(पबलक\s*ल\s*म\s*टड|पब्लिक\s*लिमिटेड)', 'publicltd'),
    (r'(ल\s*म\s*टड|लिमिटेड)', 'ltd'),
    (r'\b(limited\s+liability\s+company|l\.l\.c\.|llc)\b', 'llc'),
    (r'\b(incorporated|inc\.|inc)\b', 'inc'),
    (r'\b(corporation|corp\.|corp)\b', 'corp'),
    (r'\b(limited\s+liability\s+partnership|l\.l\.p\.|llp)\b', 'llp'),
    (r'\b(limited|ltd\.|ltd)\b', 'ltd'),
    (r'\b(company|co\.|co)\b', 'co'),
    (r'\b(s\.a\.r\.l\.|sarl)\b', 'sarl'),
    (r'\b(s\.a\.|sa)\b', 'sa'),
    (r'\b(gmbh)\b', 'gmbh'),
    (r'\b(s\.c\.i\.|sci)\b', 'sci'),
]

# Devanagari transliteration mapping
DEVANAGARI_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'ं': 'n', 'ँ': 'n', 'ः': 'h', '्': '', '़': '',
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

DOMAIN_RE = re.compile(r'\.(com|org|net|in|co|io|fr|gov|edu|biz|info)\b', re.IGNORECASE)
PREFIX_RE = re.compile(r'(@|https?://|www\.)', re.IGNORECASE)
PUNCT_RE = re.compile(r'[^\w\s]', re.UNICODE)
NUMBERS_RE = re.compile(r'\b\d+\b')

def strip_accents(s: str) -> str:
    """Normalize unicode and strip combining diacritics."""
    if not s:
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

def transliterate_devanagari(text: str) -> str:
    """Fast phonetic transliteration of Devanagari Hindi text to Latin alphabet."""
    if not text:
        return ""
    if not any('\u0900' <= c <= '\u097f' for c in text):
        return text
    res = ''.join(DEVANAGARI_MAP.get(c, c) for c in text)
    # Merge space-separated single characters (e.g., 'r j b s t' -> 'rj bst')
    words = res.split()
    merged = []
    buf = []
    for w in words:
        if len(w) == 1 and w.isalpha():
            buf.append(w)
        else:
            if buf:
                merged.append(''.join(buf))
                buf = []
            merged.append(w)
    if buf:
        merged.append(''.join(buf))
    return ' '.join(merged)

def clean_brand_name(raw_name: Any) -> Tuple[str, str, bool]:
    """
    Standardize raw name into (brand_core, legal_form, has_translit).
    Strips URL prefixes, domains, handles, extracts legal form, and transliterates.
    """
    if raw_name is None or str(raw_name).lower() in ('nan', 'null', 'none', '<null>'):
        return ("", "", False)
    
    raw_str = str(raw_name)
    has_devanagari = any('\u0900' <= c <= '\u097f' for c in raw_str)
    
    s = strip_accents(raw_str.lower().strip())
    s = PREFIX_RE.sub(' ', s)
    s = DOMAIN_RE.sub(' ', s)
    
    detected_legal = ""
    for pattern, canon in LEGAL_FORMS:
        if re.search(pattern, s):
            detected_legal = canon
            s = re.sub(pattern, ' ', s)
            break
            
    if has_devanagari:
        s = transliterate_devanagari(s)
        
    s = PUNCT_RE.sub(' ', s)
    brand_core = ' '.join(s.split())
    return (brand_core, detected_legal, has_devanagari)

def clean_address(raw_addr: Any) -> Tuple[str, str, Set[str], List[str], bool]:
    """
    Standardize address into (canonical_address, postal_code, street_numbers_set, street_words, has_translit).
    Expands street abbreviations, transliterates Devanagari, and isolates numeric/postal anchors.
    """
    if raw_addr is None or str(raw_addr).lower() in ('nan', 'null', 'none', '<null>'):
        return ("", "", set(), [], False)
        
    raw_str = str(raw_addr)
    has_devanagari = any('\u0900' <= c <= '\u097f' for c in raw_str)
    
    s = strip_accents(raw_str.lower().strip())
    if has_devanagari:
        s = transliterate_devanagari(s)
    s = PUNCT_RE.sub(' ', s)
    for pattern, full_word in STREET_ABBRS.items():
        s = re.sub(pattern, full_word, s)
    s = ' '.join(s.split())
    
    nums = NUMBERS_RE.findall(s)
    zips = [n for n in nums if len(n) in (5, 6)]
    zip_code = zips[0] if zips else ""
    street_nums = set(n for n in nums if n != zip_code and len(n) <= 5)
    
    words = [w for w in s.split() if len(w) >= 3 and not w.isdigit()]
    return (s, zip_code, street_nums, words, has_devanagari)

def normalize_silver_entity(name: Any, addr: Any) -> dict:
    """Normalize raw entity into structured Silver representation."""
    brand, legal, name_dev = clean_brand_name(name)
    c_addr, zip_code, street_nums, words, addr_dev = clean_address(addr)
    return {
        'brand': brand,
        'legal': legal,
        'addr': c_addr,
        'has_addr': bool(c_addr),
        'nums': street_nums,
        'zip': zip_code,
        'words': words,
        'is_translit': (name_dev or addr_dev)
    }
