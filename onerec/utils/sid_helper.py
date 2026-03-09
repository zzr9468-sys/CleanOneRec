import re
import json
import logging

logger = logging.getLogger(__name__)

class SIDHelper:
    """
    Utility class for handling Semantic IDs (SID).
    Handles parsing formats like <s_a_123><s_b_456><s_c_789> into IDs and generating hash keys.
    """
    
    def __init__(self, vocab_size=8192):
        self.vocab_size = vocab_size

    def parse_sid_format(self, sid_str: str):
        """
        Extracts numbers from <s_a_123><s_b_456><s_c_789>
        Returns: tuple (a, b, c) or None if invalid
        """
        # Remove any unexpected tokens or spaces
        sid_str = sid_str.strip(" \n\"'")
        
        matches = re.findall(r'<s_[abc]_(\d+)>', sid_str)
        if len(matches) == 3:
            return int(matches[0]), int(matches[1]), int(matches[2])
        return None

    def sid_to_hash_key(self, sid_str: str) -> str:
        """
        Converts the textual SID into the long integer key used in sid2pid.json.
        Formula: a * vocab_size^2 + b * vocab_size + c
        """
        parsed = self.parse_sid_format(sid_str)
        if not parsed:
            return None
            
        a, b, c = parsed
        key_int = a * (self.vocab_size ** 2) + b * self.vocab_size + c
        return str(key_int)

    def is_valid_format(self, sid_str: str) -> bool:
        return self.parse_sid_format(sid_str) is not None

def load_json_mapping(filepath):
    logger.info(f"Loading JSON from {filepath}...")
    with open(filepath, 'r') as f:
        return json.load(f)
