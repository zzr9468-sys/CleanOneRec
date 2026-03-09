"""
SID Helper Utilities

Utilities for parsing and hashing SID (Semantic ID) strings.
"""

import re


class SIDHelper:
    """Helper for SID parsing and hashing."""

    @staticmethod
    def sid_to_hash_key(sid_str: str) -> str:
        """
        Parse SID string to hash key.

        Format: '<s_a_123><s_b_456><s_c_789>' -> hash key
        Formula: s_a * 8192^2 + s_b * 8192 + s_c

        Args:
            sid_str: SID string

        Returns:
            Hash key as string, or None if invalid
        """
        matches = re.findall(r'<s_[abc]_(\d+)>', sid_str)
        if len(matches) == 3:
            a, b, c = int(matches[0]), int(matches[1]), int(matches[2])
            return str(a * 8192 * 8192 + b * 8192 + c)
        return None

    @staticmethod
    def is_valid_sid(sid_str: str) -> bool:
        """Check if SID string is valid."""
        return SIDHelper.sid_to_hash_key(sid_str) is not None
