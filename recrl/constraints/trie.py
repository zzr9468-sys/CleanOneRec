"""
Constrained Decoding with Trie

Implements prefix tree (Trie) for constraining generation to valid SIDs.
"""

from typing import Callable, List, Set
import torch


class SIDTrie:
    """
    Prefix tree for valid SID tokens.

    Ensures model can only generate valid SID sequences during rollout.
    Build from sid2pid.json so the model is constrained to the known item space.
    """

    def __init__(self):
        self.root = {}
        self.eos_token_id = None

    def insert(self, token_ids: List[int]):
        """Insert a valid token sequence into the trie."""
        node = self.root
        for token_id in token_ids:
            if token_id not in node:
                node[token_id] = {}
            node = node[token_id]

    def get_allowed_next_tokens(self, current_ids: torch.Tensor) -> Set[int]:
        """
        Get allowed next tokens given current sequence.

        Args:
            current_ids: Current token sequence [seq_len]

        Returns:
            Set of allowed next token IDs
        """
        node = self.root

        for token_id in current_ids.tolist():
            if token_id in node:
                node = node[token_id]
            else:
                # Invalid path - allow EOS to terminate
                return {self.eos_token_id} if self.eos_token_id else set()

        allowed = set(node.keys())
        if self.eos_token_id:
            allowed.add(self.eos_token_id)

        return allowed

    @classmethod
    def from_recif(cls, recif_path: str, tokenizer) -> "SIDTrie":
        """
        Build trie from RecIF SID vocabulary.

        Reverse-engineers SID strings from sid2pid.json hash keys using:
            hash_key = a * 8192 * 8192 + b * 8192 + c
        so:
            a = hash_key // (8192 * 8192)
            b = (hash_key % (8192 * 8192)) // 8192
            c = hash_key % 8192

        Args:
            recif_path: Path to OpenOneRec-RecIF directory
            tokenizer: Tokenizer for encoding SIDs

        Returns:
            SIDTrie instance covering the full known item space
        """
        import json
        import os

        trie = cls()
        trie.eos_token_id = tokenizer.eos_token_id

        sid2pid_path = os.path.join(recif_path, "benchmark_data/sid2pid.json")
        with open(sid2pid_path, 'r') as f:
            sid2pid = json.load(f)

        print(f"[SIDTrie] Building trie from {len(sid2pid)} SIDs...")

        inserted = 0
        for hash_key_str in sid2pid.keys():
            key = int(hash_key_str)
            a = key // (8192 * 8192)
            remaining = key % (8192 * 8192)
            b = remaining // 8192
            c = remaining % 8192

            sid_str = f"<|sid_begin|><s_a_{a}><s_b_{b}><s_c_{c}><|sid_end|>"
            token_ids = tokenizer.encode(sid_str, add_special_tokens=False)
            if token_ids:
                trie.insert(token_ids)
                inserted += 1

        print(f"[SIDTrie] Inserted {inserted} SIDs into trie")
        return trie

    @classmethod
    def from_sid_list(cls, sid_list: List[str], tokenizer) -> "SIDTrie":
        """
        Build trie from list of valid SID strings.

        Args:
            sid_list: List of valid SID strings
            tokenizer: Tokenizer for encoding

        Returns:
            SIDTrie instance
        """
        trie = cls()
        trie.eos_token_id = tokenizer.eos_token_id

        for sid in sid_list:
            token_ids = tokenizer.encode(sid, add_special_tokens=False)
            trie.insert(token_ids)

        return trie
