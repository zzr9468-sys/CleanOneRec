"""
Constrained Decoding with Trie

Implements prefix tree (Trie) for constraining generation to valid SIDs.
"""

from typing import Callable, List, Set
import torch


class SIDTrie:
    """
    Prefix tree for valid SID tokens.

    Ensures model can only generate valid SID sequences.
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

        # Traverse trie following current sequence
        for token_id in current_ids.tolist():
            if token_id in node:
                node = node[token_id]
            else:
                # Invalid path - allow EOS to terminate
                return {self.eos_token_id} if self.eos_token_id else set()

        # Return all valid next tokens
        allowed = set(node.keys())

        # Always allow EOS
        if self.eos_token_id:
            allowed.add(self.eos_token_id)

        return allowed

    @classmethod
    def from_recif(cls, recif_path: str, tokenizer) -> "SIDTrie":
        """
        Build trie from RecIF SID vocabulary.

        Args:
            recif_path: Path to OpenOneRec-RecIF
            tokenizer: Tokenizer for encoding SIDs

        Returns:
            SIDTrie instance
        """
        import json
        import os

        trie = cls()
        trie.eos_token_id = tokenizer.eos_token_id

        # Load SID vocabulary
        sid2pid_path = os.path.join(recif_path, "benchmark_data/sid2pid.json")
        with open(sid2pid_path, 'r') as f:
            sid2pid = json.load(f)

        # Convert hash keys back to SID strings
        # This is a simplified version - you may need to implement reverse mapping
        # For now, we'll just allow all tokens (no constraint)
        # TODO: Implement proper SID vocabulary loading

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
