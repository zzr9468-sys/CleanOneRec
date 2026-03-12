"""
Tests for SID Trie implementation.
"""

import pytest
import torch
from recrl.constraints.trie import SIDTrie


def test_trie_creation():
    """Test SIDTrie initialization."""
    trie = SIDTrie()
    assert trie.root == {}
    assert trie.eos_token_id is None


def test_trie_insert_single_sequence():
    """Test inserting a single token sequence."""
    trie = SIDTrie()
    token_ids = [100, 200, 300]

    trie.insert(token_ids)

    # Check structure (trie uses nested dicts)
    assert 100 in trie.root
    assert 200 in trie.root[100]
    assert 300 in trie.root[100][200]


def test_trie_insert_multiple_sequences():
    """Test inserting multiple token sequences."""
    trie = SIDTrie()

    seq1 = [100, 200, 300]
    seq2 = [100, 200, 400]  # Shares prefix with seq1
    seq3 = [500, 600]

    trie.insert(seq1)
    trie.insert(seq2)
    trie.insert(seq3)

    # Check seq1 and seq2 share prefix
    assert 100 in trie.root
    assert 200 in trie.root[100]
    assert 300 in trie.root[100][200]
    assert 400 in trie.root[100][200]

    # Check seq3 is separate
    assert 500 in trie.root
    assert 600 in trie.root[500]


def test_get_allowed_next_tokens_empty_completion():
    """Test getting next tokens with empty completion."""
    trie = SIDTrie()
    trie.insert([100, 200])
    trie.insert([300, 400])

    next_tokens = trie.get_allowed_next_tokens(torch.tensor([]))

    assert next_tokens == {100, 300}


def test_get_allowed_next_tokens_partial_completion():
    """Test getting next tokens with partial completion."""
    trie = SIDTrie()
    trie.insert([100, 200, 300])
    trie.insert([100, 200, 400])
    trie.insert([100, 500])

    # After [100], should get {200, 500}
    next_tokens = trie.get_allowed_next_tokens(torch.tensor([100]))
    assert next_tokens == {200, 500}

    # After [100, 200], should get {300, 400}
    next_tokens = trie.get_allowed_next_tokens(torch.tensor([100, 200]))
    assert next_tokens == {300, 400}


def test_get_allowed_next_tokens_complete_sequence():
    """Test getting next tokens after complete sequence."""
    trie = SIDTrie()
    trie.insert([100, 200])

    # After complete sequence, should return empty set
    next_tokens = trie.get_allowed_next_tokens(torch.tensor([100, 200]))
    assert next_tokens == set()


def test_get_allowed_next_tokens_invalid_path():
    """Test getting next tokens with invalid path."""
    trie = SIDTrie()
    trie.insert([100, 200])

    # Invalid path should return empty set
    next_tokens = trie.get_allowed_next_tokens(torch.tensor([999]))
    assert next_tokens == set()

    next_tokens = trie.get_allowed_next_tokens(torch.tensor([100, 999]))
    assert next_tokens == set()


def test_trie_empty():
    """Test empty trie behavior."""
    trie = SIDTrie()

    next_tokens = trie.get_allowed_next_tokens(torch.tensor([]))
    assert next_tokens == set()

    next_tokens = trie.get_allowed_next_tokens(torch.tensor([100]))
    assert next_tokens == set()


def test_trie_single_token_sequence():
    """Test trie with single-token sequences."""
    trie = SIDTrie()
    trie.insert([100])
    trie.insert([200])

    next_tokens = trie.get_allowed_next_tokens(torch.tensor([]))
    assert next_tokens == {100, 200}

    next_tokens = trie.get_allowed_next_tokens(torch.tensor([100]))
    assert next_tokens == set()


def test_trie_long_sequence():
    """Test trie with long token sequence."""
    trie = SIDTrie()
    long_seq = list(range(100, 200))  # 100 tokens

    trie.insert(long_seq)

    # Check we can traverse the entire sequence
    for i in range(len(long_seq) - 1):
        next_tokens = trie.get_allowed_next_tokens(torch.tensor(long_seq[:i+1]))
        assert long_seq[i+1] in next_tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
