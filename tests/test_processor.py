"""
Tests for constrained logits processor.
"""

import pytest
import torch
from recrl.constraints.processor import ConstrainedLogitsProcessor


def test_processor_initialization():
    """Test ConstrainedLogitsProcessor initialization."""

    def get_allowed_tokens_fn(completion_ids):
        return {100, 200, 300}

    processor = ConstrainedLogitsProcessor(
        get_allowed_tokens_fn=get_allowed_tokens_fn,
        prompt_length=10
    )

    assert processor.prompt_length == 10
    assert processor.get_allowed_tokens_fn is not None


def test_processor_no_constraints():
    """Test processor when no constraints are applied."""

    def get_allowed_tokens_fn(completion_ids):
        return set()  # Empty set means no constraints

    processor = ConstrainedLogitsProcessor(
        get_allowed_tokens_fn=get_allowed_tokens_fn,
        prompt_length=5
    )

    batch_size = 2
    vocab_size = 1000
    seq_len = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    scores = torch.randn(batch_size, vocab_size)

    original_scores = scores.clone()
    modified_scores = processor(input_ids, scores)

    # Should not modify scores when no constraints
    assert torch.allclose(modified_scores, original_scores)


def test_processor_with_constraints():
    """Test processor applies constraints correctly."""

    allowed_tokens = {100, 200, 300}

    def get_allowed_tokens_fn(completion_ids):
        return allowed_tokens

    processor = ConstrainedLogitsProcessor(
        get_allowed_tokens_fn=get_allowed_tokens_fn,
        prompt_length=5
    )

    batch_size = 2
    vocab_size = 1000
    seq_len = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    scores = torch.randn(batch_size, vocab_size)

    modified_scores = processor(input_ids, scores)

    # Check that disallowed tokens have -inf scores
    for i in range(batch_size):
        for token_id in range(vocab_size):
            if token_id in allowed_tokens:
                # Allowed tokens should keep original scores
                assert modified_scores[i, token_id] == scores[i, token_id]
            else:
                # Disallowed tokens should be -inf
                assert modified_scores[i, token_id] == float('-inf')


def test_processor_device_consistency():
    """Test processor handles different devices correctly."""

    def get_allowed_tokens_fn(completion_ids):
        return {100, 200}

    processor = ConstrainedLogitsProcessor(
        get_allowed_tokens_fn=get_allowed_tokens_fn,
        prompt_length=5
    )

    batch_size = 2
    vocab_size = 1000
    seq_len = 10

    # Test on CPU
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    scores = torch.randn(batch_size, vocab_size)

    modified_scores = processor(input_ids, scores)
    assert modified_scores.device == scores.device

    # Test on CUDA if available
    if torch.cuda.is_available():
        input_ids_cuda = input_ids.cuda()
        scores_cuda = scores.cuda()

        modified_scores_cuda = processor(input_ids_cuda, scores_cuda)
        assert modified_scores_cuda.device == scores_cuda.device


def test_processor_batch_independence():
    """Test processor handles each batch item independently."""

    call_count = [0]

    def get_allowed_tokens_fn(completion_ids):
        call_count[0] += 1
        # Return different allowed tokens based on call count
        if call_count[0] % 2 == 1:
            return {100, 200}
        else:
            return {300, 400}

    processor = ConstrainedLogitsProcessor(
        get_allowed_tokens_fn=get_allowed_tokens_fn,
        prompt_length=5
    )

    batch_size = 2
    vocab_size = 1000
    seq_len = 10

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    scores = torch.randn(batch_size, vocab_size)

    modified_scores = processor(input_ids, scores)

    # Should have called get_allowed_tokens_fn for each batch item
    assert call_count[0] == batch_size


def test_processor_extracts_completion_correctly():
    """Test processor extracts completion part correctly."""

    extracted_completions = []

    def get_allowed_tokens_fn(completion_ids):
        extracted_completions.append(completion_ids.tolist())
        return {100}

    prompt_length = 5
    processor = ConstrainedLogitsProcessor(
        get_allowed_tokens_fn=get_allowed_tokens_fn,
        prompt_length=prompt_length
    )

    batch_size = 1
    vocab_size = 1000
    seq_len = 10

    input_ids = torch.arange(seq_len).unsqueeze(0)  # [0, 1, 2, ..., 9]
    scores = torch.randn(batch_size, vocab_size)

    processor(input_ids, scores)

    # Should extract tokens after prompt_length
    expected_completion = list(range(prompt_length, seq_len))
    assert extracted_completions[0] == expected_completion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
