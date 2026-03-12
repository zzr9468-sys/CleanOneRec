# Tests

This directory contains unit tests for RecRL.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_trie.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=recrl tests/
```

## Test Structure

- `test_trie.py` - Tests for SID Trie implementation
- `test_rewards.py` - Tests for reward functions
- `test_data.py` - Tests for data loading and sampling
- `test_processor.py` - Tests for constrained logits processor
- `conftest.py` - Pytest configuration

## Writing Tests

When adding new features, please add corresponding tests:

1. Create a new test file `test_<feature>.py`
2. Import the module to test
3. Write test functions starting with `test_`
4. Use descriptive test names
5. Add docstrings explaining what each test does

Example:

```python
def test_my_feature():
    """Test that my feature works correctly."""
    result = my_feature(input_data)
    assert result == expected_output
```

## Test Coverage

Current test coverage:

- Trie: ✅ Full coverage
- Rewards: ✅ Full coverage
- Data: ✅ Full coverage
- Processor: ✅ Full coverage
- Trainer: ⚠️ Partial (integration tests needed)
- Algorithms: ⚠️ Partial (integration tests needed)
