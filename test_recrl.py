"""
RecRL Framework Test

Quick test to verify all components are working.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from recrl.core import BaseRLTrainer, RolloutEngine, BaseReward, DataEngine
        print("  ✅ Core modules")
    except ImportError as e:
        print(f"  ❌ Core modules: {e}")
        return False

    try:
        from recrl.algorithms.grpo import GRPOTrainer, GRPOConfig
        print("  ✅ GRPO algorithm")
    except ImportError as e:
        print(f"  ❌ GRPO algorithm: {e}")
        return False

    try:
        from recrl.algorithms.eepo import EEPOTrainer, EEPOConfig
        print("  ✅ EEPO algorithm")
    except ImportError as e:
        print(f"  ❌ EEPO algorithm: {e}")
        return False

    try:
        from recrl.rewards import TextSemanticReward, ExactMatchReward
        print("  ✅ Reward functions")
    except ImportError as e:
        print(f"  ❌ Reward functions: {e}")
        return False

    try:
        from recrl.constraints import SIDTrie, ConstrainedLogitsProcessor
        print("  ✅ Constraints")
    except ImportError as e:
        print(f"  ❌ Constraints: {e}")
        return False

    try:
        from recrl.data import RepeatRandomSampler
        print("  ✅ Data utilities")
    except ImportError as e:
        print(f"  ❌ Data utilities: {e}")
        return False

    try:
        from recrl.utils import SIDHelper
        print("  ✅ Utilities")
    except ImportError as e:
        print(f"  ❌ Utilities: {e}")
        return False

    return True


def test_sid_helper():
    """Test SID helper utilities."""
    print("\nTesting SID helper...")

    from recrl.utils import SIDHelper

    helper = SIDHelper()

    # Test valid SID
    valid_sid = "<s_a_1><s_b_2><s_c_3>"
    hash_key = helper.sid_to_hash_key(valid_sid)
    if hash_key:
        print(f"  ✅ Valid SID parsing: {valid_sid} -> {hash_key}")
    else:
        print(f"  ❌ Valid SID parsing failed")
        return False

    # Test invalid SID
    invalid_sid = "invalid"
    hash_key = helper.sid_to_hash_key(invalid_sid)
    if hash_key is None:
        print(f"  ✅ Invalid SID detection")
    else:
        print(f"  ❌ Invalid SID should return None")
        return False

    return True


def test_composite_reward():
    """Test composite reward."""
    print("\nTesting composite reward...")

    from recrl.core import BaseReward, CompositeReward

    class DummyReward(BaseReward):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def __call__(self, prompts, completions, **kwargs):
            return [self.value] * len(prompts)

    reward1 = DummyReward(1.0)
    reward2 = DummyReward(2.0)

    composite = CompositeReward([
        (reward1, 0.5),
        (reward2, 0.5)
    ])

    result = composite(["test"], ["test"])
    expected = 1.5  # 0.5 * 1.0 + 0.5 * 2.0

    if abs(result[0] - expected) < 1e-6:
        print(f"  ✅ Composite reward: {result[0]} == {expected}")
    else:
        print(f"  ❌ Composite reward: {result[0]} != {expected}")
        return False

    return True


def test_config():
    """Test configuration classes."""
    print("\nTesting configurations...")

    from recrl.algorithms.grpo import GRPOConfig
    from recrl.algorithms.eepo import EEPOConfig

    grpo_config = GRPOConfig(
        num_generations=16,
        temperature=0.7,
        beta=0.04
    )
    print(f"  ✅ GRPOConfig: num_generations={grpo_config.num_generations}")

    eepo_config = EEPOConfig(
        num_generations=16,
        eepo_enabled=True,
        eepo_stage1_ratio=0.5
    )
    print(f"  ✅ EEPOConfig: eepo_stage1_ratio={eepo_config.eepo_stage1_ratio}")

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("RecRL Framework Test Suite")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("SID Helper", test_sid_helper),
        ("Composite Reward", test_composite_reward),
        ("Configurations", test_config),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    if failed == 0:
        print("\n🎉 All tests passed! RecRL is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
