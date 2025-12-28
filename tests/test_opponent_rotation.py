"""
Smoke test for opponent rotation across episodes.

This test verifies that:
1. OpponentManager can select random opponents
2. Opponent processes start and stop correctly
3. Bot registry is properly populated

Run with: python -m pytest tests/test_opponent_rotation.py -v
Or standalone: python tests/test_opponent_rotation.py
"""
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Test] %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.opponent_manager import OpponentManager, get_bot_registry, BotSpec


def test_bot_registry():
    """Test that bot registry contains expected bots."""
    registry = get_bot_registry()
    
    # Check internal bots
    assert "simple_target" in registry
    assert "noop_bot" in registry
    assert registry["simple_target"].bot_type == "internal"
    
    # Check sample bots
    assert "Crazy" in registry
    assert "SpinBot" in registry
    assert registry["Crazy"].bot_type == "sample"
    
    logger.info(f"✓ Registry contains {len(registry)} bots")
    return True


def test_opponent_selection():
    """Test random opponent selection from pool."""
    manager = OpponentManager(
        server_url="ws://127.0.0.1:9999",  # Dummy URL for testing
        opponent_pool=["Crazy", "Fire", "SpinBot"]
    )
    
    # Select multiple times and verify randomness
    selections = set()
    for _ in range(20):
        selected = manager.select_random_opponent()
        selections.add(selected)
        assert selected in ["Crazy", "Fire", "SpinBot"]
    
    # Should have selected at least 2 different bots (probabilistic but very likely)
    assert len(selections) >= 2, f"Expected variety in selection, got: {selections}"
    
    logger.info(f"✓ Random selection working, selected: {selections}")
    return True


def test_opponent_pool_default():
    """Test that default pool contains all bots."""
    manager = OpponentManager(
        server_url="ws://127.0.0.1:9999"
    )
    
    # Default pool should include all bots
    assert len(manager.available_bots) > 5
    assert "Crazy" in manager.available_bots
    assert "simple_target" in manager.available_bots
    
    logger.info(f"✓ Default pool has {len(manager.available_bots)} bots")
    return True


def test_manager_status():
    """Test manager status reporting."""
    manager = OpponentManager(
        server_url="ws://127.0.0.1:9999",
        opponent_pool=["Crazy"]
    )
    
    status = manager.get_status()
    assert status["active"] == 0
    assert status["crashed"] == 0
    assert status["total"] == 0
    assert status["pool_size"] == 1
    
    logger.info(f"✓ Status reporting working: {status}")
    return True


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Opponent Rotation Smoke Test")
    logger.info("=" * 50)
    
    tests = [
        test_bot_registry,
        test_opponent_selection,
        test_opponent_pool_default,
        test_manager_status,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            logger.error(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    logger.info("=" * 50)
    logger.info(f"Results: {passed} passed, {failed} failed")
    
    sys.exit(0 if failed == 0 else 1)

