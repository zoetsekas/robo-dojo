#!/usr/bin/env python3
"""
Direct BotState class patch - adds enemy_count parameter with default value.
"""
from robocode_tank_royale.bot_api import bot_state
import inspect

# Get the original __init__ method  
original_init = bot_state.BotState.__init__

# Get the signature
sig = inspect.signature(original_init)
params = list(sig.parameters.values())

# Create new __init__ that accepts enemy_count with default
def patched_init(self, *args, enemy_count=0, **kwargs):
    # Call original __init__ (which will fail if it gets enemy_count)
    # So we filter it out from kwargs if present
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'enemy_count'}
    try:
        original_init(self, *args, **filtered_kwargs)
    except TypeError as e:
        # If original init fails, try without any kwargs
        if 'enemy_count' in str(e):
            original_init(self, *args)
        else:
            raise
    
    # Store enemy_count as attribute
    self.enemy_count = enemy_count

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [Patch] %(message)s')
logger = logging.getLogger(__name__)

# Replace __init__ in the class
bot_state.BotState.__init__ = patched_init

logger.info("BotState.__init__ patched to accept enemy_count parameter!")

