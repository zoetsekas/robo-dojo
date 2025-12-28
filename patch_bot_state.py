#!/usr/bin/env python3
"""
Patch bot_state.py to make enemy_count parameter optional.
This fixes compatibility between server v0.24.4 and Python bot API.
"""
import sys

def patch_bot_state(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the BotState __init__ method and add enemy_count parameter
    # The parameter list might span multiple lines
    lines = content.split('\n')
    modified_lines = []
    in_bot_state_init = False
    params_added = False
    
    for i, line in enumerate(lines):
        if 'class BotState' in line:
            print(f"Found BotState class at line {i+1}")
        
        if 'def __init__(self' in line and not params_added:
            # Check if this is inside BotState class
            # Simple heuristic: look back for "class BotState"
            is_bot_state = False
            for j in range(max(0, i-20), i):
                if 'class BotState' in lines[j]:
                    is_bot_state = True
                    break
            
            if is_bot_state:
                print(f"Found __init__ at line {i+1}: {line}")
                in_bot_state_init = True
                modified_lines.append(line)
                continue
        
        if in_bot_state_init and not params_added:
            # Add enemy_count parameter before the closing parenthesis
            if ')' in line and ':' in line:
                # This is the last line of parameters
                indent = len(line) - len(line.lstrip())
                enemy_count_param = ' ' * indent + 'enemy_count: int = 0,'
                modified_lines.append(enemy_count_param)
                params_added = True
                print(f"Added enemy_count parameter before line {i+1}")
        
        modified_lines.append(line)
    
    if not params_added:
        print("WARNING: Could not find BotState.__init__ to patch!")
        return False
    
    # Write back
    with open(filepath, 'w') as f:
        f.write('\n'.join(modified_lines))
    
    print(f"Successfully patched {filepath}")
    return True

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: patch_bot_state.py <path_to_bot_state.py>")
        sys.exit(1)
    
    success = patch_bot_state(sys.argv[1])
    sys.exit(0 if success else 1)
