import pickle
import torch
import os

path = "artifacts/checkpoints/best/policies/default_policy/policy_state.pkl"
try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("Keys in pickle:", data.keys())
    if 'weights' in data:
        print("Found weights!")
        # Attempt to save it as a standard torch weights file
        torch.save(data['weights'], "artifacts/serving/bot_weights.pt")
        print("Saved to artifacts/serving/bot_weights.pt")
except Exception as e:
    print(f"Error: {e}")
