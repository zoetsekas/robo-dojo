# Scaled Expert Data Collection Workflow

This guide explains how to collect expert demonstrations using multiple parallel collectors and aggregate the data for training.

## Architecture

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  Collector #1   │   │  Collector #2   │   │  Collector #N   │
│                 │   │                 │   │                 │
│ Xvfb :100      │   │ Xvfb :101      │   │ Xvfb :10N      │
│ Server :7654   │   │ Server :7655   │   │ Server :765N   │
│ SimpleTarget   │   │ SimpleTarget   │   │ SimpleTarget   │
│ SimpleSpin     │   │ SimpleSpin     │   │ SimpleSpin     │
│ Observer       │   │ Observer       │   │ Observer       │
│                 │   │                 │   │                 │
│ → data_1.json  │   │ → data_2.json  │   │ → data_N.json  │
└─────────────────┘   └─────────────────┘   └─────────────────┘
         │                     │                     │
         └─────────────────────┴─────────────────────┘
                              │
                      ┌───────▼────────┐
                      │   Aggregator   │
                      │                │
                      │  Combines all  │
                      │  JSON files    │
                      │                │
                      │ → aggregated   │
                      │   _data.npz    │
                      └────────────────┘
                              │
                      ┌───────▼────────┐
                      │  Offline Pre-  │
                      │    Training    │
                      │                │
                      │  Trains RL     │
                      │  agent on      │
                      │  expert data   │
                      └────────────────┘
```

## Step 1: Run Single Collector (Smoke Test)

First, verify that a single collector works:

```bash
docker-compose run --build --rm data-collector
```

This will:
1. Start Xvfb, Robocode server, GUI
2. Launch SimpleTarget and SimpleSpin bots
3. Trigger a match (3 rounds)
4. Collect game state observations
5. Save to `artifacts/expert_data/expert_game_states_<timestamp>.json`

## Step 2: Run Multiple Collectors in Parallel

To collect data faster, run multiple collectors simultaneously:

```bash
# Terminal 1
COLLECTOR_ID=1 docker-compose run --rm data-collector

# Terminal 2  
COLLECTOR_ID=2 docker-compose run --rm data-collector

# Terminal 3
COLLECTOR_ID=3 docker-compose run --rm data-collector
```

Each collector will:
- Run independently in its own container
- Generate unique data files with timestamps
- Save to the shared `artifacts/expert_data/` directory

**Note**: Each collector needs ~2GB shared memory and runs its own Robocode stack.

## Step 3: Aggregate Collected Data

Once all collectors finish, combine the data:

```bash
docker-compose run --rm data-collector python -m src.aggregate_data
```

This will:
1. Scan `artifacts/expert_data/` for all `expert_game_states_*.json` files
2. Load and combine all game states
3. Extract observations and actions
4. Save to `artifacts/aggregated_expert_data.npz`

## Step 4: Train RL Agent (Offline Pre-training)

Use the aggregated data to pre-train your agent:

```python
import numpy as np

# Load aggregated data
data = np.load("artifacts/aggregated_expert_data.npz", allow_pickle=True)
observations = data["observations"]  # Shape: (N, 8)
actions = data["actions"]            # Shape: (N, 5)

# Train your RL agent using behavior cloning or offline RL
# ... (implement training loop)
```

## Step 5: Test Trained Agent

After pre-training, test your agent against a sample bot:

```bash
docker-compose run --rm data-collector python -m src.test_agent
```

## Data Format

### JSON Game State (per collector)
```json
[
  {
    "turn": 1,
    "round": 1,
    "bot_states": [
      {
        "x": 400.0,
        "y": 300.0,
        "direction": 90.0,
        "gunDirection": 90.0,
        "radarDirection": 90.0,
        "speed": 4.0,
        "gunHeat": 0.0,
        "energy": 100.0,
        ...
      },
      ...
    ],
    "bullet_states": [...]
  },
  ...
]
```

### Aggregated NPZ Format
```python
{
  "observations": np.array([  # Shape: (N, 8)
    [x, y, direction, gun_dir, radar_dir, speed, gun_heat, energy],
    ...
  ]),
  "actions": np.array([  # Shape: (N, 5)
    [target_speed, turn_rate, gun_turn_rate, radar_turn_rate, fire_power],
    ...
  ]),
  "metadata": {
    "total_steps": N,
    "num_collectors": M,
    "description": "..."
  }
}
```

## Scaling Guidelines

- **Single GPU**: 1-3 collectors
- **Multi-GPU**: 5-10 collectors
- **Production**: 10-20 collectors with Ray cluster

Each collector generates ~100-500 steps per run (3 rounds).
Target: **10,000+ steps** for robust offline pre-training.

## Troubleshooting

### Collector hangs or crashes
- Check `docker logs` for errors
- Ensure sufficient shared memory (`shm_size: '2gb'`)
- Verify Xvfb is starting correctly

### Game doesn't start
- Check that 2 bots joined (`SimpleTarget`, `SimpleSpin`)
- Verify controller triggered the match
- Look for `BotState` or version errors

### Empty data files
- Ensure observer connected successfully
- Check that game actually ran (multiple turns)
- Verify JSON serialization worked

## Next Steps

After successful data collection and aggregation:
1. Implement behavior cloning trainer
2. Pre-train agent on expert data
3. Fine-tune with online RL (PPO)
4. Evaluate against sample bots
5. Deploy to production battles
