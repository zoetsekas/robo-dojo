# RoboDojo Training Pipeline - Current Status & Next Steps

## ✅ What's Working

### Infrastructure
1. **Docker-Compose Ray Cluster**: Successfully configured with `ray-head` and scalable `ray-worker` services
2. **GPU Support**: Properly configured for NVIDIA GPUs with shared memory
3. **Multi-Bot Environment**: Infrastructure can launch and coordinate multiple bots:
   - Xvfb for headless rendering
   - Robocode server on port 7654
   - GUI for visualization
   - Custom Python bots (SimpleTarget, SimpleSpin)
   - Controller to trigger matches

### Data Collection Architecture
4. **Scalable Collection**: Prepared for parallel data collection with:
   - Multiple independent collector instances
   - Unique output files per collector
   - Aggregation script to combine data
   - Observer-based passive collection (no RL agent in loop)

5. **Custom Bots**: Created simple Python bots that successfully connect to the server

## ⚠️ Current Blocker

### BotState Version Mismatch

**Error**: `TypeError: BotState.__init__() missing 1 required positional argument: 'enemy_count'`

**Root Cause**: The Robocode server (v0.24.4) is sending a newer message format that includes `enemy_count`, but the Python `robocode-tank-royale` package's `BotState` class doesn't expect this field in its constructor.

**Impact**: Sample bots crash immediately when the match starts, preventing data collection.

## 🎯 Recommended Solutions (Priority Order)

### Option 1: Install robocode-tank-royale from Source ⭐ RECOMMENDED
Build and install the Python package directly from the Tank Royale repository to ensure version compatibility:

```dockerfile
# In Dockerfile, replace pip install with:
RUN cd /tmp && \
    git clone https://github.com/robocode-dev/tank-royale.git && \
    cd tank-royale/bot-api/python && \
    pip install -e .
```

**Pros**: 
- Guaranteed compatibility with server v0.24.4
- Gets latest features and bug fixes
- Direct from source = most reliable

**Cons**:
- Slightly larger Docker image
- Longer build time

### Option 2: Downgrade Server to Match Python Package
Use an older server version that matches the PyPI package:

```dockerfile
# Try server v0.20.x or v0.21.x
RUN wget https://github.com/robocode-dev/tank-royale/releases/download/v0.20.0/robocode-tankroyale-server-0.20.0.jar
```

**Pros**:
- Quick fix
- Uses stable PyPI package

**Cons**:
- Missing newer features
- May have other compatibility issues

### Option 3: Fork and Patch the Python Package
Manually patch the `BotState` class to accept `enemy_count`:

```python
# Monkey-patch in collect_data.py
from robocode_tank_royale.bot_api import bot_state
original_init = bot_state.BotState.__init__

def patched_init(self, *args, enemy_count=None, **kwargs):
    # Add enemy_count handling
    self.enemy_count = enemy_count or 0
    original_init(self, *args, **kwargs)

bot_state.BotState.__init__ = patched_init
```

**Pros**:
- Surgical fix
- No infrastructure changes

**Cons**:
- Fragile and hacky
- May break other things
- Hard to maintain

## 📋 Immediate Action Plan

1. **Try Option 1** (install from source):
   - Update Dockerfile to clone and install from Git
   - Rebuild: `docker-compose build data-collector`
   - Test: `docker-compose run --rm data-collector`

2. **If successful**, proceed with data collection:
   - Run 3-5 parallel collectors
   - Collect 10,000+ steps
   - Aggregate data

3. **Implement offline pre-training**:
   - Behavior cloning on expert data
   - Validate that agent learns basic behaviors

4. **Switch to online RL**:
   - Use pre-trained weights as initialization
   - Train with PPO in live battles
   - Fine-tune against sample bots

## 🚀 Post-Fix Roadmap

Once the version issue is resolved:

### Phase 1: Expert Data Collection (1-2 days)
- [ ] Verify single collector works end-to-end
- [ ] Scale to 5-10 parallel collectors
- [ ] Collect 50,000+ expert steps
- [ ] Aggregate and validate data quality

### Phase 2: Offline Pre-Training (2-3 days)
- [ ] Implement behavior cloning trainer
- [ ] Train `MultimodalRoboModel` on expert data
- [ ] Evaluate imitation accuracy
- [ ] Save pre-trained checkpoint

### Phase 3: Online RL Fine-Tuning (3-5 days)
- [ ] Load pre-trained weights into RLLib PPO
- [ ] Configure reward shaping
- [ ] Run distributed training on Ray cluster
- [ ] Monitor episode length, rewards, videos

### Phase 4: Evaluation & Deployment (1-2 days)
- [ ] Test agent vs. various sample bots
- [ ] Generate battle recordings
- [ ] Benchmark performance metrics
- [ ] Document findings

## 📊 Success Metrics

- **Data Collection**: 50K+ steps collected
- **Offline Pre-training**: Agent imitates expert behavior (>60% action match)
- **Online RL**: Episode length >50 turns, positive reward trend
- **Final Eval**: Win rate >30% against sample bots

## 🔗 Key Files

- `Dockerfile`: Python dependencies and Robocode jars
- `docker-compose.yml`: Ray cluster + data collector services
- `src/bots/simple_target.py`, `simple_spin.py`: Expert sample bots
- `src/collect_data.py`: Observer-based data collector
- `src/aggregate_data.py`: Data aggregation script
- `src/env/robocode_controller.py`: Match starter
- `DATA_COLLECTION.md`: Detailed workflow guide

---

**Next Immediate Step**: Update Dockerfile to install `robocode-tank-royale` from source, rebuild, and test data collection.
