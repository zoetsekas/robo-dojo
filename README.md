# RoboDojo: Stabilized Robocode RL Training Pipeline

RoboDojo is a high-performance, distributed reinforcement learning pipeline for training Robocode bots. It leverages Ray RLLib for PPO training and Docker for environment isolation, optimized for high-end hardware like the RTX 5090.

## 🏗️ Architecture

The pipeline uses a true distributed Ray cluster with dedicated Docker containers for each worker to ensure 100% environment isolation.

### 1. High-Level Cluster Architecture
```mermaid
graph TD
    subgraph "Host Machine"
        subgraph "Ray Head Container"
            Trainer["PPO Trainer"]
            Learner["GPU Learner"]
        end

        subgraph "Ray Worker Container 1-N"
            Worker["RolloutWorker"]
            Env["RobocodeGymEnv"]
            OpMgr["OpponentManager"]
            Server["Local Game Server"]
            BotO["Opponent Bot"]
            BotG["GymBot (RL Agent)"]
            Xvfb["Xvfb Display"]
            Video["VideoCapture"]
        end

        Trainer -- "Broadcast Policy" --> Worker
        Worker -- "Collect Rollouts" --> Trainer
        Learner -- "Update Weights" --> Trainer
        Env --> OpMgr
        OpMgr -- "Per-Episode Selection" --> BotO
    end
```

### 2. Per-Episode Training Flow
```mermaid
sequenceDiagram
    participant Trainer as PPO Trainer
    participant Worker as RolloutWorker
    participant Env as RobocodeGymEnv
    participant OpMgr as OpponentManager
    participant Bot as GymBot
    participant Server as Robocode Server

    Trainer->>Worker: Request Rollout
    Worker->>Env: reset()
    Env->>OpMgr: stop_all() + select_random_opponent()
    OpMgr->>Server: Start Opponent Process
    Env->>Bot: Connect to Server
    Env-->>Worker: Initial Observation
    
    loop Episode Steps
        Worker->>Env: step(action)
        Env->>Bot: send_action(action)
        Bot->>Server: WebSocket Command
        Server-->>Bot: TickEvent + Rewards
        Bot-->>Env: Event Queue
        Env-->>Worker: (obs, reward, done, info)
    end
    
    Worker-->>Trainer: Rollout Batch (16K steps)
    Trainer->>Trainer: PPO Update (10 epochs)
```

### 3. Component Table

| Component           | Technology          | Responsibility                                   |
| ------------------- | ------------------- | ------------------------------------------------ |
| **PPO Trainer**     | Ray RLLib / PyTorch | Central algorithm and weight updates             |
| **OpponentManager** | Python              | Per-episode random bot selection, crash recovery |
| **Local Server**    | Java (v0.34.2)      | Physics engine and game simulation               |
| **GymBot**          | Python / WebSockets | Async bridge with event buffering                |
| **VideoCapture**    | mss / OpenCV / XVID | Pixel capture + recording (AVI format)           |

### 4. Robustness Features

- **Per-Episode Opponent Selection**: Random opponent from pool each episode
- **Event Buffering**: Accumulates rewards from multiple events per step
- **Timeout Recovery**: Force-reset after 3 consecutive timeouts
- **Crash Recovery**: Auto-restart crashed opponent processes
- **Health Monitoring**: `get_health_status()` for diagnostics

---

## 🚀 User Guide

### 1. Prerequisites
- Docker & Docker Compose
- NVIDIA Container Toolkit (for GPU acceleration)
- XLaunch (if using visual debugging on Windows)

### 2. Getting Started
The project uses a **Makefile** and **Hydra** for simplified management.

```bash
# Start full-scale training (12 parallel workers)
make train

# Run a quick smoke test
make smoke-test

# Monitor logs
docker compose logs ray-head -f
```

### 3. Configuration (Hydra)
Settings are organized hierarchically in `config/`. You can override any value via CLI:
```bash
# Override learning rate and worker count
python -m src.train training.lr=1e-4 hardware.num_workers=5
```

### 4. Serving (Deployment)
Once a model is trained, export and run it as a standalone bot:
```bash
# Export the best weights
make serve
# (This runs export_model.py and starts inference_bot.py)
```

---

## 🛠️ Project Management
- **`pyproject.toml`**: Dependency and build configuration.
- **`Makefile`**: Quick-access shortcuts for training and serving.
- **`.agent/workflows/`**: Interactive workflows for the AI assistant.
- **`LICENSE`**: MIT Licensed.

---

## 📊 Verification Status
- ✅ **Distributed Isolation**: Resolved "Bots joined" collisions via strict loopback binding.
- ✅ **Memory Stability**: JVM capping and Docker limits prevent OOM crashes.
- ✅ **Numerical Stability**: Observation normalization ensures no NaN loss.
- ✅ **Async Robustness**: GymBot correctly handles high-frequency async server events.

---

## 📜 Citation

If you use RoboDojo in your research or project, please cite it as:

```bibtex
@software{robodojo2025,
  author = {{Zoe Tsekas}},
  title = {RoboDojo: Stabilized Robocode RL Training Pipeline},
  version = {0.1.0},
  date = {2025-12-27},
  url = {https://github.com/zoetsekas/robo-dojo},
  license = {MIT}
}
```

