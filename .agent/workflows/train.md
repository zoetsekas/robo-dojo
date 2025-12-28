---
description: Start the full-scale distributed Robocode RL training cluster
---

// turbo-all
1. Ensure Docker is running.
2. Build the training images:
```powershell
docker compose build
```
3. Start the Ray head and workers:
```powershell
docker compose up ray-head ray-worker -d
```
4. Monitor the training logs:
```powershell
docker compose logs ray-head -f
```
