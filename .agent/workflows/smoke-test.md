---
description: Run a quick smoke test to verify the training environment
---

// turbo-all
1. Ensure Docker is running.
2. Run the smoke test container:
```powershell
docker compose up smoke-test --build
```
3. Check the logs for successful iterations:
```powershell
docker compose logs smoke-test --tail 100
```
