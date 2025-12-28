---
description: Export the best model and run the bot in inference mode
---

// turbo-all
1. Export the latest best weights:
```powershell
docker compose exec ray-head python src/serving/export_model.py --checkpoint artifacts/checkpoints/best
```
2. Start the inference bot (connects to a local server on port 7654):
```powershell
python src/serving/inference_bot.py artifacts/serving/bot_weights.pt ws://127.0.0.1:7654 :99
```
