.PHONY: train smoke-test serve clean-logs help

# Default target
help:
	@echo "RoboDojo Makefile Command Reference:"
	@echo "  make train        - Start the full-scale distributed training cluster"
	@echo "  make smoke-test   - Run a quick validation smoke test"
	@echo "  make serve        - Export the best model and start the inference bot"
	@echo "  make clean-logs   - Remove all training logs and temporary artifacts"

train:
	docker compose up ray-head ray-worker -d --build

smoke-test:
	docker compose up smoke-test --build

serve:
	@echo "Exporting best model..."
	docker compose exec ray-head python src/serving/export_model.py --checkpoint artifacts/checkpoints/best
	@echo "Launching inference bot (requires XLaunch/GUI)..."
	python src/serving/inference_bot.py artifacts/serving/bot_weights.pt ws://127.0.0.1:7654 :99

clean-logs:
	rm -rf artifacts/checkpoints/*
	rm -rf artifacts/recordings/*
	docker compose down -v
