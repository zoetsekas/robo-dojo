.PHONY: train train-resume train-selfplay train-vector smoke-test smoke-test-vector export serve serve-vector collect-expert aggregate-data stop clean help

# Default target
help:
	@echo "RoboDojo Makefile Command Reference:"
	@echo "-------------------------------------------------------------------"
	@echo "TRAINING SCENARIOS:"
	@echo "  make train             - Start training from scratch (full cluster)"
	@echo "  make train-resume      - Resume training from the best checkpoint"
	@echo "  make train-selfplay    - Start training with self-play only"
	@echo "  make train-vector      - Train with VECTOR-ONLY obs (no visual)"
	@echo "  make smoke-test        - Run a quick validation test (multimodal)"
	@echo "  make smoke-test-vector - Run a quick validation test (vector-only)"
	@echo ""
	@echo "SERVING & DEPLOYMENT:"
	@echo "  make export            - Convert the best checkpoint to standalone .pt"
	@echo "  make serve             - Export AND start the inference bot"
	@echo "  make serve-vector      - Serve with vector-only model"
	@echo ""
	@echo "EXPERT DATA COLLECTION:"
	@echo "  make collect-expert    - Collect demonstration data from sample bots"
	@echo "  make aggregate-data    - Combine collected JSON data into .npz dataset"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make stop              - Stop the cluster (pauses training, keeps checkpoints)"
	@echo "  make clean             - Stop cluster and WIPE temporary logs/checkpoints (DANGEROUS)"
	@echo "-------------------------------------------------------------------"

# 1. TRAINING
train: export TRAIN_ARGS=
train:
	docker compose up ray-head ray-worker -d --build

train-resume: export TRAIN_ARGS=resume=artifacts/checkpoints/best
train-resume:
	@echo "Resuming training from artifacts/checkpoints/best..."
	docker compose up ray-head ray-worker -d --build

train-selfplay: export TRAIN_ARGS=self_play_only=true
train-selfplay:
	@echo "Starting self-play only training..."
	docker compose up ray-head ray-worker -d --build

train-vector: export TRAIN_ARGS=env.use_visual_obs=false
train-vector:
	@echo "Starting VECTOR-ONLY training (no visual observations)..."
	docker compose up ray-head ray-worker -d --build

# 2. SMOKE TESTS
smoke-test:
	docker compose up smoke-test --build

smoke-test-vector:
	@echo "Running smoke test with VECTOR-ONLY observations..."
	docker compose up smoke-test-vector --build

# OS-specific venv path
ifeq ($(OS),Windows_NT)
    VENV_PATH = venv/Scripts/python
    PIP_PATH = venv/Scripts/pip
else
    VENV_PATH = venv/bin/python
    PIP_PATH = venv/bin/pip
endif

# 3. DEPLOYMENT
venv:
	@python -c "import os, subprocess; subprocess.run(['python', '-m', 'venv', 'venv']) if not os.path.exists('venv') else None"
	@$(PIP_PATH) install -e .

export:
	@echo "Exporting best model weights to artifacts/serving/bot_weights.pt..."
	docker compose exec ray-head python src/serving/export_model.py --checkpoint artifacts/checkpoints/best

export-vector:
	@echo "Exporting vector-only model weights..."
	docker compose exec ray-head python src/serving/export_model.py --checkpoint artifacts/checkpoints/best --vector-only

serve: venv export
	@echo "Launching inference bot using venv..."
	$(VENV_PATH) src/serving/inference_bot.py artifacts/serving/bot_weights.pt ws://127.0.0.1:7654 :99

serve-vector: venv export-vector
	@echo "Launching VECTOR-ONLY inference bot..."
	$(VENV_PATH) src/serving/inference_bot.py artifacts/serving/bot_weights.pt ws://127.0.0.1:7654 :99 --vector-only


# 4. DATA COLLECTION
collect-expert:
	@echo "Starting observer for expert data collection..."
	docker compose exec ray-head python src/collect_data.py

aggregate-data:
	@echo "Aggregating collected demonstrations..."
	docker compose exec ray-head python src/aggregate_data.py

# 5. CLEANUP
stop:
	@echo "Stopping training cluster..."
	docker compose stop

clean:
	@echo "Cleaning up cluster and temporary artifacts..."
	docker compose down -v
	@python -c "import glob, os, shutil; [shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob('artifacts/checkpoints/*')]"
	@python -c "import glob, os; [os.remove(p) for p in glob.glob('artifacts/recordings/*.avi')]"
	@python -c "import glob, os, shutil; [shutil.rmtree(p) if os.path.isdir(p) else os.remove(p) for p in glob.glob('artifacts/logs/*')]"
	@python -c "import glob, os; [os.remove(p) for p in glob.glob('artifacts/expert_data/*.json')]"
