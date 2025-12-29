.PHONY: train train-resume train-selfplay train-vector train-vector-resume smoke-test smoke-test-vector export serve serve-vector collect-expert aggregate-data stop clean tensorboard help

# Default target
help:
	@echo "RoboDojo Makefile Command Reference:"
	@echo "-------------------------------------------------------------------"
	@echo "TRAINING SCENARIOS:"
	@echo "  make train               - Start training from scratch (full cluster)"
	@echo "  make train-resume        - Resume training from the best checkpoint"
	@echo "  make train-selfplay      - Start training with self-play only"
	@echo "  make train-vector        - Train with VECTOR-ONLY obs (no visual)"
	@echo "  make train-vector-resume - Resume VECTOR-ONLY training from checkpoint"
	@echo "  make smoke-test          - Run a quick validation test (multimodal)"
	@echo "  make smoke-test-vector   - Run a quick validation test (vector-only)"
	@echo ""
	@echo "SERVING & DEPLOYMENT:"
	@echo "  make export            - Convert the best checkpoint to standalone .pt"
	@echo "  make serve             - Export AND start the inference bot"
	@echo "  make serve-vector      - Serve with vector-only model"
	@echo "  make join              - Join a manual game on host (localhost:7654)"
	@echo "  make join-vector       - Join a manual game on host with vector-only bot"
	@echo ""
	@echo "EXPERT DATA COLLECTION:"
	@echo "  make collect-expert    - Collect demonstration data from sample bots"
	@echo "  make aggregate-data    - Combine collected JSON data into .npz dataset"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make tensorboard       - Launch TensorBoard to view training metrics"
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

train-vector-resume: export TRAIN_ARGS=resume=artifacts/checkpoints/best env.use_visual_obs=false
train-vector-resume:
	@echo "Resuming VECTOR-ONLY training from best checkpoint..."
	docker compose up ray-head ray-worker -d --build

# 2. SMOKE TESTS
smoke-test:
	docker compose up smoke-test --build

smoke-test-vector:
	@echo "Running smoke test with VECTOR-ONLY observations..."
	docker compose up smoke-test-vector --build

# OS-specific paths and commands
ifeq ($(OS),Windows_NT)
    PYTHON_EXE = py
    VENV_PATH = venv/Scripts/python
    PIP_PATH = venv/Scripts/pip
else
    PYTHON_EXE = python3
    VENV_PATH = venv/bin/python
    PIP_PATH = venv/bin/pip
endif

# 3. DEPLOYMENT
venv:
	@$(PYTHON_EXE) -c "import os, subprocess; subprocess.run(['$(PYTHON_EXE)', '-m', 'venv', 'venv']) if not os.path.exists('venv') else None"
	@$(PIP_PATH) install -e .

install-host: venv
	@echo "Installing host-side dependencies from requirements.txt..."
	powerShell -Command "$$env:PIP_CONFIG_FILE='pip.ini'; & $(PIP_PATH) install -r requirements.txt"

export:
	@echo "Exporting best model weights to artifacts/serving/bot_weights.pt..."
	docker compose run --rm --no-deps ray-head python -m src.serving.export_model --checkpoint /app/artifacts/checkpoints/best

export-vector:
	@echo "Exporting vector-only model weights..."
	docker compose run --rm --no-deps ray-head python -m src.serving.export_model --checkpoint /app/artifacts/checkpoints/best --vector-only

export-host: venv
	@echo "Exporting weights on host (requires Ray in venv)..."
	$(VENV_PATH) -m src.serving.export_model --checkpoint artifacts/checkpoints/best

export-vector-host: venv
	@echo "Exporting vector weights on host..."
	$(VENV_PATH) -m src.serving.export_model --checkpoint artifacts/checkpoints/best --vector-only

# Default serving params
URL ?= ws://127.0.0.1:7688
DISPLAY ?= :99
WEIGHTS ?= artifacts/serving/bot_weights.pt

serve: venv
	@if not exist $(WEIGHTS) (echo "Weights not found. Exporting..." && make export)
	@echo "Launching inference bot..."
	$(VENV_PATH) -m src.serving.inference_bot $(WEIGHTS) $(URL) $(DISPLAY)

serve-vector: venv
	@if not exist $(WEIGHTS) (echo "Weights not found. Exporting..." && make export-vector)
	@echo "Launching VECTOR-ONLY inference bot..."
	$(VENV_PATH) -m src.serving.inference_bot $(WEIGHTS) $(URL) $(DISPLAY) --vector-only

# Join a manually started game on the host
join: venv
	@if not exist $(WEIGHTS) (echo "Weights not found. Exporting..." && make export)
	@echo "Joining manual game on $(URL)..."
	$(VENV_PATH) -m src.serving.inference_bot $(WEIGHTS) $(URL) $(DISPLAY)

join-vector: venv
	@if not exist $(WEIGHTS) (echo "Weights not found. Exporting..." && make export-vector)
	@echo "Joining manual game on $(URL) [Vector-Only]..."
	$(VENV_PATH) -m src.serving.inference_bot $(WEIGHTS) $(URL) $(DISPLAY) --vector-only


# 4. DATA COLLECTION
collect-expert:
	@echo "Starting observer for expert data collection..."
	docker compose exec ray-head python src/collect_data.py

aggregate-data:
	@echo "Aggregating collected demonstrations..."
	docker compose exec ray-head python src/aggregate_data.py

# 5. MONITORING
tensorboard:
	@echo "Starting TensorBoard on http://localhost:6006..."
	docker compose exec ray-head tensorboard --logdir=/app/artifacts/logs:/root/ray_results --host=0.0.0.0 --port=6006

# 6. CLEANUP
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
