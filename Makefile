PACKAGE ?= ml_xai_introduction
TRAIN := src/$(PACKAGE)/train.py
TRAINING_EPOCHS ?= 10
ARGS ?=

.PHONY: train test quality bootstrap

train:
	uv run python $(TRAIN) training.epochs=$(TRAINING_EPOCHS) $(ARGS)

test:
	uv run pytest -q

quality:
	uv run python scripts/quality_report.py

bootstrap:
	uv run python scripts/bootstrap_project.py
