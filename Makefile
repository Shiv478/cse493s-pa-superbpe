PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
TRAIN_DATA ?= data/processed/flores10_train.jsonl
EVAL_DATA ?= data/processed/flores10_test.jsonl
export PYTHONPATH := src

.PHONY: setup test smoke prepare-flores export-flores-files corpus-stats train-test results clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install -r requirements.txt

test:
	$(VENV_PYTHON) -m unittest discover -s tests

# --- prototype / sanity checks (NOT the final results; see outputs/final_results_40k/) ---
smoke:
	$(VENV_PYTHON) scripts/run_experiment.py --data data/sample_multilingual.jsonl --out outputs/prototype_demos/smoke_test

train-test:
	$(VENV_PYTHON) scripts/run_experiment.py \
		--data $(TRAIN_DATA) \
		--eval-data $(EVAL_DATA) \
		--out outputs/prototype_demos/flores10_demo \
		--stage1-merges 80 \
		--stage2-merges 40 \
		--min-frequency 2

# --- data prep ---
prepare-flores:
	$(VENV_PYTHON) scripts/prepare_flores.py \
		--source-dir data/external/flores200_dataset \
		--train-out data/processed/flores10_train.jsonl \
		--test-out data/processed/flores10_test.jsonl

export-flores-files:
	$(VENV_PYTHON) scripts/export_language_texts.py \
		--data data/processed/flores10_train.jsonl \
		--out-dir data/processed/flores10_dev_by_lang
	$(VENV_PYTHON) scripts/export_language_texts.py \
		--data data/processed/flores10_test.jsonl \
		--out-dir data/processed/flores10_devtest_by_lang

corpus-stats:
	$(VENV_PYTHON) scripts/corpus_stats.py $(TRAIN_DATA)

# --- rebuild the combined 40k results tables from the per-method evaluations ---
results:
	vendor/superbpe/.venv/bin/python scripts/make_results_summary.py

clean:
	rm -rf __pycache__ src/pa_superbpe/__pycache__ scripts/__pycache__ tests/__pycache__ outputs/prototype_demos/flores10_demo outputs/prototype_demos/smoke_test
