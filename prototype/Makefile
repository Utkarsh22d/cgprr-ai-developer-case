PY ?= python3

.PHONY: demo test data clean

demo:            ## build the dataset if missing, then run everything
	@test -f data/documents.json || $(PY) tools/build_dataset.py
	$(PY) run_all.py

test:            ## unit tests for the numeric guard, retrieval and matcher
	$(PY) -m pytest -q tests

data:
	$(PY) tools/build_dataset.py

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
