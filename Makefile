# One command from a fresh clone to a scored, checked and timed model:  make demo
PY ?= python

.PHONY: install demo test report
install:
	$(PY) -m pip install -e ".[dev,tracking,onnx]"

demo:
	$(PY) -m mlproject check
	$(PY) -m mlproject train
	$(PY) -m mlproject export
	$(PY) scripts/latency_benchmark.py --repeats 100

test:
	$(PY) -m pytest -q

report:
	$(PY) -m mlproject all
