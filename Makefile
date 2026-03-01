.PHONY: help setup test test-fast test-tools run run-test clean

help:
	@echo "LSR Anchor Pipeline — Multi-Institutional Validation"
	@echo ""
	@echo "  make setup        Install dependencies"
	@echo "  make test         Classifier unit tests (~10 sec)"
	@echo "  make test-tools   Full tests including RobustiPy/DoWhy/EconML (~5 min)"
	@echo "  make run-test     200 projects per institution, fast verification"
	@echo "  make run          Full run, all institutions (~20-30 min)"
	@echo "  make clean        Clear outputs and logs"

setup:
	pip install -r requirements.txt
	mkdir -p data/wb data/adb data/aiddata outputs logs
	@echo ""
	@echo "Now add data files — see README.md for download links."

test:
	python -m pytest tests/test_classifier.py -v || python tests/test_classifier.py

test-fast:
	python tests/test_classifier.py

test-tools:
	python -m pytest tests/ -v --timeout=600

run:
	python run_pipeline.py

run-test:
	python run_pipeline.py --max 200

wb:
	python run_pipeline.py --institutions wb

adb:
	python run_pipeline.py --institutions adb

aiddata:
	python run_pipeline.py --institutions aid

clean:
	rm -f outputs/* logs/*
	@echo "Cleaned."
