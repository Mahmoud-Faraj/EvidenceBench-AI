.PHONY: install test lint demo api smoke-eval scifact-data scifact-eval

install:
	pip install -r requirements-dev.txt

test:
	PYTHONPATH=src pytest -q

lint:
	python -m ruff check .
	python -m ruff format --check .

demo:
	PYTHONPATH=src streamlit run app.py

api:
	PYTHONPATH=src uvicorn api:app --reload

smoke-eval:
	PYTHONPATH=src python scripts/run_evaluation.py

scifact-data:
	PYTHONPATH=src python scripts/download_scifact.py

scifact-eval:
	PYTHONPATH=src python scripts/run_scifact_benchmark.py
