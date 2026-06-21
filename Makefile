.PHONY: install install-dev format lint test test-unit test-integration eval ingest serve docker-up docker-down docker-logs clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,eval]"
	python -m spacy download en_core_web_lg

format:
	ruff format .
	black .

lint:
	ruff check .
	mypy lexrag/ --strict

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

eval:
	pytest evaluation/deepeval_tests.py -v

ingest:
	python -m ingestion.pipeline --directory tests/fixtures/sample_docs/

serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
