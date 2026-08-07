RAW_DIR    ?= data/raw
INDEX_DIR  ?= data/processed
DATASET    ?= data/datasets/UnansweredQuestions/dataset_docs_public.json
QUERY      ?= How do I configure the OpenAI compatible server?
K          ?= 10
MAX_CHUNK  ?= 2000

MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
             --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean lint lint-strict test index search answer api

install:
	uv sync

run:
	uv run python -m src search "$(QUERY)" --k $(K)

debug:
	uv run python -m pdb -m src search "$(QUERY)" --k $(K)

clean:
	rm -rf .mypy_cache .pytest_cache __pycache__ .ruff_cache
	find . -path ./.venv -prune -o -name '__pycache__' -type d -print0 \
		| xargs -0 rm -rf
	find . -path ./.venv -prune -o -name '*.py[co]' -print0 | xargs -0 rm -f

lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run pytest -q

index:
	uv run python -m src index --max_chunk_size $(MAX_CHUNK) \
		--raw_directory $(RAW_DIR) --index_directory $(INDEX_DIR)

search:
	uv run python -m src search "$(QUERY)" --k $(K)

answer:
	uv run python -m src answer "$(QUERY)" --k $(K)

api:
	uv run python -m src serve
