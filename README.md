*This project has been created as part of the 42 curriculum by fbugri.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation system that answers questions about the
[vLLM](https://github.com/vllm-project/vllm) codebase.

A language model cannot know a codebase it was not trained on. Instead of
retraining one, this project indexes the repository, retrieves the handful of
snippets that actually answer a question, and hands only those to a small local
model (`Qwen/Qwen3-0.6B`) so its answer stays grounded in real source code.

The pipeline has four stages:

1. **Index** — walk `data/raw/`, split every file into chunks, build a BM25
   inverted index and persist it under `data/processed/`.
2. **Retrieve** — score every chunk against the question and return the top-k
   source locations (`file_path` plus character range).
3. **Augment** — read those spans back out of the corpus and pack them into the
   model's context window.
4. **Generate** — produce a natural-language answer constrained to that context.

Retrieval quality is measured with recall@k. On the public datasets the system
scores **85.0% recall@5 on docs** (threshold 80%) and **87.9% recall@5 on code**
(threshold 50%).

## Instructions

### Installation

```bash
make install          # uv sync
```

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/). The vLLM corpus
must sit under `data/raw/` and the question datasets under
`data/datasets/{Answered,Unanswered}Questions/`; neither is committed.

### Running the full pipeline

```bash
# 1. Index the corpus once (~13 s, well under the 5 min budget)
uv run python -m src index --max_chunk_size 2000

# 2. Search a whole dataset, scoping the output directory by dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 3. Score the retrieval with the provided grader
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000

# 4. Generate answers from those search results
uv run python -m src answer_dataset \
    --student_search_results_path \
        data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Development

```bash
make lint             # flake8 + mypy
make lint-strict      # flake8 + mypy --strict (also passes)
make test             # pytest
make clean            # drop caches
```

## Command-Line Interface

Every command is `uv run python -m src <command> [options]`. All input and
output paths are options; none is hard-coded.

| Command | Key options | Purpose |
|---|---|---|
| `index` | `--max_chunk_size`, `--raw_directory`, `--index_directory`, `--rebuild`, `--embeddings` | Ingest the corpus and persist the index |
| `search` | `<query>`, `--k`, `--index_directory`, `--hybrid`, `--alpha` | Top-k sources for one query |
| `search_dataset` | `--dataset_path`, `--k`, `--save_directory`, `--hybrid` | Batch retrieval, writes `StudentSearchResults` |
| `answer` | `<query>`, `--k`, `--max_sources` | Answer one query from retrieved context |
| `answer_dataset` | `--student_search_results_path`, `--save_directory`, `--limit` | Batch answering, writes `StudentSearchResultsAndAnswer` |
| `evaluate` | `--student_search_results_path`, `--dataset_path`, `--k` | Local recall@k, for iteration only |
| `serve` | `--host`, `--port` | Local HTTP API (bonus) |

`evaluate` is a development aid. The official score comes from the provided
`moulinette` executable, which this project never imports or calls.

## Example usage

```console
$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/docs/getting_started/quickstart.md [8079:9261]
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0:1951]
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [5901:6213]
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [3936:3958]
data/raw/vllm-0.10.1/docs/getting_started/quickstart.md [6294:8077]

$ uv run python -m src answer \
    "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?" --k 5
data/raw/vllm-0.10.1/docs/features/lora.md [4695:6098]
...
The HTTP endpoint used to dynamically load a LoRA adapter is
`/v1/load_lora_adapter`.
```

## System architecture

```
data/raw/  ──▶ chunking ──▶ tokenization ──▶ BM25 index ──▶ data/processed/
                                                  │
                question ──▶ retrieval ◀──────────┘
                                 │
                                 ├──▶ StudentSearchResults (JSON)
                                 │
                                 └──▶ augmentation ──▶ Qwen3-0.6B ──▶ answers (JSON)
```

| Module | Responsibility |
|---|---|
| `src/models.py` | The pydantic models validating every stage boundary |
| `src/chunking.py` | The two chunking strategies, emitting exact character spans |
| `src/tokenization.py` | Identifier-aware tokenizer shared by indexing and querying |
| `src/indexing.py` | Corpus walk, document cache, index construction and persistence |
| `src/retrieval.py` | BM25 scoring, top-k selection and optional hybrid reranking |
| `src/embedding.py` | Optional dense chunk vectors (bonus) |
| `src/pipeline.py` | Dataset-level orchestration and JSON input/output |
| `src/generation.py` | Prompt construction and grounded answer generation |
| `src/evaluation.py` | Local recall@k with IoU matching |
| `src/api.py` | Local HTTP API |
| `src/__main__.py` | The Fire CLI |

The stages only ever exchange validated pydantic models, so a malformed dataset
fails at the boundary with a readable message rather than deep inside scoring.

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so two
strategies are implemented.

**Python (`chunk_python`)** parses the file with `ast` and cuts on statement
boundaries. Consecutive top-level statements are packed together while they fit
the size cap; a class or function too large on its own is split across its own
body, so the methods of a long class become separate chunks. Spans are built
contiguously from one statement's start to the next one's, which means comments,
decorators and blank lines between definitions are never dropped. A file that
fails to parse falls back to the text strategy instead of disappearing from the
index.

**Markdown and text (`chunk_text`)** cuts on ATX headings. A section is kept
whole when it fits, merged with its neighbours when it is small, and split on
paragraph then line boundaries when it is too large.

Two properties hold for both strategies and are enforced by the test suite:

- A chunk is always a **verbatim slice** of the file: `text[first:last]` returns
  exactly the indexed body. This is what makes the reported character ranges
  trustworthy.
- A span never exceeds `--max_chunk_size` (default 2000). This matters because
  the grader rejects any source longer than its `max_context_length`, and a
  single over-long span invalidates the whole output file.

Each chunk also carries a **context** string — its file path plus the enclosing
heading trail or the qualified name of the enclosing definition. The context is
indexed with extra weight but never widens the emitted span.

## Retrieval method

Retrieval is **BM25** (`k1 = 1.5`, `b = 0.75`) over the chunk index.

The interesting part is the tokenizer. Questions and source code rarely spell
things the same way: a question asks about the "fused batched MoE layer" while
the code says `fused_batched_moe`. So every identifier is emitted **both whole
and split** into its snake_case and camelCase sub-words — `AsyncLLMEngine`
indexes as `asyncllmengine`, `async`, `llm`, `engine`. A shallow plural rule
folds `endpoints` onto `endpoint` without the over-stemming a full stemmer
brings to identifier-heavy text. Both sides of the match use the same
tokenizer, so the folding is always symmetric.

Fields are weighted rather than duplicated: context terms (path, headings,
symbol names) count three times a body occurrence, which lets a question
naming a concept reach the right file even when the body phrases it differently.

The index is stored term-major in CSR form — an `indptr` array delimiting each
term's slice of a flat `doc_ids`/`term_freqs` pair. Scoring a query is then a
handful of NumPy slice operations, and loading the index is a few array reads
rather than a walk over a nested Python structure.

Selection caps a single file at **3 chunks** inside the top-k. Without the cap,
one strongly matching file floods the whole result list and a near-miss costs
every slot; with it, the list keeps enough diversity to recover.

## Performance analysis

Measured on a CPU-only campus machine over the 1,969 indexable files
(1,847 non-empty) of vLLM 0.10.1, producing 21,355 chunks.

| Metric | Requirement | Measured |
|---|---|---|
| Indexing time | ≤ 5 min | **12.6 s** |
| Retrieval throughput | ≤ 90 s / 200 questions | **≈ 1 s / 200 questions** (index load included) |
| Recall@5, docs | ≥ 80% | **85.0%** |
| Recall@5, code | ≥ 50% | **87.9%** |

Full recall curve, as reported by the provided `moulinette`:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| `dataset_docs_public` | 65.0% | 81.0% | 85.0% | 88.0% |
| `dataset_code_public` | 61.6% | 85.9% | 87.9% | 90.9% |

Two observations. First, most of the gain sits between k=1 and k=3: the right
*file* is usually found immediately, and the extra slots mostly buy the right
*region* of it. Second, code scores higher than docs beyond k=1 even though its
threshold is lower — identifiers are far more discriminative than prose, so once
a question quotes a symbol the matching chunk is almost always in the top three.
Docs questions paraphrase more, which is exactly what the sub-word tokenizer and
the heading-weighted context are there to absorb.

Chunk size trades coverage against precision. Smaller chunks sharpen BM25 length
normalisation but need more of them to cover the region a question is about;
2000 characters, the largest the grader accepts, measured best here.

Answer generation is the slow stage: `Qwen3-0.6B` in float32 on CPU takes
roughly 20–45 s per question after a one-off model load. Retrieval is unaffected
— the deep-learning stack is imported lazily, so search-only commands never pay
for it.

## Design decisions

- **BM25 over TF-IDF.** Both were allowed; BM25's term-frequency saturation and
  length normalisation matter on a corpus mixing 40-character docstrings with
  2000-character source spans.
- **Character spans, never re-serialised text.** Chunks are stored as offsets
  and sliced back out of the corpus on demand. The index stays small and a span
  can never drift from the file it describes.
- **Weighted fields instead of repeated tokens.** Boosting by repeating the path
  in the chunk body would inflate the document length and fight BM25's own
  normalisation; weighting the term frequency directly does not.
- **Lazy model loading.** `transformers` and `torch` are imported inside the
  generation path, keeping the retrieval commands fast to start.
- **Errors as domain exceptions.** `IndexError_`, `PipelineError` and
  `GenerationError` are caught at the CLI boundary and printed as one-line
  messages. Empty queries, `k=0`, missing files, malformed JSON and a missing
  index all exit cleanly instead of raising a traceback.

## Challenges faced

- **Character offsets from the AST.** `ast` reports `col_offset` as a UTF-8
  *byte* offset while the required indices are *character* offsets. On files
  with accented characters the two diverge and every span after the first
  non-ASCII line shifts. Offsets are now converted per line, and a test chunks a
  non-ASCII document to keep the round-trip honest.
- **Losing text between definitions.** A first version derived each chunk from
  its own node's span, which silently dropped the comments and blank lines
  between two definitions — and some reference spans start exactly there. Spans
  are now built contiguously, so the chunks of a file tile it completely.
- **The 2000-character trap.** One over-long source invalidates the entire
  output file, not just that question. The cap is therefore enforced at the
  single point where a chunk is emitted, and asserted in the chunking tests
  rather than trusted.
- **Vocabulary mismatch.** Plain whitespace tokenization scored poorly on docs
  questions, which paraphrase instead of quoting. Sub-word splitting of
  identifiers plus the weighted context string is what moved docs recall@5 past
  the threshold.
- **pytest and mypy discovering the corpus.** With the vLLM repository sitting
  under `data/`, both tools tried to analyse it. `testpaths` and a mypy
  `exclude` keep them on the project's own code.

## Bonus

- **Local HTTP API** (`uv run python -m src serve`, or `make api`) exposing
  `GET /health`, `POST /search` and `POST /answer`. The index is loaded once at
  start-up and the model lazily on the first answer, so search stays available
  without model weights.

## Resources

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*
  (2009) — the scoring function and its parameters.
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*,
  chapters 1–6 — inverted indexes, tokenization and evaluation metrics.
- [vLLM documentation](https://docs.vllm.ai/) — the corpus under test.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — chat template and
  the `enable_thinking` switch.
- [Python `ast` documentation](https://docs.python.org/3/library/ast.html) —
  node positions and their byte-offset caveat.
- [Python Fire](https://github.com/google/python-fire),
  [pydantic](https://docs.pydantic.dev/) and [uv](https://docs.astral.sh/uv/)
  guides.

### Use of AI

AI assistance was used as a reviewer and an accelerator, not as an author of
record:

- **Design discussion** — comparing BM25 against TF-IDF and weighing chunking
  strategies before writing any code.
- **Boilerplate** — first drafts of the pydantic models and the CLI wiring,
  which are mechanical transcriptions of the subject's specification.
- **Debugging** — narrowing down the UTF-8 byte-offset bug in the AST chunker
  and the pytest/mypy corpus-discovery issues.
- **Documentation** — drafting and tightening this README.

The retrieval logic, the chunking invariants and the tuning decisions were
written, measured and validated against the moulinette by hand. Every generated
suggestion was read, tested and rewritten where it was wrong; nothing in this
repository is code I could not explain and defend.
