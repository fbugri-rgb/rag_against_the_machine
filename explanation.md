# RAG against the machine: how the code works

This document explains the project the way you would present it to someone:
first the one-minute version, then the walkthrough of one question through the
system, then each file in detail, and finally the questions people usually ask.

---

## Part 1: The high-level summary

### The one-sentence version

> A search engine over the vLLM source code that finds the few snippets
> relevant to a question, then hands only those snippets to a small local
> language model so it can answer from real code instead of guessing.

### The problem

A language model only knows what it was trained on. Ask a small model like
`Qwen3-0.6B` about the internals of vLLM and it will invent function names.
Retraining it on vLLM is expensive. **RAG (Retrieval-Augmented Generation)** is
the cheap alternative: look the answer up first, then let the model *read* it.

### The analogy

Think of an open-book exam:

| Exam | This project |
|---|---|
| The textbook | The vLLM repository (`data/raw/`) |
| The index at the back of the book | The BM25 inverted index (`data/processed/`) |
| Looking up the right pages | Retrieval (`retrieval.py`) |
| Photocopying just those pages | Augmentation (`build_context` in `generation.py`) |
| Writing the answer from the photocopies | Generation with Qwen3 |

### The four stages

```
 ┌──────────┐    ┌────────────┐    ┌───────────┐    ┌────────────┐
 │ 1. INDEX │ ─▶ │ 2. RETRIEVE│ ─▶ │ 3. AUGMENT│ ─▶ │ 4. GENERATE│
 └──────────┘    └────────────┘    └───────────┘    └────────────┘
  done once       per question      per question     per question
  ~13 s           ~5 ms             read spans        20–45 s on CPU
```

1. **Index:** split every file of vLLM into chunks of at most 2000 characters,
   turn each chunk into words ("tokens"), and build a lookup table from each
   word to the chunks that contain it. Save it to disk.
2. **Retrieve:** turn the question into tokens, score every chunk with BM25,
   and return the top-k as `(file_path, start, end)` locations.
3. **Augment:** read those character ranges back out of the files and paste
   them into a prompt.
4. **Generate:** Qwen3-0.6B writes the answer, told to use only those excerpts.

### The results in one table

| Metric | Required | Achieved |
|---|---|---|
| Indexing time | ≤ 5 min | 12.6 s |
| 200 searches | ≤ 90 s | ≈ 1 s |
| Recall@5 on docs questions | ≥ 80% | 85.0% |
| Recall@5 on code questions | ≥ 50% | 87.9% |

**Recall@5** means: of the reference sources a question needs, what share
shows up in my top 5 results.

### The three ideas that make it work

If you only have time to explain three things, explain these:

1. **Smart chunking.** Python is cut along functions and classes (using
   Python's own parser), Markdown along headings. A chunk is always an exact
   slice of the original file, so the reported character ranges are correct.
2. **An identifier-aware tokenizer.** `AsyncLLMEngine` is indexed as
   `asyncllmengine`, `async`, `llm` *and* `engine`, so a question in plain
   English still matches code written in camelCase or snake_case.
3. **Context weighting.** Each chunk also carries its file path, heading trail
   or class/function name. Those words count 3× as much, so a question that
   names a concept finds the file that is *about* that concept.

---

## Part 2: Following one question through the system

The easiest way to explain the code is to follow one question end to end.

**Question:** *"What HTTP endpoint is used to dynamically load a LoRA adapter?"*

### Step 0: Indexing (happened earlier, once)

`python -m src index` ran `build_document_cache` (`indexing.py`):

- It walked `data/raw/` and found `docs/features/lora.md`.
- `chunk_text` (`chunking.py`) split it on its `#` headings. One chunk covers
  characters `4695–6098`, the section *"### Using API Endpoints"*.
- That chunk's **body** was tokenized, each word counting 1. Its **context**,
  the path plus the heading trail (`.../lora.md LoRA Adapters Dynamically
  serving LoRA Adapters Using API Endpoints`), was tokenized too, each word
  counting 3.
- Everything was merged into one inverted index and saved as NumPy arrays.

### Step 1: Tokenize the question

`tokenize()` (`tokenization.py`) turns the question into:

```
["http", "endpoint", "dynamically", "load", "lora", "adapter"]
```

Stop words (`what`, `is`, `to`, `a`) are dropped and `adapters` folds to
`adapter`.

### Step 2: Score every chunk

`Retriever.score()` (`retrieval.py`) looks each of those words up in the index.
For each word it gets the list of chunks containing it and adds that word's
BM25 contribution to those chunks' scores. Rare words like `lora` count far
more than common words like `load`.

### Step 3: Pick the top k

`Retriever._select()` walks the chunks from best to worst score and keeps
at most **3 chunks per file**, so one file cannot fill the whole list.

### Step 4: Output

```
data/raw/vllm-0.10.1/docs/features/lora.md [4695:6098]
...
```

For `search_dataset`, these locations are written to a JSON file that the
grader (`moulinette`) scores.

### Step 5: Answer (only for `answer` / `answer_dataset`)

`build_context()` reads characters `4695:6098` of `lora.md` and numbers it
`[1]`. `AnswerGenerator.answer()` builds a chat prompt (system rules, then the
excerpts, then the question) and runs Qwen3 deterministically (`do_sample=False`):

> The HTTP endpoint used to dynamically load a LoRA adapter is
> `/v1/load_lora_adapter`.

---

## Part 3: The files in detail

### Map of the repository

```
rag_against_the_machine/
├── src/                 ← all the code
│   ├── __main__.py      CLI: the commands you type
│   ├── models.py        data shapes (pydantic)
│   ├── chunking.py      file → chunks
│   ├── tokenization.py  text → words
│   ├── indexing.py      chunks → inverted index on disk
│   ├── retrieval.py     question → top-k chunks (BM25)
│   ├── embedding.py     optional: dense vectors for hybrid search (bonus)
│   ├── pipeline.py      run over a whole dataset, read/write JSON
│   ├── generation.py    chunks + question → answer (Qwen3)
│   ├── evaluation.py    local recall@k score
│   └── api.py           HTTP server (bonus)
├── tests/               pytest suite
├── data/datasets/       question datasets (public)
├── Makefile             install / lint / test / run shortcuts
└── pyproject.toml       dependencies (managed by uv)
```

**Dependency order** (each file only uses the ones above it):

```
models, tokenization, chunking
        ↓
     indexing
        ↓
embedding → retrieval
        ↓
     pipeline → generation, evaluation
        ↓
   __main__ (CLI), api
```

---

### `models.py`: the data contract

All stages talk to each other only through these **pydantic** models. Pydantic
checks the types when data comes in, so a malformed JSON file fails
immediately with a clear message instead of crashing deep inside the code.

| Model | What it is |
|---|---|
| `MinimalSource` | One location: `file_path`, `first_character_index` (inclusive), `last_character_index` (exclusive) |
| `UnansweredQuestion` | `question_id` (a UUID is generated if missing) + `question` |
| `AnsweredQuestion` | adds the reference `sources` and `answer` |
| `RagDataset` | a list of questions, the input format |
| `MinimalSearchResults` | one question + its retrieved sources |
| `StudentSearchResults` | all results + `k`, the output of `search_dataset` |
| `MinimalAnswer` / `StudentSearchResultsAndAnswer` | the same with a generated `answer`, the output of `answer_dataset` |

**Key point:** these shapes are imposed by the subject, because the moulinette
reads them.

---

### `chunking.py`: cutting files into pieces

**Why chunk at all?** BM25 works on "documents". A whole file is too big (the
answer gets diluted), a single line is too small (no context). The grader also
rejects any source longer than 2000 characters, and **one over-long source
invalidates the whole output file**.

The data structure is a `Chunk`, which has `file_path`, `first_character_index`,
`last_character_index`, `context` and `body`.

#### Strategy 1: Python files (`chunk_python`)

1. Parse the file with `ast.parse`, Python's own parser.
2. `_chunk_body` walks the top-level statements (imports, functions, classes).
3. It **packs** consecutive statements together while they fit under 2000
   characters.
4. If one class or function alone is too big, it **recurses** into its body,
   so a big class becomes one chunk for the header plus chunks for its methods.
5. If a single statement is still too big, `_split_plain` cuts it on blank
   lines, then single lines, then a hard cut.
6. The context is the path plus the dotted scope, for example
   `.../llm_engine.py LLMEngine.add_request`.

If the file does not parse (a `SyntaxError`), it falls back to plain-text
chunking, so no file disappears from the index.

#### Strategy 2: Markdown and text (`chunk_text`)

1. Find every heading (`#` to `######`) with a regex.
2. Keep a **heading trail**, a stack such as `Serving > OpenAI server > Auth`,
   and use it as the context.
3. Merge small sections with the next one while they fit, and split big ones
   with `_split_plain`.

#### The two guarantees (tested)

- **Verbatim slice:** `text[first:last] == chunk.body`, always. `_emit` trims
  the whitespace *and moves the offsets with it*.
- **Size cap:** `_emit` clamps `end` to `start + max_size`. This is the one
  place every chunk passes through, so the cap cannot be bypassed.

#### The bug worth telling

`ast` gives column positions as **UTF-8 byte offsets**, not character offsets.
On a line containing `é` the two differ, and every later span would be shifted.
`_char_offset` converts them: it encodes the line, cuts at the byte, and
decodes back to count characters.

---

### `tokenization.py`: turning text into words

The same function is used for the **chunks** (at indexing time) and for the
**question** (at search time). That symmetry matters, because both sides must
spell a word the same way or they never match.

`iter_tokens(text)` does the following:

1. Find words with `WORD_RE`, which matches identifiers like `foo_bar2` or
   numbers.
2. **Normalize:** lowercase, plus a light plural rule (`policies` → `policy`,
   `endpoints` → `endpoint`, but `class` stays `class`).
3. Drop **stop words** (`the`, `is`, `how`, ...).
4. Emit the **whole** identifier, then its **sub-words**:
   - `fused_batched_moe` → `fused_batched_moe`, `fused`, `batched`, `moe`
   - `AsyncLLMEngine` → `asyncllmengine`, `async`, `llm`, `engine`
     (`CAMEL_RE` handles the `LLM` acronym)

**Why not a real stemmer (Porter, etc.)?** On code, it over-stems:
`embedding` and `embed`, or `scheduler` and `schedule`, can collide wrongly.
A shallow plural rule gets most of the benefit without that damage.

---

### `indexing.py`: building and saving the index

#### What an inverted index is

A normal mapping is *chunk → words*. An inverted index is *word → chunks*:

```
"lora"     → [chunk 812 (tf 7), chunk 813 (tf 3), chunk 4410 (tf 1)]
"endpoint" → [chunk 12 (tf 2), chunk 812 (tf 1), ...]
```

Searching becomes: look up each question word and only touch the chunks that
contain it.

#### How it is stored: CSR (Compressed Sparse Row)

Instead of a dictionary of lists, which is slow to save and load, everything
sits in three flat NumPy arrays:

```
indptr     = [0, 3, 5, ...]          one entry per word, plus one
doc_ids    = [812, 813, 4410, 12, 812, ...]
term_freqs = [7.0, 3.0, 1.0, 2.0, 1.0, ...]
```

The postings of word number `t` are `doc_ids[indptr[t]:indptr[t+1]]`. Looking
up a word is one array slice, and loading is a few file reads. The index is
even **memory-mapped** (`mmap_mode="r"`), so a single query only reads what
it touches.

Files written in `data/processed/`:

| File | Content |
|---|---|
| `index.npz` | the CSR arrays + per-chunk length, file id, start and end |
| `vocabulary.json` | word → word id |
| `metadata.json` | the list of file paths, max chunk size and chunk count |
| `documents.npz` / `documents.json` | the per-file cache (see below) |

#### Main pieces

- `iter_corpus_files`: walks `data/raw/` in **sorted** order, so chunk ids are
  the same on every rebuild. It keeps `.py`, `.pyi`, `.md`, `.txt`, `.rst` and
  `.markdown` files.
- `_tokenize_file`: chunks one file and counts the words, with body = +1 and
  context = +3 (`CONTEXT_WEIGHT`).
- `FileEntry`: the tokenized result of one file, plus its `mtime` and size.
- `DocumentCache`: the list of all `FileEntry`, organised per file.
- `DocumentCache.to_index()`: converts per-file data into the per-word CSR
  index with vectorised NumPy (`argsort`, `bincount` and `cumsum`, no Python
  loop over postings).

#### Incremental re-indexing

`build_document_cache` receives the previous cache. If a file's path,
modification time and size are unchanged (and `max_chunk_size` is the same),
it **reuses** the cached entry instead of re-chunking it. `--rebuild` forces a
full rebuild. A test checks that the incremental result equals a full rebuild.

---

### `retrieval.py`: ranking with BM25

#### BM25 in plain words

For each question word that appears in a chunk, the chunk gets points.
The points are higher when:

- the word is **rare** in the whole corpus (IDF: `lora` beats `config`),
- the word appears **often** in this chunk (TF), but with **diminishing
  returns**, so 10 occurrences are not worth 10× one occurrence (that is `k1`),
- the chunk is **short** compared with the average (length normalisation, `b`).

The formula, per word:

```
            tf · (k1 + 1)
idf  ·  ───────────────────────────────────
        tf + k1 · (1 − b + b · len/avg_len)

idf = log(1 + (N − df + 0.5) / (df + 0.5))
k1 = 1.5,   b = 0.75
```

`N` is the number of chunks and `df` the number of chunks containing the word.

#### The `Retriever` class

- `__init__`: precomputes the IDF of every word (`df` is simply
  `np.diff(indptr)`) and the normalised chunk lengths, once.
- `score(query)`: for each question word, one slice of the postings and one
  vectorised update of the `scores` array. It returns `None` if no word is
  known.
- `search(query, k)`: the public entry point. It handles `k <= 0` and has an
  **LRU cache** (512 entries) so a repeated question is answered instantly,
  which is useful for the API.
- `_ranked_candidates`: avoids sorting all ~21,000 chunks. `argpartition`
  takes the best `4k`, then the best `32k` if the per-file cap threw too many
  away, and only falls back to a full sort as a last resort.
- `_select`: applies the **max 3 chunks per file** rule.
- `source_of`: chunk id → `MinimalSource`.

#### `HybridRetriever` (bonus, `--hybrid`)

1. BM25 picks the best 100 chunks (the shortlist).
2. Each of them gets a semantic score: the cosine similarity between the
   question embedding and the chunk embedding.
3. Final score = `0.9 × BM25 (rescaled to 0–1) + 0.1 × cosine`.

BM25 decides **what** is considered and embeddings only **reorder** it.
Letting embeddings pick candidates on their own lowered recall, because
questions quote identifiers that a semantic model blurs.

`load_retriever()` loads the index (and embeddings if `--hybrid`) and returns
the right retriever.

---

### `embedding.py`: dense vectors (bonus)

- Model: `sentence-transformers/all-MiniLM-L6-v2`.
- `iter_chunk_texts` rebuilds each chunk's text by slicing the files with the
  offsets stored in the index (no duplicate copy of the corpus), prefixed with
  the file path.
- `build_embedding_index` encodes chunks in batches of 64 into unit-length
  vectors, so a dot product equals the cosine similarity. They are saved as
  `embeddings.npy`.
- Only built with `index --embeddings`, because it takes minutes.

---

### `pipeline.py`: batch work and JSON input/output

- `_load_json`: turns "file missing", "bad JSON" and "cannot read" into one
  clear `PipelineError`.
- `load_rag_dataset` / `load_student_search_results`: JSON → validated
  pydantic model.
- `search_dataset`: loops over every question with a progress bar and builds
  `StudentSearchResults`.
- `write_model`: model → pretty JSON file, creating the folder if needed.
- `read_source_text`: `MinimalSource` → the actual text (`text[start:end]`).
  This is the "augment" step's building block.

---

### `generation.py`: writing the answer

- `build_context(sources)`: reads up to 5 sources and formats them as:
  ```
  [1] data/raw/.../lora.md
  <excerpt text>

  [2] ...
  ```
- `AnswerGenerator`:
  - **Lazy loading:** `torch` and `transformers` are only imported inside
    `_load()`, on the first answer. So `search` starts in milliseconds instead
    of loading a deep-learning stack it does not need.
  - `_truncate`: cuts the excerpts to 3072 tokens so the prompt fits the
    model.
  - `answer`: system prompt (*"use only the excerpts, say so if they do not
    contain the answer, never invent names"*) + excerpts + question, rendered
    with Qwen's chat template using `enable_thinking=False`. Greedy decoding
    (`do_sample=False`) makes the output reproducible, capped at 256 new
    tokens.
  - `_clean`: removes any leftover `<think>...</think>` reasoning block.
- If no source was retrieved, it answers *"No relevant source was retrieved"*
  without even loading the model.

---

### `evaluation.py`: local recall@k

This is a development tool for iterating without the official grader.

- `intersection_over_union(a, b)`: the overlap of two character ranges divided
  by their combined extent. It is 0 if the ranges are in different files.
- A reference source counts as **found** if some retrieved source overlaps
  it with IoU > 0.05.
- `recall_at_k`: for each question, (sources found in the top k) / (reference
  sources), averaged over all questions, for k = 1, 3, 5 and 10.

**Important:** the official numbers come from `moulinette`, which the code
never imports or calls.

---

### `api.py`: HTTP server (bonus)

A FastAPI app started with `python -m src serve` (port 8000):

| Endpoint | Does |
|---|---|
| `GET /health` | status + number of chunks and files |
| `POST /search` `{"query": "...", "k": 10}` | top-k sources |
| `POST /answer` | sources + generated answer (HTTP 503 if the model fails) |

The index loads once at start-up and the model on the first `/answer`.

---

### `__main__.py`: the command line

Built with **Python Fire**: every method of `RagCli` becomes a command and
every parameter becomes a `--flag`.

| Command | Calls |
|---|---|
| `index` | `build_document_cache` → `to_index` → `save` (+ embeddings) |
| `search` | `load_retriever` → `search` → print |
| `search_dataset` | `load_rag_dataset` → `search_dataset` → `write_model` |
| `answer` | `search` → `AnswerGenerator.answer` |
| `answer_dataset` | `load_student_search_results` → answer each → `write_model` |
| `evaluate` | `recall_at_k` → `format_report` |
| `serve` | `uvicorn.run(create_app())` |

**Error handling:** every domain error (`IndexError_`, `PipelineError`,
`GenerationError`, `EmbeddingError`) is caught here and printed as one line,
`Error: ...`, followed by exit code 1. Ctrl-C exits with 130. No user mistake
produces a Python traceback.

---

### `tests/`: what is checked

| File | Checks |
|---|---|
| `test_chunking.py` | chunks slice back out of the file exactly; every definition is covered; heading trail in context; unparsable Python falls back; empty file yields nothing; a real non-ASCII file round-trips |
| `test_pipeline.py` | tokenizer splitting and plurals; index save/load; clean errors on missing index, empty corpus or bad dataset; incremental indexing == full rebuild; query cache; relevant file ranked first; spans ≤ max size; IoU and recall maths |
| `test_hybrid.py` | embeddings save/load; mismatched vectors rejected; reranking can promote a semantic match (with a fake model, no download) |

Run with `make test`. `make lint` runs flake8 + mypy, and `make lint-strict`
also passes `mypy --strict`.

---

## Part 4: Design decisions and why

| Decision | Why |
|---|---|
| **BM25, not TF-IDF** | Saturating TF and length normalisation matter when chunks range from 40 to 2000 characters |
| **BM25, not embeddings, as the base** | Questions quote exact identifiers, and lexical matching is precise on those. It is also fast and needs no GPU. Embeddings only rerank. |
| **Store offsets, not text** | The index stays small and a span can never drift from its file |
| **Weight context ×3 instead of repeating words** | Repeating words would inflate chunk length and fight BM25's own normalisation |
| **Max 3 chunks per file** | Stops one file from taking every slot, so a near-miss does not cost the whole list |
| **Chunk size 2000** | The largest the grader accepts, and it measured best |
| **Lazy model loading** | Search commands start instantly |
| **pydantic at every boundary** | Bad input fails early with a readable message |

---

## Part 5: Questions you will probably be asked

**Q: What is RAG, in one sentence?**
Search first, then let the model answer from what was found, instead of from
memory.

**Q: Why not just give the whole repository to the model?**
It is far too large for a 0.6B model's context window, and more irrelevant text
makes answers worse, not better.

**Q: What happens if the question uses words that are not in the code?**
`score()` returns `None` and the result is empty: *"No matching source found."*
Partial matches still work because each word is scored independently.

**Q: Why are docs questions harder than code questions?**
Code questions often quote an identifier, which is very discriminative. Docs
questions paraphrase. The sub-word tokenizer and heading context are there to
close that gap.

**Q: Why is recall@1 only ~63% but recall@3 ~83%?**
The right *file* is usually found immediately. The extra slots find the right
*region* of that file.

**Q: How do you guarantee no source exceeds 2000 characters?**
Every chunk is created through `_emit`, which clamps the end offset, and a test
asserts it on every chunk.

**Q: What does `k1` / `b` do?**
`k1` = how fast repeated words stop adding points. `b` = how much long chunks
are penalised (0 = never, 1 = fully).

**Q: How does the model avoid making things up?**
The system prompt restricts it to the excerpts and tells it to say when the
answer is missing. Decoding is greedy (no randomness). It is not a 100%
guarantee with a 0.6B model, but grounding greatly reduces it.

**Q: What would you improve next?**
A cross-encoder reranker on the top 20, query expansion, or tuning the hybrid
`alpha` per dataset (docs vs code).

**Q: Did you use AI?**
Yes, as a reviewer and for boilerplate, debugging help and the README. The
retrieval logic, the chunking invariants and the tuning were written and
validated by hand against the moulinette. (See the README section *Use of AI*.)

---

## Glossary

| Term | Meaning |
|---|---|
| **Chunk** | A piece of a file (≤ 2000 characters) treated as one searchable document |
| **Token / term** | A normalised word as stored in the index |
| **Inverted index** | Map from each word to the chunks containing it |
| **Posting** | One `(chunk, frequency)` entry in a word's list |
| **TF** | Term frequency: how often a word occurs in a chunk |
| **IDF** | Inverse document frequency: how rare a word is across chunks |
| **BM25** | The ranking formula combining TF, IDF and chunk length |
| **CSR** | Storage format: flat arrays + an `indptr` array of slice boundaries |
| **Recall@k** | Share of the needed sources found in the top k results |
| **IoU** | Intersection over union: the overlap measure between two character ranges |
| **Embedding** | A vector representing meaning; similar texts give nearby vectors |
| **Hybrid retrieval** | BM25 shortlist reranked with embedding similarity |
| **Grounding** | Making the model answer only from provided text |
