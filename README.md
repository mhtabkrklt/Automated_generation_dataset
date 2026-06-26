# Automated Dataset Generation Algorithm for Episodic Memory Evaluation in LLMs

A pipeline for generating synthetic Russian banking dialogues and knowledge graphs to benchmark episodic memory in LLMs. Each generated sample includes a multi-turn client–bank chat (or a knowledge graph) plus a probing QA task.

Two independent pipelines:

- **dialogue** — multi-turn chat + QA tasks over the conversation
- **graph** — knowledge graph built from the same scenario + navigation tasks

# Prompts and dataset samples
The benchmark dataset and all generation prompts are in Russian, reflecting the target evaluation domain (Russian-language financial dialogues).

Russian prompts: prompts_all.py
Russian dataset samples: data/dataset_200.zip

To support the international research community, we provide English translations of all generation prompts and 10 representative dataset samples:

English prompts: prompts_all_en.py
English dataset samples (10 examples): samples_en.zip

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install openai tiktoken faker networkx numpy json_repair tqdm python-dotenv
# optional for metrics:
pip install rouge-score bert-score
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=your_endpoint
MODEL_NAME=gpt-4o-mini
VALIDATOR_MODEL=gpt-4o
```

---

## Difficulty levels

| Level  | Tokens   | Batches | Noise | Distractors | Decoys |
|--------|----------|---------|-------|-------------|--------|
| easy   | 20–25K   | 3       | 0%    | 0%          | 0%     |
| medium | 50–60K   | 5       | 20%   | 20%         | 15%    |
| hard   | 90–100K  | 8       | 40%   | 40%         | 35%    |

---

## Dialogue pipeline

```bash
# run everything at once
python run_pipeline.py --stage all --difficulty easy --count 10

# or stage by stage
python run_pipeline.py --stage topics   --difficulty easy --count 10
python run_pipeline.py --stage plan     --difficulty easy --topics topics.json
python run_pipeline.py --stage messages
python run_pipeline.py --stage answers  --difficulty easy
python run_pipeline.py --stage truncate --difficulty easy
python run_pipeline.py --stage tasks    --difficulty easy
```

Results go to `data/results/dialogue/{difficulty}/{task_type}/{id}/`.

Each stage is idempotent — skips directories that already have the output file.

### Pipeline stages

```
topics → plan → messages → answers → truncate → tasks
```

**topics** — generates banking scenarios with a task type and client profile.

**plan** — LLM writes a batched dialogue plan with specific amounts, dates, and rates. For medium/hard, the plan includes noise markers (`[SEMANTIC NOISE]`), distractors (`[DISTRACTOR]`), and decoy facts (`[DECOY]`).

**messages** — generates realistic client messages from the plan (parallel, up to 2 workers per batch).

**answers** — bank operator responds to each message. History is compressed via `ConversationSummaryBuffer` when it gets too long.

**truncate** — trims or pads the dialogue to the target token range. Hard chats are never trimmed, only padded.

**tasks** — generates one QA task per dialogue. Five-stage validation: structure → grounding → non-triviality → LLM judge → adversarial no-context check.

---

## Graph pipeline

```bash
# run everything at once
python run_graph_pipeline.py --stage all --difficulty easy --count 10

# or stage by stage
python run_graph_pipeline.py --stage plan        --difficulty easy
python run_graph_pipeline.py --stage graph_gen   --difficulty easy
python run_graph_pipeline.py --stage graph_tasks --difficulty easy
```

Results go to `data/results/graph/{difficulty}/{task_type}/{id}/`.

### Pipeline stages

```
topics → plan → graph_gen → graph_tasks
```

**graph_gen** — builds a knowledge graph from the plan: nodes (`person`, `account`, `event`, `amount`, `rate`, ...) and edges (`owns`, `transferred_to`, `changed_to`, `caused`, ...). For medium/hard adds stale edges, invalid edges, and duplicate nodes.

**graph_tasks** — generates navigation tasks over the graph. Models receive the graph as text, but with hidden structure depending on difficulty: medium hides `occurred_in`, hard also hides `followed_by`.

---

## Task types

| Type                    | What it tests                                              |
|-------------------------|------------------------------------------------------------|
| `information_extraction`| retrieve a specific fact (amount, date, rate)              |
| `knowledge_update`      | track a value change (A → B → C)                          |
| `temporal_reasoning`    | compute when / how much time passed                        |
| `interference`          | distinguish similar facts (diff ≤ 200 rub / 1 day)        |
| `composite`             | multi-step reasoning across a chain of events              |

---

## Output structure

```
data/results/dialogue/{difficulty}/{task_type}/{id}/
    topic.json           — scenario metadata
    plan.txt             — batched dialogue plan
    user_profile.json    — generated client profile
    chat_truncated.json  — final dialogue
    tasks.json           — QA task

data/results/graph/{difficulty}/{task_type}/{id}/
    graph.json           — knowledge graph (nodes + edges)
    tasks_graph.json     — QA task

```

---

## Evaluation results

**Dataset:** 200 tasks — 100 dialogue + 100 graph (easy / medium / hard)  
**Metric:** Final Accuracy — EM=1 → 1.0; otherwise LLM-judge (0.0 / 0.5 / 1.0)

### Dialogue pipeline (100 tasks)

| Model      | Easy (35) | Medium (35) | Hard (30) | Mean  |
|------------|-----------|-------------|-----------|-------|
| GPT-5-mini | 75.4%     | 73.4%       | 51.0%     | 66.6% |
| GLM-5.1    | 81.4%     | 70.2%       | 58.0%     | 69.9% |

### Graph pipeline (100 tasks)

| Model      | Easy (35) | Medium (30) | Hard (35) | Mean  |
|------------|-----------|-------------|-----------|-------|
| GPT-5-mini | 96.0%     | 88.0%       | 51.9%     | 78.6% |
| GLM-5.1    | 92.0%     | 86.0%       | 41.5%     | 73.2% |

### Overall (200 tasks, bootstrap CI 95%, n=10 000)

| Model      | Mean  | 95% CI        |
|------------|-------|---------------|
| GPT-5-mini | 72.5% | 67.5% – 77.3% |
| GLM-5.1    | 71.5% | 66.3% – 76.4% |
