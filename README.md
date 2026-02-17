# RSAgent

End-to-end RL training for **Recursive Self-Aggregation (RSA)** — a multi-step inference algorithm where a language model generates candidate solutions, then iteratively aggregates them to improve quality.

RSAgent wraps any [verifiers](https://github.com/novaskyai/verifiers) environment and runs the full RSA loop as one rollout. The model learns to both generate good initial answers *and* effectively aggregate candidate solutions, trained end-to-end with GRPO.

## How It Works

For a single problem, one RSAgent rollout does:

1. **Step 0**: Generate M×N initial candidates from the question
2. **Steps 1..T-1**: For each candidate, sample K others from its island, build an aggregation prompt, generate a new candidate
3. **Score** the final population and assign rewards to all T×M×N trajectory steps

```
                RSA Loop (T=3 steps, M=1, N=4, K=2)

Step 0:  Question ──┬──> Gen 0 ──> "answer A"
(initial)           ├──> Gen 1 ──> "answer B"
                    ├──> Gen 2 ──> "answer C"
                    └──> Gen 3 ──> "answer D"

Step 1:  Agg(sample K=2) ──┬──> Gen 0' ──> "refined A"
(aggregate)                 ├──> Gen 1' ──> "refined B"
                            ├──> Gen 2' ──> "refined C"
                            └──> Gen 3' ──> "refined D"

Step 2:  Agg(sample K=2) ──┬──> Gen 0'' ──> "final A"
(aggregate)                 ├──> Gen 1'' ──> "final B"
                            ├──> Gen 2'' ──> "final C"
                            └──> Gen 3'' ──> "final D"

Reward = score(final population) ──> assigned to all 12 trajectory steps
```

Each individual generation becomes an independent training sample via [prime-rl](https://github.com/novaskyai/prime-rl)'s **branching trajectory strategy**.

## Repository Structure

```
RSAgent/
├── rsa/                    # Core RSA algorithm (inference, prompts, islands)
│   ├── engine.py           # RSAEngine: orchestrates the multi-step RSA loop
│   ├── prompts.py          # Aggregation prompt construction
│   ├── islands.py          # Island topology and merge scheduling
│   ├── evaluation.py       # Math scoring utilities (boxed extraction, equivalence)
│   ├── verifiers_eval.py   # Integration with verifiers/reasoning_gym for eval
│   └── adapters/           # LLM backend adapters (vLLM, OpenAI, Anthropic, Gemini)
├── rsagent/                # Verifiers MultiTurnEnv for RL training
│   ├── __init__.py         # load_environment() entry point for prime-rl
│   └── env.py              # RSAgent environment
├── experiments/
│   └── rsagent/
│       └── rl.toml         # Training config for prime-rl
├── tests/
│   ├── test_rsa.py         # RSA algorithm unit tests (no GPU)
│   └── test_rsagent.py     # RSAgent env unit tests (no GPU)
├── scripts/
│   ├── inspect_rollout.py  # Debug: decode rollout binary, show rewards/text
│   └── inspect_rsa_steps.py # Debug: show prompts across RSA steps
└── pyproject.toml
```

## Quick Start: RL Training with prime-rl

### Prerequisites

1. Clone and set up [prime-rl](https://github.com/novaskyai/prime-rl)
2. Clone this repo and install into the prime-rl venv:
   ```bash
   uv pip install -e /path/to/RSAgent
   ```

### Launch Training

```bash
# From your prime-rl directory:
uv run rl @ /path/to/RSAgent/experiments/rsagent/rl.toml \
  --inference_gpu_ids 0,1 --trainer_gpu_ids 2,3 \
  --inference.parallel.dp 2
```

The config uses `sokoban-env` by default. Change `inner_env_id` in the TOML to use other environments:

```toml
# Countdown (simpler, good for testing)
args = { inner_env_id = "countdown", inner_env_args = { num_train_examples = 5000, seed = 42 }, M = 1, N = 4, K = 2, T = 3, task = "rg" }

# Any reasoning_gym env
args = { inner_env_id = "basic_arithmetic", inner_env_args = { num_train_examples = 5000, seed = 42 }, M = 1, N = 4, K = 2, T = 3, task = "rg" }
```

### Key Training Config

```toml
[orchestrator]
trajectory_strategy = "branching"  # REQUIRED: each trajectory step = independent sample
rollouts_per_example = 4           # >= 2 for GRPO signal
batch_size = 64                    # Total rollouts (= unique_problems × rollouts_per_example)
```

### GPU Layout

Default: 4 GPUs on one node.
- GPUs 0-1: vLLM inference (DP=2)
- GPUs 2-3: Trainer (FSDP)

## RSA Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `M` | Number of islands (power of 2) | 1 |
| `N` | Population per island | 4 |
| `K` | Candidates sampled per aggregation | 2 |
| `T` | Total RSA steps | 3 |
| `greedy_reward_weight` | Weight for per-candidate greedy reward | 0.0 |
| `task` | Prompt format: `"rg"`, `"math"`, `"supergpqa"` | `"rg"` |
| `final_reward_metric` | `"mean_accuracy"`, `"majority_vote"`, `"pass_at_n"` | `"mean_accuracy"` |

### Island Merging

With M > 1 islands, candidates within each island only see each other during aggregation. Islands merge progressively over T steps following a pairwise schedule based on `log2(M)`.

## Standalone RSA Inference (No RL)

The `rsa/` package can be used independently for RSA inference with any LLM backend:

```python
from rsa import RSAEngine
from rsa.adapters import VLLMAdapter

adapter = VLLMAdapter(model="Qwen/Qwen3-4B-Instruct-2507")
engine = RSAEngine(adapter, M=1, N=4, K=2, T=3, task="math")

results = engine.run(questions=["What is 2+2?"], temperature=0.7)
```

Supported backends: vLLM (local), OpenAI API, Anthropic API, Google Gemini API.

## Reward Structure

- **Rollout reward** = `final_metric + greedy_weight × mean(greedy_scores)`
- With branching strategy, all training samples from one rollout share the same rollout-level reward and advantage
- Requires `rollouts_per_example >= 2` for meaningful GRPO signal (different RSA runs on the same problem can have different outcomes)

## Sequence Length Considerations

Aggregation prompts embed K previous candidate responses. With `max_tokens=4096` and `K=2`:

| Step | Prompt tokens (approx) | Completion tokens | Total |
|------|----------------------|-------------------|-------|
| Step 0 | ~200 (question only) | up to 4096 | ~4300 |
| Step 1+ | ~8500 (K=2 candidates + question) | up to 4096 | ~12600 |

Set `seq_len >= 16384` to accommodate the longest samples.

## Testing

```bash
# Unit tests (no GPU required)
python -m pytest tests/ -v
```

## How It Maps to RL

| Concept | RSAgent |
|---------|---------|
| State s_t | Current candidate pool at RSA step t |
| Action a_t | M×N generated responses |
| Transition | Build aggregation prompts from responses + island merging |
| Reward | Evaluation of final population |

For policy gradient, each individual generation contributes independently:
```
loss_i = -advantage × log pi(response_i | prompt_i)
```
