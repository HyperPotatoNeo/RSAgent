# CLAUDE.md — RSAgent Development Guide

## What This Repo Is

RSAgent is an end-to-end RL training harness for Recursive Self-Aggregation (RSA). It contains two packages:

1. **`rsa/`** — Core RSA algorithm: multi-step inference with island topology, aggregation prompts, and LLM adapters (vLLM, OpenAI, Anthropic, Gemini). No RL dependencies.
2. **`rsagent/`** — Verifiers `MultiTurnEnv` subclass that wraps any single-step verifiers environment and runs the RSA loop as one rollout. Integrates with [prime-rl](https://github.com/novaskyai/prime-rl) for GRPO training.

## Architecture

### RSA Loop (rsa/)

The RSA algorithm runs T steps of generate-then-aggregate over M×N candidates:
- `engine.py`: `RSAEngine` orchestrates the loop
- `islands.py`: `get_num_islands(step, M, T)` determines island count at each step; `validate_island_params(M, N, K, T)` checks config validity
- `prompts.py`: `build_prompt(question, candidates, task)` builds aggregation prompts
- `evaluation.py`: Math scoring utilities (boxed extraction, symbolic equivalence)
- `adapters/`: Backend-agnostic LLM interface via `BaseAdapter.generate_batch()`

### RSAgent Environment (rsagent/)

- `env.py`: `RSAgent(MultiTurnEnv)` — the main class. Hooks into verifiers' rollout loop:
  - `setup_state()`: initializes RSA tracking (candidates dict, RNG, step counter)
  - `get_prompt_messages()`: returns question (step 0) or aggregation prompt (step t>0)
  - `add_trajectory_step()`: stores candidate text, advances step/candidate index
  - `render_completion()`: calls `_assign_rewards()` to score final population
  - `_assign_rewards()`: scores candidates, computes metrics, sets rollout-level reward

- `__init__.py`: `load_environment()` — entry point for prime-rl's env discovery via `importlib.import_module("rsagent")`

### Integration with prime-rl

RSAgent works with **unmodified prime-rl**. Key config requirements:
- `trajectory_strategy = "branching"` — each trajectory step becomes an independent `TrainingSample`
- `rollouts_per_example >= 2` — needed for GRPO advantage variance
- `batch_size` = total rollouts (= unique_problems × rollouts_per_example)

The reward pipeline:
1. RSAgent sets `state["reward"]` in `_assign_rewards()`
2. A passthrough rubric (`score_rollouts=True`) preserves this reward through verifiers' scoring pipeline
3. prime-rl's orchestrator reads `rollout["reward"]` and computes GRPO advantages per problem group

### Passthrough Rubric Pattern

verifiers' `dummy_score_group()` unconditionally sets `state["reward"] = 0.0` when `score_rollouts=False`. RSAgent uses `score_rollouts=True` with a rubric function that reads back the pre-computed reward:
```python
def _passthrough_reward(state, **kwargs):
    return state.get("reward", 0.0) or 0.0
```

## Key Invariants

- Trajectory has exactly T × M × N steps (each step generates one candidate)
- `state["rsa_candidates"]` is a dict mapping step → list of M×N candidate texts
- Island boundaries are deterministic given (step, M, T) — see `get_num_islands()`
- The RNG for candidate sampling is seeded per-problem for reproducibility

## Testing

```bash
python -m pytest tests/ -v          # All unit tests (no GPU)
python -m pytest tests/test_rsa.py  # RSA algorithm only
python -m pytest tests/test_rsagent.py  # RSAgent env only
```

Tests use mock inner_env and mock client — no GPU, no API keys, no prime-rl needed.

## Common Tasks

### Adding a new inner environment
Change `inner_env_id` in the TOML config. Works with:
- Registered verifiers envs: `"sokoban-env"` (has `load_environment()` module)
- Any reasoning_gym env name: `"countdown"`, `"basic_arithmetic"`, etc. (falls back to `vf.ReasoningGymEnv`)

### Adding a new prompt format
Edit `rsa/prompts.py`. The `build_prompt(question, candidates, task)` function dispatches on `task` string. Add a new branch for your format.

### Changing reward computation
Edit `rsagent/env.py` `_assign_rewards()`. The `final_reward_metric` parameter controls population-level scoring. Add new metrics there.

### Adding a new LLM adapter
Create `rsa/adapters/your_adapter.py` inheriting from `BaseAdapter`. Implement `generate_batch(prompts, **kwargs) -> List[GenerationResult]`. Register in `rsa/adapters/__init__.py`.

## File Quick Reference

| File | Purpose |
|------|---------|
| `rsa/engine.py` | Core RSA loop |
| `rsa/prompts.py` | `build_prompt()` — aggregation prompt builder |
| `rsa/islands.py` | `get_num_islands()`, `validate_island_params()` |
| `rsa/evaluation.py` | Math scoring (boxed extraction, `is_equiv`) |
| `rsa/adapters/base.py` | `BaseAdapter` ABC + `GenerationResult` |
| `rsagent/__init__.py` | `load_environment()` for prime-rl |
| `rsagent/env.py` | `RSAgent(MultiTurnEnv)` — the RL environment |
| `experiments/rsagent/rl.toml` | Training config for prime-rl |
