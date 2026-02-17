"""Inspect RSA trajectory structure - show prompts across steps.

Usage:
    python scripts/inspect_rsa_steps.py <prime-rl-src-dir> <rollouts-bin-path> [model-name]

Example:
    python scripts/inspect_rsa_steps.py /path/to/prime-rl/src outputs/rsagent/run_default/rollouts/step_0/rollouts.bin
"""
import sys

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

prime_rl_src = sys.argv[1]
rollout_path = sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-4B-Instruct-2507"

sys.path.insert(0, prime_rl_src)
import msgspec
from prime_rl.transport.types import TrainingBatch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(rollout_path, "rb") as f:
    batch = msgspec.msgpack.decode(f.read(), type=TrainingBatch)

# With T=3, M=1, N=4, each rollout has 12 trajectory steps
# Examples 0-3: step 0 candidates (initial question)
# Examples 4-7: step 1 candidates (aggregation of step 0)
# Examples 8-11: step 2 candidates (aggregation of step 1)
# This repeats for each rollout

# Show one rollout's 3 RSA steps (examples 0, 4, 8 from first rollout of first problem)
print("=== First rollout of first problem ===")
print(f"Total examples: {len(batch.examples)}")
print()

for rsa_step in range(3):
    idx = rsa_step * 4  # First candidate of each RSA step
    if idx >= len(batch.examples):
        break
    ex = batch.examples[idx]
    prompt_text = tokenizer.decode(ex.prompt_ids, skip_special_tokens=False)
    completion_text = tokenizer.decode(ex.completion_ids, skip_special_tokens=False)

    print(f"--- RSA Step {rsa_step}, Candidate 0 (example idx={idx}) ---")
    print(f"  Reward: {ex.reward}, Advantage: {ex.advantage}")
    print(f"  Prompt len: {len(ex.prompt_ids)}, Completion len: {len(ex.completion_ids)}")
    # Show just the user message part of prompt
    if "<|im_start|>user" in prompt_text:
        user_part = prompt_text.split("<|im_start|>user")[-1]
        if len(user_part) > 500:
            print(f"  User prompt (first 500 chars): {user_part[:500]}")
        else:
            print(f"  User prompt: {user_part}")
    else:
        print(f"  Prompt (last 300 chars): ...{prompt_text[-300:]}")
    print(f"  Completion (first 200 chars): {completion_text[:200]}")
    print()
