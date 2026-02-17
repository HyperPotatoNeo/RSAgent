"""Inspect rollout data from a training step.

Usage:
    python scripts/inspect_rollout.py <prime-rl-src-dir> <rollouts-bin-path> [model-name]

Example:
    python scripts/inspect_rollout.py /path/to/prime-rl/src outputs/rsagent/run_default/rollouts/step_0/rollouts.bin
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

print(f"Step: {batch.step}")
print(f"Number of examples: {len(batch.examples)}")
print(f"Rewards: {[ex.reward for ex in batch.examples]}")
print(f"Advantages: {[round(ex.advantage, 4) if ex.advantage else None for ex in batch.examples]}")
print()

# Show first few examples with decoded text
for i, ex in enumerate(batch.examples[:4]):
    prompt_text = tokenizer.decode(ex.prompt_ids, skip_special_tokens=False)
    completion_text = tokenizer.decode(ex.completion_ids, skip_special_tokens=False)
    trainable = sum(ex.completion_mask)
    print(f"=== Example {i} ===")
    print(f"  Reward: {ex.reward}")
    print(f"  Advantage: {ex.advantage}")
    print(f"  Prompt len: {len(ex.prompt_ids)} tokens")
    print(f"  Completion len: {len(ex.completion_ids)} tokens ({trainable} trainable)")
    print(f"  Prompt (last 200 chars): ...{prompt_text[-200:]}")
    print(f"  Completion (first 300 chars): {completion_text[:300]}")
    print()
