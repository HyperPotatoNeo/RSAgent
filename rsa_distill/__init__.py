"""RSA-Distill: Self-Aggregation Context Distillation Environment."""

from rsa_distill.env import RSADistillEnv

__all__ = ["RSADistillEnv"]


def load_environment(
    inner_env_id="countdown",
    inner_env_args=None,
    K=4,
    task="rg",
    per_step_grpo=True,
    enable_distill=False,
    greedy_reward_weight=0.0,
    **kwargs,
):
    """Load RSADistillEnv for prime-rl / verifiers.

    Called by ``vf.load_environment("rsa_distill", **args)`` when prime-rl
    discovers the env via ``importlib.import_module("rsa_distill")``.

    Args:
        inner_env_id: Verifiers env id for the base single-turn env.
        inner_env_args: Dict of kwargs passed to the inner env.
        K: Number of student responses per problem.
        task: Prompt format ("rg", "math", "general").
        per_step_grpo: If True, compute per-sample GRPO advantages.
        enable_distill: If True, set prompt_text overrides for bidirectional KL.
        greedy_reward_weight: Weight for per-candidate greedy reward.
    """
    import verifiers as vf

    inner_env_args = inner_env_args or {}
    try:
        inner_env = vf.load_environment(inner_env_id, **inner_env_args)
    except (ValueError, RuntimeError):
        gym_entry = {"name": inner_env_id, "weight": 1.0, "config": inner_env_args.pop("config", {})}
        inner_env = vf.ReasoningGymEnv(gym=[gym_entry], **inner_env_args)

    return RSADistillEnv(
        inner_env=inner_env,
        K=K,
        task=task,
        per_step_grpo=per_step_grpo,
        enable_distill=enable_distill,
        greedy_reward_weight=greedy_reward_weight,
        **kwargs,
    )
