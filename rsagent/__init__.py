"""RSAgent: Multi-step RSA environment for RL training with verifiers."""

from rsagent.env import RSAgent

__all__ = ["RSAgent"]


def load_environment(
    inner_env_id="sokoban-env",
    inner_env_args=None,
    M=1,
    N=4,
    K=2,
    T=3,
    greedy_reward_weight=0.0,
    task="rg",
    seed=1234,
    final_reward_metric="mean_accuracy",
    per_step_grpo=False,
    **kwargs,
):
    """Load RSAgent environment for prime-rl / verifiers.

    Called by ``vf.load_environment("rsagent", **args)`` when prime-rl
    discovers the env via ``importlib.import_module("rsagent")``.

    Args:
        inner_env_id: Verifiers env id for the base single-turn env.
        inner_env_args: Dict of kwargs passed to the inner env.
        M, N, K, T: RSA parameters.
        greedy_reward_weight: Weight for per-candidate greedy reward.
        task: Prompt format ("rg", "math", etc.).
        seed: Base random seed for RSA candidate sampling.
        final_reward_metric: "mean_accuracy", "majority_vote", or "pass_at_n".
        per_step_grpo: If True, compute per-sample GRPO advantages at each RSA step.
    """
    import verifiers as vf

    inner_env_args = inner_env_args or {}
    try:
        # First try as a registered verifiers env (e.g. "sokoban-env")
        inner_env = vf.load_environment(inner_env_id, **inner_env_args)
    except (ValueError, RuntimeError):
        # Fall back to reasoning_gym env name (e.g. "basic_arithmetic", "countdown")
        gym_entry = {"name": inner_env_id, "weight": 1.0, "config": inner_env_args.pop("config", {})}
        inner_env = vf.ReasoningGymEnv(gym=[gym_entry], **inner_env_args)
    return RSAgent(
        inner_env=inner_env,
        M=M,
        N=N,
        K=K,
        T=T,
        greedy_reward_weight=greedy_reward_weight,
        task=task,
        seed=seed,
        final_reward_metric=final_reward_metric,
        per_step_grpo=per_step_grpo,
        **kwargs,
    )
