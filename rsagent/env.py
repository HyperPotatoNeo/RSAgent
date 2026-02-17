"""RSAgent: Multi-step RSA environment for RL training with verifiers.

Wraps any single-step verifiers env (e.g. ReasoningGymEnv) and runs the full
RSA loop (T steps x M*N candidates) as one rollout. Each individual generation
across all steps becomes a TrajectoryStep, and with prime-rl's "branch"
trajectory strategy, each becomes an independent training sample for GRPO.

Usage:
    import verifiers as vf
    inner_env = vf.ReasoningGymEnv(gym="countdown", ...)
    env = RSAgent(inner_env, M=1, N=4, K=2, T=4)

For prime-rl training, use trajectory_strategy="branch" and
rollouts_per_example >= 2 for meaningful GRPO signal.
"""

import random
from collections import Counter
from typing import Any, Dict, List, Optional

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State, TrajectoryStep

from rsa.islands import get_num_islands, validate_island_params
from rsa.prompts import build_prompt


def _passthrough_reward(state: State, **kwargs) -> float:
    """Reward function that returns the pre-computed reward from render_completion.

    RSAgent computes rewards in render_completion() → _assign_rewards().
    This function simply reads back that value so the rubric doesn't overwrite it.
    """
    return state.get("reward", 0.0) or 0.0


class RSAgent(MultiTurnEnv):
    """Multi-step RSA environment for end-to-end RL training.

    For a single problem, one rollout does:
      1. Step 0: Generate M*N initial candidates from the question
      2. Steps 1..T-1: For each candidate, sample K from its island,
         build aggregation prompt, generate new candidate
      3. Score final population, assign rewards to all T*M*N trajectory steps

    The trajectory has T*M*N TrajectorySteps. With prime-rl's "branch"
    trajectory strategy, each becomes an independent training sample.

    Args:
        inner_env: A single-step verifiers env (e.g. ReasoningGymEnv).
        M: Number of islands (must be power of 2).
        N: Population per island.
        K: Candidates sampled per aggregation step.
        T: Total RSA steps.
        greedy_reward_weight: Weight for per-candidate greedy reward (default 0).
        task: Prompt format type ("rg", "math", etc.).
        seed: Base random seed for RSA candidate sampling.
        final_reward_metric: How to compute the final reward from the population.
            "mean_accuracy" (default), "majority_vote", or "pass_at_n".
        per_step_grpo: If True, compute per-sample rewards and advantages at each
            RSA step. Each candidate's advantage = greedy_score - mean(step scores).
            Requires branch_rollout in prime-rl to pass through per-step values.
    """

    def __init__(
        self,
        inner_env: vf.SingleTurnEnv,
        M: int = 1,
        N: int = 4,
        K: int = 2,
        T: int = 4,
        greedy_reward_weight: float = 0.0,
        task: str = "rg",
        seed: int = 1234,
        final_reward_metric: str = "mean_accuracy",
        per_step_grpo: bool = False,
        **kwargs,
    ):
        validate_island_params(M, N, K, T)
        self.inner_env = inner_env
        self.M = M
        self.N = N
        self.K = K
        self.T = T
        self.total_candidates = M * N
        self.greedy_reward_weight = greedy_reward_weight
        self.task = task
        self.rsa_seed = seed
        self.final_reward_metric = final_reward_metric
        self.per_step_grpo = per_step_grpo

        # Use a passthrough rubric so score_group() reads back our pre-computed
        # reward instead of overwriting it with 0.0 (the dummy_score_rollout bug).
        rubric = vf.Rubric(funcs=[_passthrough_reward])

        super().__init__(
            max_turns=T * M * N,
            dataset=inner_env.dataset,
            eval_dataset=inner_env.eval_dataset,
            system_prompt=inner_env.system_prompt,
            parser=inner_env.parser,
            rubric=rubric,
            score_rollouts=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # MultiTurnEnv hooks (called by the @final rollout loop)
    # ------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """Initialize RSA tracking state."""
        question = self._extract_question(state["prompt"])
        state["rsa_question"] = question
        state["rsa_step"] = 0
        state["rsa_candidate_idx"] = 0
        state["rsa_candidates"] = {0: [None] * self.total_candidates}
        state["rsa_rng"] = random.Random(
            self.rsa_seed + hash(str(state.get("example_id", 0)))
        )
        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        """Return the prompt for the current RSA step/candidate.

        Step 0: original question (from inner_env's formatted prompt).
        Step t>0: aggregation prompt with K sampled candidates from the island.
        """
        step = state["rsa_step"]
        idx = state["rsa_candidate_idx"]

        if step == 0:
            # Use inner_env's formatted prompt (system + user messages)
            return state["prompt"]

        # Aggregation step: sample K candidates from this candidate's island
        prev_candidates = state["rsa_candidates"][step - 1]
        num_islands = get_num_islands(step, self.M, self.T)
        island_size = self.total_candidates // num_islands
        island_idx = idx // island_size
        island_start = island_idx * island_size
        island_end = island_start + island_size
        pool = prev_candidates[island_start:island_end]

        rng = state["rsa_rng"]
        sampled = rng.sample(pool, self.K)
        question = state["rsa_question"]
        agg_text = build_prompt(question, sampled, self.task)

        return self._build_chat_prompt(agg_text)

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ):
        """Store trajectory step and advance RSA state."""
        state["trajectory"].append(trajectory_step)

        # Extract response text and store as current candidate
        text = self._text_from_completion(trajectory_step["completion"])
        step = state["rsa_step"]
        idx = state["rsa_candidate_idx"]
        state["rsa_candidates"][step][idx] = text

        # Advance to next candidate / step
        idx += 1
        if idx >= self.total_candidates:
            idx = 0
            step += 1
            if step < self.T:
                state["rsa_candidates"][step] = [None] * self.total_candidates
        state["rsa_step"] = step
        state["rsa_candidate_idx"] = idx

    async def render_completion(self, state: State):
        """Score final candidates and assign rewards after RSA loop."""
        self._assign_rewards(state)
        if state["trajectory"]:
            state["completion"] = state["trajectory"][-1]["completion"]
        else:
            state["completion"] = []

    async def env_response(self, messages: Messages, state: State) -> Messages:
        """Not used — get_prompt_messages handles all prompt construction."""
        return []

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _assign_rewards(self, state: State):
        """Score final candidates and assign rewards to all trajectory steps."""
        last_step = self.T - 1
        final_candidates = state["rsa_candidates"].get(last_step, [])
        if not final_candidates or any(c is None for c in final_candidates):
            state["reward"] = 0.0
            state["metrics"] = {}
            return

        answer_idx = state["answer"]

        # Score each final candidate
        final_scores = [
            self._score_candidate(c, answer_idx) for c in final_candidates
        ]

        # Compute final reward metric
        if self.final_reward_metric == "majority_vote":
            final_reward = self._majority_vote_reward(
                final_candidates, answer_idx
            )
        elif self.final_reward_metric == "mean_accuracy":
            n_correct = sum(1.0 for s in final_scores if s >= 1.0)
            final_reward = n_correct / len(final_scores)
        elif self.final_reward_metric == "pass_at_n":
            final_reward = 1.0 if any(s >= 1.0 for s in final_scores) else 0.0
        else:
            final_reward = sum(final_scores) / len(final_scores)

        # Score each trajectory step's candidate (for metrics + greedy weight)
        all_greedy_scores: List[float] = []
        for traj_step in state["trajectory"]:
            text = self._text_from_completion(traj_step["completion"])
            greedy = self._score_candidate(text, answer_idx)
            all_greedy_scores.append(greedy)

        # Per-step GRPO: set per-sample reward and advantage on each trajectory step
        # so branch_rollout() can pass them through to TrainingSample, bypassing
        # the orchestrator's rollout-level reward/advantage assignment.
        if self.per_step_grpo:
            for t in range(self.T):
                start = t * self.total_candidates
                end = start + self.total_candidates
                step_scores = all_greedy_scores[start:end]
                step_mean = sum(step_scores) / len(step_scores)
                for i in range(start, end):
                    state["trajectory"][i]["reward"] = all_greedy_scores[i]
                    state["trajectory"][i]["advantage"] = (
                        all_greedy_scores[i] - step_mean
                    )

        # Rollout-level reward (used by prime-rl for advantage computation)
        mean_greedy = sum(all_greedy_scores) / max(1, len(all_greedy_scores))
        state["reward"] = final_reward + self.greedy_reward_weight * mean_greedy

        # Metrics for logging
        n_correct = sum(1.0 for s in final_scores if s >= 1.0)
        state["metrics"] = {
            "final_reward": final_reward,
            "mean_greedy": mean_greedy,
            "final_mean_accuracy": n_correct / len(final_scores),
            "final_majority_vote": self._majority_vote_reward(
                final_candidates, answer_idx
            ),
            "final_pass_at_n": (
                1.0 if any(s >= 1.0 for s in final_scores) else 0.0
            ),
        }

        # Per-RSA-step accuracy (for logging)
        for t in range(self.T):
            start = t * self.total_candidates
            end = start + self.total_candidates
            step_scores = all_greedy_scores[start:end]
            n_correct = sum(1.0 for s in step_scores if s >= 1.0)
            state["metrics"][f"step_{t}_accuracy"] = n_correct / len(step_scores)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_question(self, prompt_messages: Messages) -> str:
        """Extract user message text from chat messages."""
        if isinstance(prompt_messages, str):
            return prompt_messages
        for m in prompt_messages:
            if m.get("role") == "user":
                return m.get("content", "")
        return str(prompt_messages)

    def _build_chat_prompt(self, text: str) -> Messages:
        """Build chat messages for aggregation steps."""
        msgs: List[Dict[str, str]] = []
        if self.inner_env.system_prompt:
            msgs.append(
                {"role": "system", "content": self.inner_env.system_prompt}
            )
        msgs.append({"role": "user", "content": text})
        return msgs

    @staticmethod
    def _text_from_completion(completion_messages: Messages) -> str:
        """Get text from completion Messages."""
        if isinstance(completion_messages, str):
            return completion_messages
        for m in completion_messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content", "")
        if completion_messages:
            last = completion_messages[-1]
            if isinstance(last, dict):
                return str(last.get("content", ""))
        return ""

    def _score_candidate(self, text: str, answer_idx: str) -> float:
        """Score a single candidate using inner_env's scoring."""
        entry = self.inner_env.rg_dataset[int(answer_idx)]
        parsed = self.inner_env.parser.parse_answer(text)
        parsed_str = str(parsed).strip() if parsed is not None else None
        return self.inner_env.rg_dataset.score_answer(
            answer=parsed_str, entry=entry
        )

    def _majority_vote_reward(
        self, candidates: List[str], answer_idx: str
    ) -> float:
        """Compute majority vote reward from candidates."""
        entry = self.inner_env.rg_dataset[int(answer_idx)]
        parsed_answers: List[str] = []
        for c in candidates:
            parsed = self.inner_env.parser.parse_answer(c)
            parsed_str = str(parsed).strip() if parsed is not None else None
            if parsed_str is not None:
                parsed_answers.append(parsed_str)
        if not parsed_answers:
            return 0.0
        most_common = Counter(parsed_answers).most_common(1)[0][0]
        score = self.inner_env.rg_dataset.score_answer(
            answer=most_common, entry=entry
        )
        return 1.0 if score >= 1.0 else 0.0
