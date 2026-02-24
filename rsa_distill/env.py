"""RSADistillEnv: Self-Aggregation Context Distillation Environment.

Uses single-step self-aggregation as an implicit teacher for context distillation.
For each problem:
  1. Generate K student responses to the raw question
  2. Generate K teacher responses, each from a leave-one-out aggregation prompt
     (teacher j sees K-1 student responses, excluding student j's response)
  3. Train both groups with per-step GRPO; optional bidirectional KL via prompt_text

The model learns to both answer questions well (student) and aggregate
reasoning chains (teacher). Shared weights mean improvements in either
mode benefit the other.

Usage:
    import verifiers as vf
    inner_env = vf.ReasoningGymEnv(gym="countdown", ...)
    env = RSADistillEnv(inner_env, K=4)

For prime-rl training, use trajectory_strategy="branching" and
rollouts_per_example >= 2 for meaningful GRPO signal.
"""

from collections import Counter
from typing import Dict, List

import verifiers as vf
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.types import Messages, State, TrajectoryStep

from rsa.prompts import build_prompt


def _passthrough_reward(state: State, **kwargs) -> float:
    """Reward function that returns the pre-computed reward from render_completion.

    RSADistillEnv computes rewards in render_completion() → _assign_rewards().
    This function simply reads back that value so the rubric doesn't overwrite it.
    """
    return state.get("reward", 0.0) or 0.0


class RSADistillEnv(MultiTurnEnv):
    """Self-Aggregation Context Distillation Environment.

    For each problem, one rollout produces 2K trajectory steps:
      - Steps 0..K-1 (Student): K responses to the raw question
      - Steps K..2K-1 (Teacher): K responses, each from a leave-one-out aggregation
        prompt (teacher j sees K-1 student responses, excluding student j)

    With per_step_grpo=True (default):
      - Student advantage = score - mean(K student scores) [GRPO within student group]
      - Teacher advantage = score - mean(K teacher scores) [GRPO within teacher group]

    With enable_distill=True (Phase 2):
      - Student prompt_text → leave-one-out aggregation prompt (K-1 other responses)
      - Teacher prompt_text → raw question (for teacher logprob evaluation)
      - Bidirectional KL regularization via prime-rl's context distillation

    Args:
        inner_env: A single-step verifiers env (e.g. ReasoningGymEnv).
        K: Number of student responses per problem (>= 1).
        task: Prompt format type for build_prompt ("rg", "math", "general").
        per_step_grpo: If True, set per-sample reward and advantage on trajectory steps.
        enable_distill: If True, override prompt_text for bidirectional distillation.
        greedy_reward_weight: Weight for per-candidate greedy reward in rollout reward.
    """

    def __init__(
        self,
        inner_env: vf.SingleTurnEnv,
        K: int = 4,
        task: str = "rg",
        per_step_grpo: bool = True,
        enable_distill: bool = False,
        greedy_reward_weight: float = 0.0,
        **kwargs,
    ):
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.inner_env = inner_env
        self.K = K
        self.task = task
        self.per_step_grpo = per_step_grpo
        self.enable_distill = enable_distill
        self.greedy_reward_weight = greedy_reward_weight

        rubric = vf.Rubric(funcs=[_passthrough_reward])

        super().__init__(
            max_turns=2 * K,  # K student + K teacher
            dataset=inner_env.dataset,
            eval_dataset=inner_env.eval_dataset,
            system_prompt=inner_env.system_prompt,
            parser=inner_env.parser,
            rubric=rubric,
            score_rollouts=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # MultiTurnEnv hooks
    # ------------------------------------------------------------------

    async def setup_state(self, state: State) -> State:
        """Initialize tracking state for student/teacher generation."""
        question = self._extract_question(state["prompt"])
        state["rsa_distill_question"] = question
        state["rsa_distill_raw_prompt"] = state["prompt"]  # Save original for prompt_text override
        state["rsa_distill_step"] = 0
        state["rsa_distill_student_responses"] = []
        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        """Return prompt for current step.

        Steps 0..K-1: raw question (student generation).
        Steps K..2K-1: leave-one-out aggregation prompt (teacher j excludes student j).
        """
        step = state["rsa_distill_step"]

        if step < self.K:
            # Student step: use original question prompt
            return state["rsa_distill_raw_prompt"]

        # Teacher step j: build leave-one-out aggregation prompt excluding student j
        j = step - self.K
        question = state["rsa_distill_question"]
        student_responses = state["rsa_distill_student_responses"]
        other_responses = [r for idx, r in enumerate(student_responses) if idx != j]
        if other_responses:
            agg_text = build_prompt(question, other_responses, self.task)
            return self._build_chat_prompt(agg_text)
        else:
            # K=1: no other responses, fall back to raw question
            return state["rsa_distill_raw_prompt"]

    async def add_trajectory_step(
        self, state: State, trajectory_step: TrajectoryStep
    ):
        """Store trajectory step and advance state."""
        state["trajectory"].append(trajectory_step)

        # Extract response text
        text = self._text_from_completion(trajectory_step["completion"])
        step = state["rsa_distill_step"]

        if step < self.K:
            # Store student response for later aggregation
            state["rsa_distill_student_responses"].append(text)

        state["rsa_distill_step"] = step + 1

    async def render_completion(self, state: State):
        """Score all completions, assign rewards, and set prompt_text overrides."""
        self._assign_rewards(state)

        # Set prompt_text overrides for context distillation (Phase 2)
        if self.enable_distill:
            self._set_prompt_text_overrides(state)

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
        """Score all 2K completions and assign per-step rewards/advantages."""
        trajectory = state["trajectory"]
        if not trajectory:
            state["reward"] = 0.0
            state["metrics"] = {}
            return

        answer_idx = state["answer"]

        # Score every completion
        all_scores: List[float] = []
        for step in trajectory:
            text = self._text_from_completion(step["completion"])
            score = self._score_candidate(text, answer_idx)
            all_scores.append(score)

        student_scores = all_scores[:self.K]
        teacher_scores = all_scores[self.K:2 * self.K]

        student_mean = sum(student_scores) / max(1, len(student_scores))
        teacher_mean = sum(teacher_scores) / max(1, len(teacher_scores))

        # Per-step GRPO: each group forms its own GRPO baseline
        if self.per_step_grpo:
            # Student group: advantage relative to student mean
            for i in range(min(self.K, len(trajectory))):
                trajectory[i]["reward"] = student_scores[i]
                trajectory[i]["advantage"] = student_scores[i] - student_mean
            # Teacher group: advantage relative to teacher mean
            for j in range(len(teacher_scores)):
                idx = self.K + j
                trajectory[idx]["reward"] = teacher_scores[j]
                trajectory[idx]["advantage"] = teacher_scores[j] - teacher_mean

        # Rollout-level reward
        n_student_correct = sum(1.0 for s in student_scores if s >= 1.0)
        n_teacher_correct = sum(1.0 for s in teacher_scores if s >= 1.0)
        student_acc = n_student_correct / max(1, len(student_scores))
        teacher_acc = n_teacher_correct / max(1, len(teacher_scores))
        all_mean = sum(all_scores) / max(1, len(all_scores))

        # Teacher mean + optional greedy weight on student quality
        state["reward"] = teacher_mean + self.greedy_reward_weight * student_mean

        # Metrics
        state["metrics"] = {
            "student_accuracy": student_acc,
            "teacher_accuracy": teacher_acc,
            "student_mean_score": student_mean,
            "teacher_mean_score": teacher_mean,
            "teacher_uplift": teacher_mean - student_mean,
            "mean_score": all_mean,
            "n_student_correct": n_student_correct,
            "n_teacher_correct": n_teacher_correct,
            "majority_vote": self._majority_vote_reward(
                state["rsa_distill_student_responses"], answer_idx
            ),
        }

    # ------------------------------------------------------------------
    # Context distillation: prompt_text override
    # ------------------------------------------------------------------

    def _set_prompt_text_overrides(self, state: State):
        """Override prompt on trajectory steps for bidirectional distillation.

        branch_rollout() reads step["prompt"] via _extract_prompt_text() to set
        TrainingSample.prompt_text. By overriding step["prompt"] here (after
        generation is complete, tokens already computed), we control what
        prompt the teacher model uses for logprob evaluation WITHOUT affecting
        the training prompt (which uses prompt_ids from the original generation).

        Student completions (0..K-1): prompt → leave-one-out aggregation prompt
            → For student i, aggregation prompt contains K-1 OTHER responses (not i)
            → Prevents information leak where teacher sees the exact response it's evaluating
            → KL pushes student toward aggregation-quality outputs

        Teacher completions (K..2K-1): prompt → raw question messages
            → teacher model evaluates under raw question
            → KL keeps teacher output in-distribution
        """
        trajectory = state["trajectory"]
        if not trajectory:
            return

        question = state["rsa_distill_question"]
        student_responses = state["rsa_distill_student_responses"]
        raw_messages = state["rsa_distill_raw_prompt"]

        # Student completions: leave-one-out aggregation prompt
        # For student i, build prompt from all responses EXCEPT response i
        for i in range(min(self.K, len(trajectory))):
            other_responses = [r for j, r in enumerate(student_responses) if j != i]
            if other_responses:
                loo_text = build_prompt(question, other_responses, self.task)
                trajectory[i]["prompt"] = self._build_chat_prompt(loo_text)
            else:
                # K=1: no other responses, fall back to raw question
                trajectory[i]["prompt"] = raw_messages

        # Teacher completions: override prompt to raw question
        for j in range(self.K):
            idx = self.K + j
            if idx < len(trajectory):
                trajectory[idx]["prompt"] = raw_messages

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
        """Build chat messages for aggregation prompt."""
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
