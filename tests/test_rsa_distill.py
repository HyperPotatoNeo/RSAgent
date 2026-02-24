"""Unit tests for RSADistillEnv (no GPU required).

Uses mock inner_env to test self-aggregation logic,
prompt construction, reward computation, and prompt_text overrides.

Can run on login nodes without verifiers installed by mocking the module.
"""

import asyncio
import os
import sys
import types
import unittest
from unittest.mock import MagicMock

# Add repo root to path (parent of tests/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Mock verifiers module (so tests run without verifiers installed)
# ---------------------------------------------------------------------------

def _setup_mock_verifiers():
    """Create mock verifiers module if not already importable."""
    try:
        import verifiers
        return  # Already available, no mocking needed
    except ImportError:
        pass

    # Create mock module hierarchy
    vf = types.ModuleType("verifiers")
    vf_envs = types.ModuleType("verifiers.envs")
    vf_envs_mt = types.ModuleType("verifiers.envs.multiturn_env")
    vf_types = types.ModuleType("verifiers.types")

    # MultiTurnEnv base class (minimal mock)
    class MockMultiTurnEnv:
        def __init__(self, max_turns=10, dataset=None, eval_dataset=None,
                     system_prompt=None, parser=None, rubric=None,
                     score_rollouts=True, **kwargs):
            self.max_turns = max_turns
            self.dataset = dataset
            self.eval_dataset = eval_dataset
            self.system_prompt = system_prompt
            self.parser = parser
            self.rubric = rubric
            self.score_rollouts = score_rollouts

    vf_envs_mt.MultiTurnEnv = MockMultiTurnEnv

    # Types
    vf_types.Messages = list
    vf_types.State = dict
    vf_types.TrajectoryStep = dict

    # Top-level verifiers attributes
    vf.SingleTurnEnv = type("SingleTurnEnv", (), {})
    vf.Rubric = lambda funcs: MagicMock(funcs=funcs)
    vf.load_environment = MagicMock()
    vf.ReasoningGymEnv = MagicMock()

    # Register in sys.modules
    vf.envs = vf_envs
    vf_envs.multiturn_env = vf_envs_mt
    vf.types = vf_types
    sys.modules["verifiers"] = vf
    sys.modules["verifiers.envs"] = vf_envs
    sys.modules["verifiers.envs.multiturn_env"] = vf_envs_mt
    sys.modules["verifiers.types"] = vf_types


_setup_mock_verifiers()


# ---------------------------------------------------------------------------
# Mock infrastructure (same pattern as RSAgent tests)
# ---------------------------------------------------------------------------

def make_mock_inner_env(
    num_problems=3,
    correct_answer="42",
    system_prompt="You are a helpful assistant. Answer in <answer> tags.",
):
    """Create a mock inner_env that mimics ReasoningGymEnv.

    Uses a simple list-of-dicts as dataset (no HF datasets dependency).
    """

    # Simple dataset: list of dicts with __getitem__ and __len__
    data = []
    for i in range(num_problems):
        data.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"What is the answer to problem {i}?"},
            ],
            "answer": str(i),
            "example_id": i,
            "task": "test",
        })

    class SimpleDataset:
        def __init__(self, items):
            self._items = items
        def __getitem__(self, idx):
            return self._items[idx]
        def __len__(self):
            return len(self._items)

    ds = SimpleDataset(data)

    env = MagicMock()
    env.dataset = ds
    env.eval_dataset = ds
    env.system_prompt = system_prompt

    parser = MagicMock()
    def parse_answer(text):
        if "<answer>" in text and "</answer>" in text:
            start = text.index("<answer>") + len("<answer>")
            end = text.index("</answer>")
            return text[start:end].strip()
        return None
    parser.parse_answer = parse_answer
    env.parser = parser

    entries = {}
    for i in range(num_problems):
        entries[i] = {
            "question": f"What is the answer to problem {i}?",
            "answer": correct_answer,
        }

    class MockRGDataset:
        def __getitem__(self, idx):
            return entries[int(idx)]
        def score_answer(self, answer=None, entry=None):
            if answer is None:
                return 0.0
            return 1.0 if answer == entry["answer"] else 0.0

    env.rg_dataset = MockRGDataset()
    return env


def run_async(coro):
    """Run an async function synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def create_env(K=4, **kwargs):
    """Create RSADistillEnv with mock inner_env."""
    from rsa_distill.env import RSADistillEnv
    inner_env = make_mock_inner_env(**kwargs.pop("env_kwargs", {}))
    return RSADistillEnv(
        inner_env=inner_env,
        K=K,
        **kwargs,
    ), inner_env


def make_trajectory_step(text, prompt=None):
    """Create a trajectory step dict with completion text."""
    return {
        "completion": [{"role": "assistant", "content": text}],
        "prompt": prompt or [],
        "response": None,
        "tokens": None,
        "reward": None,
        "advantage": None,
        "is_truncated": False,
        "trajectory_id": "test",
        "extras": {},
        "temperature": 1.0,
    }


def make_state(env, problem_idx=0, inner_env=None):
    """Create initial state for testing, simulating setup_state."""
    ds = inner_env.dataset if inner_env else env.inner_env.dataset
    prompt = ds[problem_idx]["prompt"]
    state = {
        "prompt": prompt,
        "answer": ds[problem_idx]["answer"],
        "example_id": problem_idx,
        "trajectory": [],
        "error": None,
        "completion": [],
    }
    state = run_async(env.setup_state(state))
    return state


def run_full_rollout(env, inner_env, student_texts, teacher_texts, problem_idx=0):
    """Run a full 2K rollout: K student + K teacher responses."""
    state = make_state(env, problem_idx=problem_idx, inner_env=inner_env)
    for text in student_texts:
        run_async(env.get_prompt_messages(state))
        run_async(env.add_trajectory_step(state, make_trajectory_step(text)))
    for text in teacher_texts:
        run_async(env.get_prompt_messages(state))
        run_async(env.add_trajectory_step(state, make_trajectory_step(text)))
    run_async(env.render_completion(state))
    return state


# ===========================================================================
# Test: Initialization
# ===========================================================================

class TestInit(unittest.TestCase):

    def test_default_params(self):
        env, _ = create_env(K=4)
        self.assertEqual(env.K, 4)
        self.assertEqual(env.max_turns, 8)  # 2 * K
        self.assertTrue(env.per_step_grpo)
        self.assertFalse(env.enable_distill)

    def test_k_equals_1(self):
        env, _ = create_env(K=1)
        self.assertEqual(env.K, 1)
        self.assertEqual(env.max_turns, 2)  # 2 * 1

    def test_k_equals_2(self):
        env, _ = create_env(K=2)
        self.assertEqual(env.max_turns, 4)  # 2 * 2

    def test_k_equals_8(self):
        env, _ = create_env(K=8)
        self.assertEqual(env.max_turns, 16)  # 2 * 8

    def test_invalid_k_zero(self):
        with self.assertRaises(ValueError):
            create_env(K=0)

    def test_invalid_k_negative(self):
        with self.assertRaises(ValueError):
            create_env(K=-1)

    def test_dataset_passthrough(self):
        env, inner_env = create_env(K=4)
        self.assertEqual(len(env.dataset), len(inner_env.dataset))

    def test_task_param(self):
        env, _ = create_env(K=4, task="math")
        self.assertEqual(env.task, "math")


# ===========================================================================
# Test: State Management
# ===========================================================================

class TestStateManagement(unittest.TestCase):

    def test_setup_state(self):
        env, inner_env = create_env(K=4)
        state = make_state(env, inner_env=inner_env)
        self.assertEqual(state["rsa_distill_step"], 0)
        self.assertEqual(state["rsa_distill_student_responses"], [])
        self.assertIn("rsa_distill_question", state)
        self.assertIn("rsa_distill_raw_prompt", state)

    def test_question_extraction(self):
        env, inner_env = create_env(K=4)
        state = make_state(env, inner_env=inner_env)
        self.assertEqual(state["rsa_distill_question"], "What is the answer to problem 0?")

    def test_step_advances(self):
        env, inner_env = create_env(K=2)
        state = make_state(env, inner_env=inner_env)

        # Step 0: first student
        run_async(env.add_trajectory_step(state, make_trajectory_step("resp_0")))
        self.assertEqual(state["rsa_distill_step"], 1)
        self.assertEqual(len(state["rsa_distill_student_responses"]), 1)

        # Step 1: second student
        run_async(env.add_trajectory_step(state, make_trajectory_step("resp_1")))
        self.assertEqual(state["rsa_distill_step"], 2)
        self.assertEqual(len(state["rsa_distill_student_responses"]), 2)

        # Step 2: first teacher
        run_async(env.add_trajectory_step(state, make_trajectory_step("teacher_0")))
        self.assertEqual(state["rsa_distill_step"], 3)
        # Teacher responses NOT added to student_responses
        self.assertEqual(len(state["rsa_distill_student_responses"]), 2)

        # Step 3: second teacher
        run_async(env.add_trajectory_step(state, make_trajectory_step("teacher_1")))
        self.assertEqual(state["rsa_distill_step"], 4)
        self.assertEqual(len(state["rsa_distill_student_responses"]), 2)

    def test_student_responses_stored(self):
        env, inner_env = create_env(K=3)
        state = make_state(env, inner_env=inner_env)

        responses = ["<answer>42</answer>", "<answer>99</answer>", "<answer>42</answer>"]
        for r in responses:
            run_async(env.add_trajectory_step(state, make_trajectory_step(r)))

        self.assertEqual(state["rsa_distill_student_responses"], responses)

    def test_trajectory_length(self):
        env, inner_env = create_env(K=3)
        state = make_state(env, inner_env=inner_env)

        # K student + K teacher = 2K steps
        for i in range(3):
            run_async(env.add_trajectory_step(state, make_trajectory_step(f"student_{i}")))
        for i in range(3):
            run_async(env.add_trajectory_step(state, make_trajectory_step(f"teacher_{i}")))

        self.assertEqual(len(state["trajectory"]), 6)  # 2K


# ===========================================================================
# Test: Prompt Construction
# ===========================================================================

class TestPromptConstruction(unittest.TestCase):

    def test_student_steps_get_raw_question(self):
        """Steps 0..K-1 should return the raw question prompt."""
        env, inner_env = create_env(K=3)
        state = make_state(env, inner_env=inner_env)

        # All student steps should return original prompt
        for i in range(3):
            prompt = run_async(env.get_prompt_messages(state))
            self.assertEqual(prompt, state["rsa_distill_raw_prompt"])
            run_async(env.add_trajectory_step(state, make_trajectory_step(f"resp_{i}")))

    def test_teacher_steps_get_leave_one_out_prompts(self):
        """Steps K..2K-1 should return leave-one-out aggregation prompts."""
        env, inner_env = create_env(K=3)
        state = make_state(env, inner_env=inner_env)

        # Generate K student responses
        student_texts = ["<answer>A</answer>", "<answer>B</answer>", "<answer>C</answer>"]
        for text in student_texts:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        # Teacher step 0 (step K): excludes student 0 (A), includes B and C
        prompt_0 = run_async(env.get_prompt_messages(state))
        user_msg_0 = next(m for m in prompt_0 if m["role"] == "user")
        self.assertNotIn("<answer>A</answer>", user_msg_0["content"])
        self.assertIn("<answer>B</answer>", user_msg_0["content"])
        self.assertIn("<answer>C</answer>", user_msg_0["content"])
        run_async(env.add_trajectory_step(state, make_trajectory_step("teacher_0")))

        # Teacher step 1 (step K+1): excludes student 1 (B), includes A and C
        prompt_1 = run_async(env.get_prompt_messages(state))
        user_msg_1 = next(m for m in prompt_1 if m["role"] == "user")
        self.assertIn("<answer>A</answer>", user_msg_1["content"])
        self.assertNotIn("<answer>B</answer>", user_msg_1["content"])
        self.assertIn("<answer>C</answer>", user_msg_1["content"])
        run_async(env.add_trajectory_step(state, make_trajectory_step("teacher_1")))

        # Teacher step 2 (step K+2): excludes student 2 (C), includes A and B
        prompt_2 = run_async(env.get_prompt_messages(state))
        user_msg_2 = next(m for m in prompt_2 if m["role"] == "user")
        self.assertIn("<answer>A</answer>", user_msg_2["content"])
        self.assertIn("<answer>B</answer>", user_msg_2["content"])
        self.assertNotIn("<answer>C</answer>", user_msg_2["content"])

    def test_teacher_prompt_contains_question(self):
        """Aggregation prompt should include the original question."""
        env, inner_env = create_env(K=2)
        state = make_state(env, inner_env=inner_env)

        for text in ["resp_0", "resp_1"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        prompt = run_async(env.get_prompt_messages(state))
        user_msg = next(m for m in prompt if m["role"] == "user")
        # Should reference the original question
        self.assertIn("problem 0", user_msg["content"])

    def test_system_prompt_in_teacher(self):
        """System prompt from inner_env should be included in teacher prompt."""
        env, inner_env = create_env(K=2)
        state = make_state(env, inner_env=inner_env)

        for text in ["resp_0", "resp_1"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        prompt = run_async(env.get_prompt_messages(state))
        roles = [m["role"] for m in prompt]
        self.assertIn("system", roles)

    def test_k1_teacher_gets_raw_question(self):
        """With K=1, teacher step has no other responses -> raw question prompt."""
        env, inner_env = create_env(K=1)
        state = make_state(env, inner_env=inner_env)

        run_async(env.get_prompt_messages(state))
        run_async(env.add_trajectory_step(state, make_trajectory_step("response")))

        prompt = run_async(env.get_prompt_messages(state))
        # K=1: leave-one-out has 0 responses -> falls back to raw question
        self.assertEqual(prompt, state["rsa_distill_raw_prompt"])

    def test_teacher_prompts_contain_candidate_keyword(self):
        """Teacher aggregation prompts should contain 'candidate' from build_prompt."""
        env, inner_env = create_env(K=2)
        state = make_state(env, inner_env=inner_env)

        for text in ["<answer>42</answer>", "<answer>99</answer>"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        for _ in range(2):
            prompt = run_async(env.get_prompt_messages(state))
            user_msg = next(m for m in prompt if m["role"] == "user")
            self.assertIn("candidate", user_msg["content"].lower())
            run_async(env.add_trajectory_step(state, make_trajectory_step("teacher")))


# ===========================================================================
# Test: Reward Computation
# ===========================================================================

class TestRewardComputation(unittest.TestCase):

    def test_all_correct(self):
        """All 2K correct -> per-step advantages centered within each group."""
        env, inner_env = create_env(K=3)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>"] * 3,
            teacher_texts=["<answer>42</answer>"] * 3,
        )

        # All rewards should be 1.0
        for step in state["trajectory"]:
            self.assertAlmostEqual(step["reward"], 1.0)
        # All advantages should be 0 (all equal to their group mean)
        for step in state["trajectory"]:
            self.assertAlmostEqual(step["advantage"], 0.0)

    def test_all_wrong(self):
        """All 2K wrong -> all rewards 0, advantages 0."""
        env, inner_env = create_env(K=3)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>99</answer>"] * 3,
            teacher_texts=["<answer>99</answer>"] * 3,
        )

        for step in state["trajectory"]:
            self.assertAlmostEqual(step["reward"], 0.0)
            self.assertAlmostEqual(step["advantage"], 0.0)

    def test_mixed_students(self):
        """Some students correct, others wrong -> advantages reflect difference."""
        env, inner_env = create_env(K=4)
        state = run_full_rollout(
            env, inner_env,
            student_texts=[
                "<answer>42</answer>",  # correct
                "<answer>99</answer>",  # wrong
                "<answer>42</answer>",  # correct
                "<answer>99</answer>",  # wrong
            ],
            teacher_texts=["<answer>42</answer>"] * 4,  # all correct
        )

        # Student scores: [1, 0, 1, 0], mean = 0.5
        # Student advantages: [0.5, -0.5, 0.5, -0.5]
        self.assertAlmostEqual(state["trajectory"][0]["advantage"], 0.5)
        self.assertAlmostEqual(state["trajectory"][1]["advantage"], -0.5)
        self.assertAlmostEqual(state["trajectory"][2]["advantage"], 0.5)
        self.assertAlmostEqual(state["trajectory"][3]["advantage"], -0.5)
        # Teacher scores: [1,1,1,1], teacher_mean = 1.0, all advantages = 0.0
        for i in range(4, 8):
            self.assertAlmostEqual(state["trajectory"][i]["advantage"], 0.0)

    def test_mixed_teachers(self):
        """Some teachers correct, others wrong -> teacher GRPO within teacher group."""
        env, inner_env = create_env(K=2)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>42</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>99</answer>"],
        )

        # Student scores: [1,1], student_mean=1.0, advantages=[0,0]
        self.assertAlmostEqual(state["trajectory"][0]["advantage"], 0.0)
        self.assertAlmostEqual(state["trajectory"][1]["advantage"], 0.0)
        # Teacher scores: [1,0], teacher_mean=0.5, advantages=[0.5,-0.5]
        self.assertAlmostEqual(state["trajectory"][2]["advantage"], 0.5)
        self.assertAlmostEqual(state["trajectory"][3]["advantage"], -0.5)

    def test_teacher_group_grpo_baseline(self):
        """Teacher advantage uses teacher_mean, NOT student_mean."""
        env, inner_env = create_env(K=3)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>99</answer>"] * 3,  # all wrong
            teacher_texts=[
                "<answer>42</answer>",  # correct
                "<answer>42</answer>",  # correct
                "<answer>99</answer>",  # wrong
            ],
        )

        # Student mean = 0.0, teacher scores = [1,1,0], teacher_mean = 2/3
        teacher_mean = 2.0 / 3.0
        self.assertAlmostEqual(state["trajectory"][3]["advantage"], 1.0 - teacher_mean)
        self.assertAlmostEqual(state["trajectory"][4]["advantage"], 1.0 - teacher_mean)
        self.assertAlmostEqual(state["trajectory"][5]["advantage"], 0.0 - teacher_mean)

    def test_student_advantage_is_grpo_within_group(self):
        """Student advantage should be score - mean(student scores), not including teacher."""
        env, inner_env = create_env(K=2)
        state = run_full_rollout(
            env, inner_env,
            student_texts=[
                "<answer>42</answer>",  # correct: 1.0
                "<answer>99</answer>",  # wrong: 0.0
            ],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )

        # Student mean = (1.0 + 0.0) / 2 = 0.5
        # Student advantages: [0.5, -0.5]
        self.assertAlmostEqual(state["trajectory"][0]["advantage"], 0.5)
        self.assertAlmostEqual(state["trajectory"][1]["advantage"], -0.5)

    def test_per_step_grpo_sets_reward(self):
        """With per_step_grpo=True, each trajectory step should have reward set."""
        env, inner_env = create_env(K=2, per_step_grpo=True)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>99</answer>"],
        )

        self.assertAlmostEqual(state["trajectory"][0]["reward"], 1.0)
        self.assertAlmostEqual(state["trajectory"][1]["reward"], 0.0)
        self.assertAlmostEqual(state["trajectory"][2]["reward"], 1.0)
        self.assertAlmostEqual(state["trajectory"][3]["reward"], 0.0)

    def test_no_per_step_grpo(self):
        """With per_step_grpo=False, trajectory steps should NOT have reward/advantage set."""
        env, inner_env = create_env(K=2, per_step_grpo=False)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )

        # Without per_step_grpo, reward/advantage left as None on trajectory steps
        for step in state["trajectory"]:
            self.assertIsNone(step["reward"])
            self.assertIsNone(step["advantage"])

    def test_rollout_reward_teacher_mean(self):
        """Rollout-level reward = teacher_mean when greedy_reward_weight=0."""
        env, inner_env = create_env(K=2, greedy_reward_weight=0.0)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>99</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>99</answer>"],
        )
        # teacher_scores = [1.0, 0.0], teacher_mean = 0.5
        self.assertAlmostEqual(state["reward"], 0.5)

    def test_rollout_reward_with_greedy_weight(self):
        """greedy_reward_weight adds student_mean to teacher_mean."""
        env, inner_env = create_env(K=2, greedy_reward_weight=1.0)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )
        # teacher_mean=1.0, student_mean=0.5
        # reward = 1.0 + 1.0 * 0.5 = 1.5
        self.assertAlmostEqual(state["reward"], 1.5)


# ===========================================================================
# Test: Metrics
# ===========================================================================

class TestMetrics(unittest.TestCase):

    def test_metrics_keys(self):
        env, inner_env = create_env(K=2)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )
        expected_keys = {
            "student_accuracy", "teacher_accuracy",
            "student_mean_score", "teacher_mean_score",
            "teacher_uplift", "mean_score",
            "n_student_correct", "n_teacher_correct",
            "majority_vote",
        }
        self.assertTrue(expected_keys.issubset(set(state["metrics"].keys())))

    def test_metrics_values(self):
        env, inner_env = create_env(K=4)
        state = run_full_rollout(
            env, inner_env,
            student_texts=[
                "<answer>42</answer>", "<answer>42</answer>",
                "<answer>99</answer>", "<answer>99</answer>",
            ],
            teacher_texts=[
                "<answer>42</answer>", "<answer>42</answer>",
                "<answer>42</answer>", "<answer>99</answer>",
            ],
        )
        self.assertAlmostEqual(state["metrics"]["student_accuracy"], 0.5)
        self.assertAlmostEqual(state["metrics"]["teacher_accuracy"], 0.75)
        self.assertAlmostEqual(state["metrics"]["student_mean_score"], 0.5)
        self.assertAlmostEqual(state["metrics"]["teacher_mean_score"], 0.75)
        self.assertAlmostEqual(state["metrics"]["teacher_uplift"], 0.25)
        self.assertAlmostEqual(state["metrics"]["n_student_correct"], 2.0)
        self.assertAlmostEqual(state["metrics"]["n_teacher_correct"], 3.0)


# ===========================================================================
# Test: prompt_text Override (Phase 2 distillation)
# ===========================================================================

class TestPromptTextOverride(unittest.TestCase):

    def test_no_override_when_disabled(self):
        """With enable_distill=False, prompt should NOT be overridden."""
        env, inner_env = create_env(K=2, enable_distill=False)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )

        # Prompts should be unchanged (whatever they were during generation)
        for step in state["trajectory"]:
            # Original prompt is [] (from make_trajectory_step)
            self.assertEqual(step["prompt"], [])

    def test_student_gets_leave_one_out_aggregation_prompt(self):
        """With enable_distill=True, student steps get leave-one-out aggregation prompt."""
        env, inner_env = create_env(K=2, enable_distill=True)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )

        # Student step 0: should see response 1 (99) but NOT response 0 (42)
        prompt_0 = state["trajectory"][0]["prompt"]
        self.assertIsInstance(prompt_0, list)
        user_msg_0 = next(m for m in prompt_0 if m["role"] == "user")
        self.assertIn("<answer>99</answer>", user_msg_0["content"])
        self.assertNotIn("<answer>42</answer>", user_msg_0["content"])
        self.assertIn("candidate", user_msg_0["content"].lower())

        # Student step 1: should see response 0 (42) but NOT response 1 (99)
        prompt_1 = state["trajectory"][1]["prompt"]
        user_msg_1 = next(m for m in prompt_1 if m["role"] == "user")
        self.assertIn("<answer>42</answer>", user_msg_1["content"])
        self.assertNotIn("<answer>99</answer>", user_msg_1["content"])

    def test_all_teacher_steps_get_raw_question(self):
        """With enable_distill=True, ALL K teacher steps get raw question as prompt_text."""
        env, inner_env = create_env(K=3, enable_distill=True)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>A</answer>", "<answer>B</answer>", "<answer>C</answer>"],
            teacher_texts=["<answer>42</answer>"] * 3,
        )

        raw_prompt = state["rsa_distill_raw_prompt"]
        # All 3 teacher steps (indices 3, 4, 5) should have raw question prompt
        for i in range(3, 6):
            self.assertEqual(state["trajectory"][i]["prompt"], raw_prompt)

    def test_override_preserves_trajectory_structure(self):
        """prompt override should not break other trajectory step fields."""
        env, inner_env = create_env(K=2, enable_distill=True)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>", "<answer>99</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )

        for step in state["trajectory"]:
            # Completion should still be intact
            self.assertIn("completion", step)
            self.assertIsNotNone(step["completion"])
            # Reward/advantage should still be set (per_step_grpo=True by default)
            self.assertIsNotNone(step["reward"])
            self.assertIsNotNone(step["advantage"])

    def test_k1_distill_falls_back_to_raw_question(self):
        """With K=1 and enable_distill=True, student gets raw question (no other responses)."""
        env, inner_env = create_env(K=1, enable_distill=True)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>"],
            teacher_texts=["<answer>42</answer>"],
        )

        # K=1: leave-one-out has zero other responses -> falls back to raw question
        student_prompt = state["trajectory"][0]["prompt"]
        raw_prompt = state["rsa_distill_raw_prompt"]
        self.assertEqual(student_prompt, raw_prompt)

    def test_k3_student_leave_one_out_completeness(self):
        """K=3: verify each student step's leave-one-out prompt has exactly K-1 responses."""
        env, inner_env = create_env(K=3, enable_distill=True)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>X</answer>", "<answer>Y</answer>", "<answer>Z</answer>"],
            teacher_texts=["<answer>42</answer>"] * 3,
        )

        # Student 0: should have Y and Z but NOT X
        p0 = next(m for m in state["trajectory"][0]["prompt"] if m["role"] == "user")["content"]
        self.assertNotIn("<answer>X</answer>", p0)
        self.assertIn("<answer>Y</answer>", p0)
        self.assertIn("<answer>Z</answer>", p0)

        # Student 1: should have X and Z but NOT Y
        p1 = next(m for m in state["trajectory"][1]["prompt"] if m["role"] == "user")["content"]
        self.assertIn("<answer>X</answer>", p1)
        self.assertNotIn("<answer>Y</answer>", p1)
        self.assertIn("<answer>Z</answer>", p1)

        # Student 2: should have X and Y but NOT Z
        p2 = next(m for m in state["trajectory"][2]["prompt"] if m["role"] == "user")["content"]
        self.assertIn("<answer>X</answer>", p2)
        self.assertIn("<answer>Y</answer>", p2)
        self.assertNotIn("<answer>Z</answer>", p2)


# ===========================================================================
# Test: Edge Cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):

    def test_k_equals_1(self):
        """K=1: single student + single teacher."""
        env, inner_env = create_env(K=1)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>"],
            teacher_texts=["<answer>42</answer>"],
        )

        self.assertEqual(len(state["trajectory"]), 2)  # 2K = 2
        self.assertAlmostEqual(state["trajectory"][0]["reward"], 1.0)
        self.assertAlmostEqual(state["trajectory"][1]["reward"], 1.0)

    def test_empty_response(self):
        """Empty student response -> score 0 (no answer tags)."""
        env, inner_env = create_env(K=2)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["", "<answer>42</answer>"],
            teacher_texts=["<answer>42</answer>", "<answer>42</answer>"],
        )

        self.assertAlmostEqual(state["trajectory"][0]["reward"], 0.0)  # empty
        self.assertAlmostEqual(state["trajectory"][1]["reward"], 1.0)  # correct

    def test_all_identical_students(self):
        """All K students give the same answer."""
        env, inner_env = create_env(K=4)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["<answer>42</answer>"] * 4,
            teacher_texts=["<answer>42</answer>"] * 4,
        )

        # All scores 1.0, all advantages 0.0
        for step in state["trajectory"]:
            self.assertAlmostEqual(step["advantage"], 0.0)

    def test_no_answer_tags(self):
        """Responses without answer tags -> score 0."""
        env, inner_env = create_env(K=2)
        state = run_full_rollout(
            env, inner_env,
            student_texts=["The answer is 42", "I think 42"],
            teacher_texts=["Definitely 42", "For sure 42"],
        )

        for step in state["trajectory"]:
            self.assertAlmostEqual(step["reward"], 0.0)

    def test_empty_trajectory(self):
        """render_completion with empty trajectory -> graceful handling."""
        env, inner_env = create_env(K=2)
        state = make_state(env, inner_env=inner_env)
        run_async(env.render_completion(state))
        self.assertAlmostEqual(state["reward"], 0.0)
        self.assertEqual(state["metrics"], {})

    def test_majority_vote_metric(self):
        """Majority vote in metrics should work."""
        env, inner_env = create_env(K=3)
        state = run_full_rollout(
            env, inner_env,
            student_texts=[
                "<answer>42</answer>",
                "<answer>42</answer>",
                "<answer>99</answer>",
            ],
            teacher_texts=["<answer>42</answer>"] * 3,
        )

        # Majority of students say 42 (correct) -> majority_vote = 1.0
        self.assertAlmostEqual(state["metrics"]["majority_vote"], 1.0)


# ===========================================================================
# Test: Helpers
# ===========================================================================

class TestHelpers(unittest.TestCase):

    def test_extract_question_from_chat(self):
        env, _ = create_env(K=2)
        q = env._extract_question([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "my question"},
        ])
        self.assertEqual(q, "my question")

    def test_extract_question_from_string(self):
        env, _ = create_env(K=2)
        q = env._extract_question("plain question")
        self.assertEqual(q, "plain question")

    def test_text_from_completion(self):
        from rsa_distill.env import RSADistillEnv
        text = RSADistillEnv._text_from_completion([
            {"role": "assistant", "content": "hello world"}
        ])
        self.assertEqual(text, "hello world")

    def test_text_from_empty_completion(self):
        from rsa_distill.env import RSADistillEnv
        text = RSADistillEnv._text_from_completion([])
        self.assertEqual(text, "")

    def test_text_from_string_completion(self):
        from rsa_distill.env import RSADistillEnv
        text = RSADistillEnv._text_from_completion("plain text")
        self.assertEqual(text, "plain text")

    def test_build_chat_prompt(self):
        env, _ = create_env(K=2)
        prompt = env._build_chat_prompt("aggregation text")
        self.assertIsInstance(prompt, list)
        self.assertEqual(prompt[-1]["role"], "user")
        self.assertEqual(prompt[-1]["content"], "aggregation text")
        self.assertEqual(prompt[0]["role"], "system")

    def test_score_candidate_correct(self):
        env, inner_env = create_env(K=2)
        score = env._score_candidate("<answer>42</answer>", "0")
        self.assertAlmostEqual(score, 1.0)

    def test_score_candidate_wrong(self):
        env, inner_env = create_env(K=2)
        score = env._score_candidate("<answer>99</answer>", "0")
        self.assertAlmostEqual(score, 0.0)

    def test_score_candidate_no_tags(self):
        env, inner_env = create_env(K=2)
        score = env._score_candidate("42 without tags", "0")
        self.assertAlmostEqual(score, 0.0)


# ===========================================================================
# Test: Full Rollout Flow
# ===========================================================================

class TestFullRolloutFlow(unittest.TestCase):
    """End-to-end tests simulating the full get_prompt -> add_step -> render flow."""

    def test_k4_full_rollout(self):
        """Full rollout with K=4: 4 student + 4 teacher."""
        env, inner_env = create_env(K=4, enable_distill=True)
        state = make_state(env, inner_env=inner_env)

        student_texts = [
            "<answer>42</answer>",
            "<answer>99</answer>",
            "<answer>42</answer>",
            "<answer>42</answer>",
        ]

        # Generate student responses
        for i, text in enumerate(student_texts):
            prompt = run_async(env.get_prompt_messages(state))
            # All student prompts should be the raw question
            self.assertEqual(prompt, state["rsa_distill_raw_prompt"])
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        # Generate teacher responses (each gets leave-one-out prompt)
        teacher_texts = ["<answer>42</answer>"] * 4
        for i, text in enumerate(teacher_texts):
            prompt = run_async(env.get_prompt_messages(state))
            # Teacher prompts should be aggregation (different from raw)
            self.assertNotEqual(prompt, state["rsa_distill_raw_prompt"])
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        # Render
        run_async(env.render_completion(state))

        # Verify structure: 2K = 8 steps
        self.assertEqual(len(state["trajectory"]), 8)

        # Verify student rewards: [1,0,1,1]
        expected_student_scores = [1.0, 0.0, 1.0, 1.0]
        for i, expected in enumerate(expected_student_scores):
            self.assertAlmostEqual(state["trajectory"][i]["reward"], expected)

        # Verify teacher rewards: [1,1,1,1]
        for i in range(4, 8):
            self.assertAlmostEqual(state["trajectory"][i]["reward"], 1.0)

        # Verify student advantages (student mean = 3/4 = 0.75)
        student_mean = 0.75
        for i in range(4):
            expected_adv = expected_student_scores[i] - student_mean
            self.assertAlmostEqual(state["trajectory"][i]["advantage"], expected_adv)

        # Verify teacher advantages (teacher mean = 1.0, all advantages = 0)
        for i in range(4, 8):
            self.assertAlmostEqual(state["trajectory"][i]["advantage"], 0.0)

        # Verify prompt_text overrides (enable_distill=True)
        for i in range(4):
            # Student prompts should be overridden to leave-one-out aggregation
            prompt = state["trajectory"][i]["prompt"]
            user_msg = next(m for m in prompt if m["role"] == "user")
            self.assertIn("candidate", user_msg["content"].lower())

        # All teacher prompts should be overridden to raw question
        for i in range(4, 8):
            teacher_prompt = state["trajectory"][i]["prompt"]
            self.assertEqual(teacher_prompt, state["rsa_distill_raw_prompt"])

        # Verify metrics
        self.assertAlmostEqual(state["metrics"]["student_accuracy"], 0.75)
        self.assertAlmostEqual(state["metrics"]["teacher_accuracy"], 1.0)

    def test_k2_without_distill(self):
        """Rollout without distillation -- prompts not overridden."""
        env, inner_env = create_env(K=2, enable_distill=False)
        state = make_state(env, inner_env=inner_env)

        for text in ["<answer>42</answer>", "<answer>99</answer>"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        for text in ["<answer>42</answer>", "<answer>42</answer>"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        run_async(env.render_completion(state))

        # Prompts should NOT be overridden
        for step in state["trajectory"]:
            self.assertEqual(step["prompt"], [])  # Original empty from mock

    def test_k2_teacher_prompts_are_leave_one_out(self):
        """Verify teacher generation prompts during rollout are leave-one-out."""
        env, inner_env = create_env(K=2)
        state = make_state(env, inner_env=inner_env)

        student_texts = ["<answer>A</answer>", "<answer>B</answer>"]
        for text in student_texts:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        # Teacher 0: excludes student 0 (A), includes B only
        prompt_t0 = run_async(env.get_prompt_messages(state))
        user_t0 = next(m for m in prompt_t0 if m["role"] == "user")
        self.assertNotIn("<answer>A</answer>", user_t0["content"])
        self.assertIn("<answer>B</answer>", user_t0["content"])
        run_async(env.add_trajectory_step(state, make_trajectory_step("teacher_0")))

        # Teacher 1: excludes student 1 (B), includes A only
        prompt_t1 = run_async(env.get_prompt_messages(state))
        user_t1 = next(m for m in prompt_t1 if m["role"] == "user")
        self.assertIn("<answer>A</answer>", user_t1["content"])
        self.assertNotIn("<answer>B</answer>", user_t1["content"])


# ===========================================================================
# Test: Task Types
# ===========================================================================

class TestTaskTypes(unittest.TestCase):

    def test_rg_task(self):
        """Task='rg' should use reasoning_gym format in aggregation prompt."""
        env, inner_env = create_env(K=2, task="rg")
        state = make_state(env, inner_env=inner_env)

        for text in ["<answer>42</answer>", "<answer>99</answer>"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        prompt = run_async(env.get_prompt_messages(state))
        user_msg = next(m for m in prompt if m["role"] == "user")
        # rg format uses <answer> tags
        self.assertIn("<answer>", user_msg["content"])

    def test_math_task(self):
        """Task='math' should use math format in aggregation prompt."""
        env, inner_env = create_env(K=2, task="math")
        state = make_state(env, inner_env=inner_env)

        for text in ["\\boxed{42}", "\\boxed{99}"]:
            run_async(env.get_prompt_messages(state))
            run_async(env.add_trajectory_step(state, make_trajectory_step(text)))

        prompt = run_async(env.get_prompt_messages(state))
        user_msg = next(m for m in prompt if m["role"] == "user")
        # math format uses \boxed
        self.assertIn("boxed", user_msg["content"])


if __name__ == "__main__":
    unittest.main()
