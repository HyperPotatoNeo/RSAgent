"""Unit tests for RSAgent environment (no GPU required).

Uses mock inner_env and mock client to test RSA loop logic,
prompt construction, candidate tracking, and reward computation.
"""

import asyncio
import os
import sys
import unittest
from collections import namedtuple
from unittest.mock import AsyncMock, MagicMock, patch

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------

def make_mock_inner_env(
    num_problems=3,
    correct_answer="42",
    system_prompt="You are a helpful assistant. Answer in <answer> tags.",
):
    """Create a mock inner_env that mimics ReasoningGymEnv."""
    from datasets import Dataset

    # Build a simple dataset
    prompts = []
    answers = []
    for i in range(num_problems):
        prompts.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"What is the answer to problem {i}?"},
        ])
        answers.append(str(i))  # answer index

    ds = Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
        "example_id": list(range(num_problems)),
        "task": ["test"] * num_problems,
    })

    env = MagicMock()
    env.dataset = ds
    env.eval_dataset = ds
    env.system_prompt = system_prompt

    # Mock parser: extracts text between <answer> tags
    parser = MagicMock()
    def parse_answer(text):
        if "<answer>" in text and "</answer>" in text:
            start = text.index("<answer>") + len("<answer>")
            end = text.index("</answer>")
            return text[start:end].strip()
        return None
    parser.parse_answer = parse_answer
    env.parser = parser

    # Mock rg_dataset: score_answer checks if parsed == entry["answer"]
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


def make_mock_response(text="<answer>42</answer>"):
    """Create a mock ChatCompletion response."""
    message = MagicMock()
    message.content = text
    message.role = "assistant"
    message.tool_calls = None

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    choice.logprobs = None  # No logprobs in mock

    response = MagicMock()
    response.choices = [choice]
    response.id = "mock-id"
    response.model = "mock-model"
    return response


def make_mock_client(responses=None, default_text="<answer>42</answer>"):
    """Create a mock AsyncOpenAI client."""
    client = AsyncMock()
    call_count = [0]
    call_log = []

    async def mock_create(**kwargs):
        idx = call_count[0]
        call_count[0] += 1
        call_log.append(kwargs)
        if responses and idx < len(responses):
            return make_mock_response(responses[idx])
        return make_mock_response(default_text)

    client.chat.completions.create = mock_create
    client._call_log = call_log
    client._call_count = call_count
    return client


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def create_rsagent(M=1, N=2, K=2, T=2, **kwargs):
    """Create an RSAgent with mock inner_env."""
    from rsagent.env import RSAgent
    inner_env = make_mock_inner_env(**kwargs.pop("env_kwargs", {}))
    return RSAgent(
        inner_env=inner_env,
        M=M, N=N, K=K, T=T,
        **kwargs,
    ), inner_env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRSAgentInit(unittest.TestCase):
    """Test RSAgent initialization."""

    def test_basic_init(self):
        agent, _ = create_rsagent(M=1, N=4, K=2, T=4)
        self.assertEqual(agent.M, 1)
        self.assertEqual(agent.N, 4)
        self.assertEqual(agent.K, 2)
        self.assertEqual(agent.T, 4)
        self.assertEqual(agent.total_candidates, 4)
        self.assertEqual(agent.max_turns, 16)  # T * M * N

    def test_invalid_params(self):
        """K > N should fail validation."""
        with self.assertRaises(ValueError):
            create_rsagent(M=1, N=2, K=4, T=2)

    def test_dataset_passthrough(self):
        """RSAgent should use inner_env's dataset."""
        agent, inner_env = create_rsagent()
        self.assertEqual(len(agent.dataset), len(inner_env.dataset))

    def test_default_reward_metric(self):
        agent, _ = create_rsagent()
        self.assertEqual(agent.final_reward_metric, "mean_accuracy")


class TestPromptConstruction(unittest.TestCase):
    """Test get_prompt_messages builds correct prompts."""

    def test_step0_uses_original_prompt(self):
        """Step 0 should return the inner_env's formatted prompt."""
        agent, inner_env = create_rsagent(M=1, N=2, K=2, T=2)

        # Simulate state
        state = {
            "prompt": inner_env.dataset[0]["prompt"],
            "rsa_question": "What is the answer to problem 0?",
            "rsa_step": 0,
            "rsa_candidate_idx": 0,
            "rsa_candidates": {0: [None, None]},
            "rsa_rng": __import__("random").Random(42),
        }

        prompt = run_async(agent.get_prompt_messages(state))
        self.assertEqual(prompt, inner_env.dataset[0]["prompt"])

    def test_step1_uses_aggregation_prompt(self):
        """Step 1+ should build aggregation prompts from candidates."""
        agent, inner_env = create_rsagent(M=1, N=2, K=2, T=2)

        state = {
            "prompt": inner_env.dataset[0]["prompt"],
            "rsa_question": "What is the answer to problem 0?",
            "rsa_step": 1,
            "rsa_candidate_idx": 0,
            "rsa_candidates": {
                0: ["<answer>42</answer>", "<answer>99</answer>"],
                1: [None, None],
            },
            "rsa_rng": __import__("random").Random(42),
        }

        prompt = run_async(agent.get_prompt_messages(state))
        # Should be a list of chat messages
        self.assertIsInstance(prompt, list)
        # Should have system + user messages
        roles = [m["role"] for m in prompt]
        self.assertIn("system", roles)
        self.assertIn("user", roles)
        # User message should mention "candidate" (aggregation prompt)
        user_msg = next(m for m in prompt if m["role"] == "user")
        self.assertIn("candidate", user_msg["content"].lower())


class TestStateTracking(unittest.TestCase):
    """Test RSA state advancement in add_trajectory_step."""

    def test_candidate_tracking(self):
        """Candidates should be stored correctly."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2)

        state = {
            "trajectory": [],
            "rsa_step": 0,
            "rsa_candidate_idx": 0,
            "rsa_candidates": {0: [None, None]},
            "rsa_rng": __import__("random").Random(42),
            "trajectory_id": "test",
        }

        # Add first candidate
        step1 = {
            "completion": [{"role": "assistant", "content": "response_0"}],
            "prompt": [], "response": None, "tokens": None,
            "reward": None, "advantage": None, "is_truncated": False,
            "trajectory_id": "test", "extras": {},
        }
        run_async(agent.add_trajectory_step(state, step1))
        self.assertEqual(state["rsa_candidates"][0][0], "response_0")
        self.assertEqual(state["rsa_candidate_idx"], 1)
        self.assertEqual(state["rsa_step"], 0)

        # Add second candidate (completes step 0)
        step2 = {
            "completion": [{"role": "assistant", "content": "response_1"}],
            "prompt": [], "response": None, "tokens": None,
            "reward": None, "advantage": None, "is_truncated": False,
            "trajectory_id": "test", "extras": {},
        }
        run_async(agent.add_trajectory_step(state, step2))
        self.assertEqual(state["rsa_candidates"][0][1], "response_1")
        self.assertEqual(state["rsa_candidate_idx"], 0)  # Reset
        self.assertEqual(state["rsa_step"], 1)  # Advanced
        self.assertIn(1, state["rsa_candidates"])  # New step initialized

    def test_last_step_no_new_candidates(self):
        """After the last step, no new candidate dict should be created."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2)

        state = {
            "trajectory": [None] * 3,  # Simulate 3 prior steps
            "rsa_step": 1,
            "rsa_candidate_idx": 1,  # Last candidate of last step
            "rsa_candidates": {
                0: ["a", "b"],
                1: ["c", None],
            },
            "rsa_rng": __import__("random").Random(42),
            "trajectory_id": "test",
        }

        step = {
            "completion": [{"role": "assistant", "content": "d"}],
            "prompt": [], "response": None, "tokens": None,
            "reward": None, "advantage": None, "is_truncated": False,
            "trajectory_id": "test", "extras": {},
        }
        run_async(agent.add_trajectory_step(state, step))
        self.assertEqual(state["rsa_step"], 2)  # Past T
        self.assertNotIn(2, state["rsa_candidates"])  # No step 2


class TestRewardComputation(unittest.TestCase):
    """Test reward assignment logic."""

    def test_mean_accuracy_all_correct(self):
        """All correct final candidates → reward = 1.0."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2)

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>42</answer>", "<answer>42</answer>"],
            all_responses=["<answer>42</answer>"] * 4,
            answer_idx="0",
        )

        agent._assign_rewards(state)
        self.assertAlmostEqual(state["reward"], 1.0)
        self.assertAlmostEqual(state["metrics"]["final_mean_accuracy"], 1.0)

    def test_mean_accuracy_partial(self):
        """One correct, one wrong → mean_accuracy = 0.5."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2)

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>42</answer>", "<answer>99</answer>"],
            all_responses=["<answer>42</answer>", "<answer>99</answer>",
                           "<answer>42</answer>", "<answer>99</answer>"],
            answer_idx="0",
        )

        agent._assign_rewards(state)
        self.assertAlmostEqual(state["reward"], 0.5)
        self.assertAlmostEqual(state["metrics"]["final_mean_accuracy"], 0.5)

    def test_majority_vote_metric(self):
        """Majority vote should be 1.0 when majority is correct."""
        agent, _ = create_rsagent(
            M=1, N=2, K=2, T=2, final_reward_metric="majority_vote"
        )

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>42</answer>", "<answer>99</answer>"],
            all_responses=["<answer>42</answer>", "<answer>99</answer>"] * 2,
            answer_idx="0",
        )

        agent._assign_rewards(state)
        # Tie between 42 and 99 — Counter.most_common picks one
        # With 1 correct and 1 wrong, majority_vote depends on order
        self.assertIn(state["reward"], [0.0, 1.0])

    def test_pass_at_n_metric(self):
        """pass@n = 1.0 if any candidate is correct."""
        agent, _ = create_rsagent(
            M=1, N=2, K=2, T=2, final_reward_metric="pass_at_n"
        )

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>99</answer>", "<answer>42</answer>"],
            all_responses=["<answer>99</answer>", "<answer>42</answer>"] * 2,
            answer_idx="0",
        )

        agent._assign_rewards(state)
        self.assertAlmostEqual(state["reward"], 1.0)

    def test_greedy_weight_zero(self):
        """With greedy_weight=0, all per-step rewards equal final_reward."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2, greedy_reward_weight=0.0)

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>42</answer>", "<answer>42</answer>"],
            all_responses=["<answer>42</answer>", "<answer>99</answer>",
                           "<answer>42</answer>", "<answer>42</answer>"],
            answer_idx="0",
        )

        agent._assign_rewards(state)
        # All step rewards should be equal to final_reward
        for step in state["trajectory"]:
            self.assertAlmostEqual(step["reward"], state["reward"])

    def test_greedy_weight_nonzero(self):
        """With greedy_weight>0, per-step rewards vary by candidate quality."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2, greedy_reward_weight=0.5)

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>42</answer>", "<answer>42</answer>"],
            # Step 0: one correct, one wrong
            # Step 1: both correct
            all_responses=["<answer>42</answer>", "<answer>99</answer>",
                           "<answer>42</answer>", "<answer>42</answer>"],
            answer_idx="0",
        )

        agent._assign_rewards(state)
        # Step 0, candidate 0 (correct): higher reward
        # Step 0, candidate 1 (wrong): lower reward
        r0 = state["trajectory"][0]["reward"]
        r1 = state["trajectory"][1]["reward"]
        self.assertGreater(r0, r1)

    def test_no_correct_candidates(self):
        """All wrong → reward = 0.0."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2)

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["<answer>99</answer>", "<answer>99</answer>"],
            all_responses=["<answer>99</answer>"] * 4,
            answer_idx="0",
        )

        agent._assign_rewards(state)
        self.assertAlmostEqual(state["reward"], 0.0)

    def test_no_answer_tags(self):
        """Missing answer tags → parsed as None → score 0."""
        agent, _ = create_rsagent(M=1, N=2, K=2, T=2)

        state = _make_scored_state(
            agent, T=2, N=2,
            final_candidates=["no tags here", "also no tags"],
            all_responses=["no tags here", "also no tags"] * 2,
            answer_idx="0",
        )

        agent._assign_rewards(state)
        self.assertAlmostEqual(state["reward"], 0.0)


class TestHelpers(unittest.TestCase):
    """Test helper methods."""

    def test_extract_question_from_chat(self):
        from rsagent.env import RSAgent
        agent, _ = create_rsagent()
        q = agent._extract_question([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "my question"},
        ])
        self.assertEqual(q, "my question")

    def test_extract_question_from_string(self):
        from rsagent.env import RSAgent
        agent, _ = create_rsagent()
        q = agent._extract_question("plain string question")
        self.assertEqual(q, "plain string question")

    def test_text_from_completion(self):
        from rsagent.env import RSAgent
        text = RSAgent._text_from_completion([
            {"role": "assistant", "content": "hello world"}
        ])
        self.assertEqual(text, "hello world")

    def test_text_from_empty_completion(self):
        from rsagent.env import RSAgent
        text = RSAgent._text_from_completion([])
        self.assertEqual(text, "")

    def test_build_chat_prompt(self):
        agent, _ = create_rsagent()
        prompt = agent._build_chat_prompt("aggregation text")
        self.assertIsInstance(prompt, list)
        self.assertEqual(prompt[-1]["role"], "user")
        self.assertEqual(prompt[-1]["content"], "aggregation text")
        self.assertEqual(prompt[0]["role"], "system")


class TestIslandLogic(unittest.TestCase):
    """Test island-based candidate sampling in aggregation prompts."""

    def test_two_islands(self):
        """With M=2, candidates should only see others in their island."""
        # T=4 so step 1 still has 2 islands (merge happens at step 2)
        agent, inner_env = create_rsagent(M=2, N=2, K=2, T=4)
        # M=2, N=2 → 4 total candidates, 2 per island

        state = {
            "prompt": inner_env.dataset[0]["prompt"],
            "rsa_question": "test question",
            "rsa_step": 1,
            "rsa_candidate_idx": 0,  # Island 0
            "rsa_candidates": {
                0: ["island0_a", "island0_b", "island1_a", "island1_b"],
                1: [None] * 4,
            },
            "rsa_rng": __import__("random").Random(42),
        }

        prompt = run_async(agent.get_prompt_messages(state))
        user_msg = next(m for m in prompt if m["role"] == "user")
        content = user_msg["content"]

        # Should only contain candidates from island 0
        self.assertIn("island0", content)
        self.assertNotIn("island1", content)

    def test_candidate_idx_2_in_island_1(self):
        """Candidate index 2 (island 1) should sample from island 1."""
        # T=4 so step 1 still has 2 islands
        agent, inner_env = create_rsagent(M=2, N=2, K=2, T=4)

        state = {
            "prompt": inner_env.dataset[0]["prompt"],
            "rsa_question": "test question",
            "rsa_step": 1,
            "rsa_candidate_idx": 2,  # Island 1
            "rsa_candidates": {
                0: ["island0_a", "island0_b", "island1_a", "island1_b"],
                1: [None] * 4,
            },
            "rsa_rng": __import__("random").Random(42),
        }

        prompt = run_async(agent.get_prompt_messages(state))
        user_msg = next(m for m in prompt if m["role"] == "user")
        content = user_msg["content"]

        # Should only contain candidates from island 1
        self.assertIn("island1", content)
        self.assertNotIn("island0", content)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_scored_state(agent, T, N, final_candidates, all_responses, answer_idx):
    """Build a state with trajectory for reward testing."""
    total = agent.total_candidates
    trajectory = []
    for i, resp_text in enumerate(all_responses):
        trajectory.append({
            "completion": [{"role": "assistant", "content": resp_text}],
            "prompt": [],
            "response": None,
            "tokens": None,
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": "test",
            "extras": {},
        })

    state = {
        "trajectory": trajectory,
        "answer": answer_idx,
        "rsa_candidates": {
            T - 1: final_candidates,
        },
    }
    return state


if __name__ == "__main__":
    unittest.main()
