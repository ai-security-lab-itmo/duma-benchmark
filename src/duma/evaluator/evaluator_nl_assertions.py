import json

from duma.config import DEFAULT_LLM_NL_ASSERTIONS, DEFAULT_LLM_NL_ASSERTIONS_ARGS
from duma.data_model.message import Message, SystemMessage, UserMessage
from duma.data_model.simulation import NLAssertionCheck, RewardInfo
from duma.data_model.tasks import RewardType, Task
from duma.utils.llm_utils import generate


class NLAssertionsEvaluator:
    """
    Judge that evaluates whether a trajectory adheres to all the natural-language assertions.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
        solo_mode: bool = False,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation by using an LLM to evaluate whether the trajectory adheres to all the natural-language assertions
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )
        nl_assertions = task.evaluation_criteria.nl_assertions
        if not nl_assertions:
            return RewardInfo(
                reward=1.0,
                nl_assertions=[],
                info={"note": "No nl_assertions to evaluate"},
                reward_breakdown={RewardType.NL_ASSERTION: 1.0},
            )

        nl_assertions_checks = cls.evaluate_nl_assertions(
            full_trajectory, nl_assertions, solo_mode=solo_mode
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.met for result in nl_assertions_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            nl_assertions=nl_assertions_checks,
            reward_breakdown={RewardType.NL_ASSERTION: reward},
        )

    @classmethod
    def evaluate_nl_assertions(
        cls,
        trajectory: list[Message],
        nl_assertions: list[str],
        solo_mode: bool = False,
    ) -> list[NLAssertionCheck]:
        """
        Evaluate whether the trajectory meets each expected outcome.

        Args:
            trajectory: List of messages from the conversation
            nl_assertions: List of natural-language assertions to evaluate
            solo_mode: If True, adjust evaluation for solo agent runs (no user messages)

        Returns:
            List of evaluation results for each NL assertion, containing:
            - nl_assertion: The NL assertion being evaluated
            - metExpectation: Boolean indicating if the assertion was met
            - reasoning: Explanation for the evaluation
        """

        def format_message(message):
            if message.role == "assistant" and message.content is None:
                # Show tool call names for solo mode visibility
                if hasattr(message, "tool_calls") and message.tool_calls:
                    calls = ", ".join(tc.name for tc in message.tool_calls)
                    return f"assistant: [tool calls: {calls}]"
                return None  # skip empty messages
            return f"{message.role}: {message.content}"

        trajectory_str = "\n".join(
            filter(None, [format_message(msg) for msg in trajectory])
        )

        solo_context = ""
        if solo_mode:
            solo_context = """
        NOTE: This is a solo agent run — there are no user messages, only tool calls and a
        resolution_note at the end. Evaluate the assertions based on the tool call results
        and the resolution_note content. The agent communicates its reasoning through the
        resolution_note at the end of the conversation.
"""

        # System prompt similar to the TypeScript implementation
        system_prompt = f"""
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.
{solo_context}
        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading

        Example response structure:
        {{
            "results": [
                {{
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }}
            ]
        }}
        """

        user_prompt = f"""
        conversation:
        {trajectory_str}
        
        expectedOutcomes:
        {nl_assertions}
        """

        messages = [
            SystemMessage(role="system", content=system_prompt),
            UserMessage(role="user", content=user_prompt),
        ]

        assistant_message = generate(
            model=DEFAULT_LLM_NL_ASSERTIONS,
            messages=messages,
            **DEFAULT_LLM_NL_ASSERTIONS_ARGS,
        )
        result_data = json.loads(assistant_message.content)
        return [
            NLAssertionCheck(
                nl_assertion=result["expectedOutcome"],
                met=result["metExpectation"],
                justification=result["reasoning"],
            )
            for result in result_data.get("results", [])
        ]
