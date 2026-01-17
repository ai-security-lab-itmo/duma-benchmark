from tau2.data_model.message import AssistantMessage, Message, ToolCall, UserMessage
from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase


class ActionEvaluator(EvaluatorBase):
    """
    Evaluates whether or not the agent communicated the required information.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent communicated the required information.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(full_trajectory, golden_actions)

        # Separate critical and non-critical actions
        critical_checks = [r for r in action_checks if r.action.critical]
        non_critical_checks = [r for r in action_checks if not r.action.critical]

        # If any critical action failed, reward is 0
        if critical_checks and not all(r.action_match for r in critical_checks):
            reward = 0.0
        else:
            # Non-critical actions contribute proportionally
            if non_critical_checks:
                reward = sum(r.action_reward for r in non_critical_checks) / len(non_critical_checks)
            else:
                reward = 1.0

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
        )

    @classmethod
    def evaluate_actions(
        cls,
        full_trajectory: list[Message],
        golden_actions: list[Action],
    ) -> list[ActionCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(golden_actions) == 0:
            return []

        predicted_tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
                or isinstance(message, UserMessage)
            ) and message.is_tool_call():
                predicted_tool_calls.extend(message.tool_calls)

        # Check if all the gold actions are in the predicted actions
        action_checks = []
        for gold_action in golden_actions:
            found = False
            for pred_tool_call in predicted_tool_calls:
                if gold_action.compare_with_tool_call(pred_tool_call):
                    found = True
                    break
            if not found:
                gold_action_reward = 0.0
                gold_action_match = False
            else:
                gold_action_reward = 1.0
                gold_action_match = True
            action_checks.append(
                ActionCheck(
                    action=gold_action,
                    action_match=gold_action_match,
                    action_reward=gold_action_reward,
                )
            )
        return action_checks
