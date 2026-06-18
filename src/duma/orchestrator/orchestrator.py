import time
import uuid
from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from loguru import logger

from duma.agent.base import BaseAgent, is_valid_agent_history_message
from duma.agent.llm_agent import LLMSoloAgent
from duma.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from duma.data_model.simulation import SimulationRun, TerminationReason
from duma.data_model.tasks import EnvFunctionCall, InitializationData, Task
from duma.environment.environment import Environment, EnvironmentInfo
from duma.user.base import BaseUser, is_valid_user_history_message
from duma.user.user_simulator import DummyUser, UserSimulator, UserState
from duma.utils.llm_utils import get_cost
from duma.utils.signatures import message_signature, tool_signature
from duma.utils.utils import format_time, get_now


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)


class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        loop_guard_window: int = 10,
        loop_guard_max_unique_messages: int = 2,
        max_completion_tokens_per_message: Optional[int] = 5000,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.solo_mode = solo_mode
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None
        self.loop_guard_window = max(loop_guard_window, 0)
        self.loop_guard_max_unique_messages = max(loop_guard_max_unique_messages, 1)
        self.max_completion_tokens_per_message = (
            max_completion_tokens_per_message
            if max_completion_tokens_per_message is None
            or max_completion_tokens_per_message > 0
            else None
        )
        self._recent_participant_signatures: deque[str] = deque(
            maxlen=self.loop_guard_window if self.loop_guard_window > 0 else 1
        )
        # Prose-insensitive tool-call signatures, to catch loops where the agent
        # repeats the same tool call while varying its surrounding text.
        self._recent_tool_signatures: deque[str] = deque(
            maxlen=self.loop_guard_window if self.loop_guard_window > 0 else 1
        )

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None

        # Add timestamps to the message history
        message_history = self._add_timestamps(message_history)

        if self.solo_mode:
            assert self.environment.solo_mode, "Environment should be in solo mode"
            assert isinstance(self.agent, LLMSoloAgent), (
                "Agent must be a LLMSoloAgent in solo mode"
            )
            assert isinstance(self.user, DummyUser), (
                "User must be a DummyUser in solo mode"
            )

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds for the agent, user
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Initialize the agent and user states
        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            # Last message is an assistant message
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():  # Last message is for the user
                    self.to_role = Role.USER
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            # Last message is a user message
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():  # Last message is for the agent
                    self.to_role = Role.AGENT
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            # Last message is a tool message
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history

        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            if not self.solo_mode:
                first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                first_message.timestamp = get_now()
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.USER
            else:
                first_message, agent_state = self.agent.generate_next_message(
                    None, self.agent_state
                )
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.ENV
                self.done = self.agent.is_stop(first_message)
                if self.done:
                    self.termination_reason = TerminationReason.AGENT_STOP

        self.environment.sync_tools()
        self._seed_loop_guard_from_history()

    def _seed_loop_guard_from_history(self):
        self._recent_participant_signatures.clear()
        self._recent_tool_signatures.clear()
        if self.loop_guard_window <= 0:
            return
        for msg in self.trajectory:
            if isinstance(msg, (AssistantMessage, UserMessage)):
                self._recent_participant_signatures.append(message_signature(msg))
                tool_sig = tool_signature(msg)
                if tool_sig is not None:
                    self._recent_tool_signatures.append(tool_sig)

    def _mark_error(
        self, context: str, exc: Optional[Exception] = None, *, fatal: bool = False
    ):
        if exc is not None:
            logger.warning(f"{context}: {exc}")
        else:
            logger.warning(context)
        self.num_errors += 1
        if fatal or self.num_errors >= self.max_errors:
            self.done = True
            self.termination_reason = TerminationReason.TOO_MANY_ERRORS
            self.num_errors = max(self.num_errors, self.max_errors)

    def _check_message_token_guard(self, message: AssistantMessage | UserMessage):
        if self.max_completion_tokens_per_message is None:
            return
        usage = message.usage or {}
        completion_tokens = usage.get("completion_tokens")
        if not isinstance(completion_tokens, (int, float)):
            return
        if completion_tokens > self.max_completion_tokens_per_message:
            self._mark_error(
                (
                    f"Completion token guard triggered for {message.role} message: "
                    f"{completion_tokens} > {self.max_completion_tokens_per_message}"
                ),
                fatal=True,
            )

    def _check_loop_guard(self, message: AssistantMessage | UserMessage):
        if self.loop_guard_window <= 0:
            return
        # Content-sensitive guard: near-identical participant turns.
        self._recent_participant_signatures.append(message_signature(message))
        if len(self._recent_participant_signatures) >= self.loop_guard_window:
            unique_messages = len(set(self._recent_participant_signatures))
            if unique_messages <= self.loop_guard_max_unique_messages:
                recent_preview = " || ".join(
                    list(self._recent_participant_signatures)[-4:]
                )
                self._mark_error(
                    (
                        "Loop guard triggered: "
                        f"{unique_messages} unique participant messages in the last "
                        f"{self.loop_guard_window} messages. Recent={recent_preview}"
                    ),
                    fatal=True,
                )
                return
        # Prose-insensitive guard: the SAME tool call repeated with varied text
        # (e.g. an agent re-sending the same verification code each turn). Strict —
        # fires only when the whole window collapses to a single identical tool call,
        # so legitimate alternation between a couple of (e.g. read-only) tools does not
        # false-trigger an early termination.
        tool_sig = tool_signature(message)
        if tool_sig is not None:
            self._recent_tool_signatures.append(tool_sig)
            if (
                len(self._recent_tool_signatures) >= self.loop_guard_window
                and len(set(self._recent_tool_signatures)) <= 1
            ):
                self._mark_error(
                    "Tool loop guard triggered: identical tool call repeated "
                    f"{self.loop_guard_window} times.",
                    fatal=True,
                )

    def _append_participant_message(self, message: AssistantMessage | UserMessage):
        self.trajectory.append(message)
        self._check_message_token_guard(message)
        if self.done:
            return
        self._check_loop_guard(message)

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.initialize()
        while not self.done:
            self.step()
            if not self.done and self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
            if not self.done and self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        duration = time.perf_counter() - start
        messages = self.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res
        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            assistant_system_prompt=getattr(self.agent, "system_prompt", None),
            user_system_prompt=getattr(self.user, "system_prompt", None),
            seed=self.seed,
        )
        return simulation_run

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        try:
            # AGENT/ENV -> USER
            if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
                try:
                    user_msg, self.user_state = self.user.generate_next_message(
                        self.message, self.user_state
                    )
                    user_msg.validate()
                except Exception as e:
                    self._mark_error(
                        "Failed to generate/validate user message",
                        e,
                        fatal=True,
                    )
                    return
                self._append_participant_message(user_msg)
                if not self.done and UserSimulator.is_stop(user_msg):
                    self.done = True
                    self.termination_reason = TerminationReason.USER_STOP
                self.message = user_msg
                self.from_role = Role.USER
                if user_msg.is_tool_call():
                    self.to_role = Role.ENV
                else:
                    self.to_role = Role.AGENT
            # USER/ENV -> AGENT
            elif (
                self.from_role == Role.USER or self.from_role == Role.ENV
            ) and self.to_role == Role.AGENT:
                try:
                    agent_msg, self.agent_state = self.agent.generate_next_message(
                        self.message, self.agent_state
                    )
                    agent_msg.validate()
                except Exception as e:
                    self._mark_error(
                        "Failed to generate/validate agent message",
                        e,
                        fatal=True,
                    )
                    return
                self._append_participant_message(agent_msg)
                if not self.done and self.agent.is_stop(agent_msg):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
                self.message = agent_msg
                self.from_role = Role.AGENT
                if agent_msg.is_tool_call():
                    self.to_role = Role.ENV
                else:
                    self.to_role = Role.USER
            # AGENT/USER -> ENV
            elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
                if not self.message.is_tool_call():
                    self._mark_error(
                        "Agent or User should send tool call to environment",
                        fatal=True,
                    )
                    return
                tool_msgs = []
                for tool_call in self.message.tool_calls:
                    tool_msg = self.environment.get_response(tool_call)
                    tool_msgs.append(tool_msg)
                if len(self.message.tool_calls) != len(tool_msgs):
                    self._mark_error(
                        "Number of tool calls and tool messages should be the same",
                        fatal=True,
                    )
                    return
                self.trajectory.extend(tool_msgs)
                if (
                    len(tool_msgs) > 1
                ):  # Packaging multiple tool messages into a MultiToolMessage
                    self.message = MultiToolMessage(
                        role="tool",
                        tool_messages=tool_msgs,
                    )
                else:
                    self.message = tool_msgs[0]
                self.to_role = self.from_role
                self.from_role = Role.ENV
            else:
                self._mark_error(
                    f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}",
                    fatal=True,
                )
                return
        finally:
            self.step_count += 1
            self.environment.sync_tools()

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages == 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ):
        """
        Initialize the environment.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _get_environment_info(self) -> EnvironmentInfo:
        """
        Get the environment info.
        """
        return self.environment.get_info()

    def _count_errors(self, message_history: list[Message]) -> int:
        """
        Count the number of errors in the message history.
        """
        return sum(
            1 for msg in message_history if isinstance(msg, ToolMessage) and msg.error
        )

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history
