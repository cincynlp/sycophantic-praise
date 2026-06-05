from __future__ import annotations

import json
from enum import Enum
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from tqdm import tqdm

from sypr.schemas import (
    AtIssueStatus,
    BeliefFraming,
    BeliefOwner,
    BenchmarkArtifact,
    BenchmarkInstance,
    ContextMetadata,
    ContextType,
    ExperimentMetadata,
    Persona,
    PersonaMetadata,
    PersonaSecurity,
    PromptCondition,
    Utterance,
    UtteranceMetadata,
)


def stable_uuid(kind: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return str(uuid5(NAMESPACE_URL, f"sypr:{kind}:{canonical}"))


def _enum_or_default(enum_cls: type[Enum], value: Any, default: Enum) -> Enum:
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _scalar_value(values: dict[str, float], relevance: dict[str, float]) -> float | None:
    total_weight = sum(relevance.values())
    if total_weight <= 0:
        return None
    return sum(relevance[dim] * values[dim] for dim in values if dim in relevance) / total_weight


def _count_tokens_roughly(turns: list[dict[str, str]]) -> int:
    return sum(len(turn.get("content", "").split()) for turn in turns)


def _persona_history(persona: Persona) -> list[dict[str, str]]:
    history = persona.metadata.get("conversation_history", [])
    if not isinstance(history, list):
        return []
    turns: list[dict[str, str]] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = turn.get("content")
        if role is None or content is None:
            continue
        turns.append({"role": str(role), "content": str(content)})
    return turns


def build_persona_metadata(persona: Persona) -> PersonaMetadata:
    metadata = persona.metadata
    return PersonaMetadata(
        persona_id=persona.persona_id,
        persona_type=metadata.get("persona_type") or metadata.get("persona_group"),
        persona_source=metadata.get("persona_source"),
        persona_security=_enum_or_default(
            PersonaSecurity,
            metadata.get("persona_security"),
            PersonaSecurity.UNKNOWN,
        ),
        persona_self_confidence_score=metadata.get("persona_self_confidence_score"),
        persona_validation_seeking=metadata.get("persona_validation_seeking"),
        persona_validation_tendency=metadata.get("persona_validation_tendency"),
        persona_text=persona.context,
        expected_ability_static=persona.expected_abilities,
        expected_ability_static_std=metadata.get("expected_ability_static_std", {}),
        expected_ability_from_context_ground_truth=metadata.get(
            "expected_ability_from_context_ground_truth",
            {},
        ),
        expected_ability_from_context_ground_truth_std=metadata.get(
            "expected_ability_from_context_ground_truth_std",
            {},
        ),
        expected_ability_from_context_llm_estimated=metadata.get(
            "expected_ability_from_context_llm_estimated",
            {},
        ),
        expected_ability_from_context_llm_estimated_std=metadata.get(
            "expected_ability_from_context_llm_estimated_std",
            metadata.get("expected_ability_estimated_std", {}),
        ),
        expected_ability_variance=metadata.get("expected_ability_variance"),
        metadata=metadata,
    )


def build_utterance_metadata(utterance: Utterance) -> UtteranceMetadata:
    metadata = utterance.metadata
    scalar_ground_truth_value = _scalar_value(utterance.value, utterance.relevance)
    difficulty = metadata.get("difficulty")
    difficulty_score = metadata.get("difficulty_score")
    if difficulty_score is None and isinstance(difficulty, (int, float)):
        difficulty_score = float(difficulty)

    return UtteranceMetadata(
        utterance_id=utterance.utterance_id,
        utterance_source=metadata.get("utterance_source"),
        utterance_text=utterance.text,
        domain=metadata.get("domain"),
        capability=metadata.get("capability") or metadata.get("value_type"),
        subdomain=metadata.get("subdomain"),
        difficulty_source=metadata.get("difficulty_source"),
        difficulty=str(difficulty) if difficulty is not None else None,
        difficulty_score=difficulty_score,
        difficulty_bin=metadata.get("difficulty_bin"),
        human_difficulty=metadata.get("human_difficulty"),
        irt_difficulty=metadata.get("irt_difficulty"),
        ground_truth_value=metadata.get("ground_truth_value", scalar_ground_truth_value),
        ground_truth_correctness=metadata.get("ground_truth_correctness")
        if "ground_truth_correctness" in metadata
        else (
            scalar_ground_truth_value >= 0.5
            if scalar_ground_truth_value is not None
            else None
        ),
        value_dimensions=utterance.value,
        llm_rated_value=metadata.get("llm_rated_value"),
        llm_rated_correctness=metadata.get("llm_rated_correctness"),
        llm_rated_value_dimensions=metadata.get("llm_rated_value_dimensions", {}),
        belief_framing=_enum_or_default(
            BeliefFraming,
            metadata.get("belief_framing"),
            BeliefFraming.NONE,
        ),
        at_issue_status=_enum_or_default(
            AtIssueStatus,
            metadata.get("at_issue_status"),
            AtIssueStatus.UNKNOWN,
        ),
        belief_owner=_enum_or_default(
            BeliefOwner,
            metadata.get("belief_owner"),
            BeliefOwner.UNKNOWN,
        ),
        metadata=metadata,
    )


def build_context_metadata(
    condition: dict[str, Any],
    conversation_history: list[dict[str, str]],
) -> ContextMetadata:
    metadata = dict(condition.get("metadata", {}))
    context_type = metadata.get("context_type") or condition.get("context_type")
    return ContextMetadata(
        context_id=stable_uuid(
            "context",
            {
                "condition": condition.get("name"),
                "conversation_history": conversation_history,
                "metadata": metadata,
            },
        ),
        context_type=_enum_or_default(ContextType, context_type, ContextType.NONE),
        context_text="\n".join(
            f"{turn['role']}: {turn['content']}" for turn in conversation_history
        )
        or None,
        context_length_turns=len(conversation_history),
        context_length_tokens=metadata.get(
            "context_length_tokens",
            _count_tokens_roughly(conversation_history),
        ),
        context_length_bucket=metadata.get("context_length_bucket")
        or condition.get("context_length_bucket"),
        context_question_count=metadata.get("context_question_count", 0),
        context_difficulty_bins=metadata.get("context_difficulty_bins", []),
        context_difficulty_mean=metadata.get("context_difficulty_mean"),
        context_difficulty_std=metadata.get("context_difficulty_std"),
        context_user_accuracy=metadata.get("context_user_accuracy"),
        context_model_accuracy=metadata.get("context_model_accuracy"),
        context_user_correct_count=metadata.get("context_user_correct_count"),
        context_user_incorrect_count=metadata.get("context_user_incorrect_count"),
        context_domains=metadata.get("context_domains", []),
        context_capabilities=metadata.get("context_capabilities", []),
        metadata=metadata,
    )


def _is_calibrated_cross_difficulty_persona(persona: Persona) -> bool:
    metadata = persona.metadata
    return (
        metadata.get("persona_type") == "calibrated_question_history"
        or metadata.get("persona_source") == "BatsResearch/Cross-Difficulty"
        or metadata.get("context_mode") == "calibrated_question_answering"
    )


def _benchmark_task_for_utterance(utterance: Utterance) -> str | None:
    task = utterance.metadata.get("benchmark_task")
    if task:
        return str(task)
    for key in utterance.value:
        if key.startswith("benchmark:"):
            return key.removeprefix("benchmark:")
    return None


def _calibrated_persona_task(persona: Persona) -> str | None:
    task = persona.metadata.get("benchmark_task")
    if task:
        return str(task)
    ability_key = persona.metadata.get("benchmark_ability_key")
    if isinstance(ability_key, str) and ability_key.startswith("benchmark:"):
        return ability_key.removeprefix("benchmark:")
    benchmark_keys = [
        key.removeprefix("benchmark:")
        for key in persona.expected_abilities
        if key.startswith("benchmark:")
    ]
    return benchmark_keys[0] if len(benchmark_keys) == 1 else None


def _persona_benchmark_tasks(persona: Persona) -> set[str]:
    tasks = persona.metadata.get("benchmark_tasks")
    if isinstance(tasks, str):
        return {tasks}
    if isinstance(tasks, list):
        return {str(task) for task in tasks}
    task = persona.metadata.get("benchmark_task")
    if task:
        return {str(task)}
    return set()


def persona_applies_to_utterance(persona: Persona, utterance: Utterance) -> bool:
    utterance_task = _benchmark_task_for_utterance(utterance)
    persona_tasks = _persona_benchmark_tasks(persona)
    if persona_tasks:
        return utterance_task in persona_tasks
    if not _is_calibrated_cross_difficulty_persona(persona):
        return True
    persona_task = _calibrated_persona_task(persona)
    return persona_task is not None and persona_task == utterance_task


def build_instances(config: dict) -> list[BenchmarkInstance]:
    personas = [Persona(**p) for p in config["personas"]]
    utterances = [Utterance(**u) for u in config["utterances"]]

    prompt_conditions = config.get(
        "prompt_conditions",
        [{"name": "unprompted", "conversation_history": []}],
    )

    instances: list[BenchmarkInstance] = []

    for persona in tqdm(personas, desc="Building instances from personas and utterances", unit="persona"):
        persona_metadata = persona.persona_metadata or build_persona_metadata(persona)
        persona_with_metadata = persona.model_copy(
            update={"persona_metadata": persona_metadata}
        )
        persona_conversation_history = _persona_history(persona)
        for utterance in utterances:
            if not persona_applies_to_utterance(persona, utterance):
                continue
            utterance_metadata = utterance.utterance_metadata or build_utterance_metadata(
                utterance
            )
            for condition in prompt_conditions:
                condition_name = condition["name"]
                prompt_condition = PromptCondition(condition_name)
                utterance_suffix = str(condition.get("utterance_suffix") or "")
                condition_utterance = utterance
                if utterance_suffix:
                    condition_utterance = utterance.model_copy(
                        update={"text": f"{utterance.text.rstrip()}{utterance_suffix}"}
                    )
                utterance_with_metadata = condition_utterance.model_copy(
                    update={"utterance_metadata": utterance_metadata}
                )
                conversation_history = [
                    *persona_conversation_history,
                    *condition.get("conversation_history", []),
                ]
                context_metadata = build_context_metadata(condition, conversation_history)
                experiment_metadata = ExperimentMetadata(
                    experiment_id=config.get("experiment_id")
                    or config.get("metadata", {}).get("experiment_id"),
                    condition_id=stable_uuid(
                        "condition",
                        {
                            "prompt_condition": prompt_condition.value,
                            "condition": condition,
                        },
                    ),
                    prompt_condition=prompt_condition.value,
                    is_prompted_for_evaluation=condition.get(
                        "is_prompted_for_evaluation",
                        prompt_condition
                        in {
                            PromptCondition.ASKED_FOR_EVAL,
                            PromptCondition.ASKED_WHAT_DO_YOU_THINK,
                        },
                    ),
                    is_prompted_for_reassurance=condition.get(
                        "is_prompted_for_reassurance",
                        prompt_condition == PromptCondition.ASKED_FOR_REASSURANCE,
                    ),
                    is_prompted_at_issue_self=condition.get(
                        "is_prompted_at_issue_self",
                        prompt_condition == PromptCondition.AT_ISSUE_SELF,
                    ),
                    is_prompted_at_issue_other=condition.get(
                        "is_prompted_at_issue_other",
                        prompt_condition == PromptCondition.AT_ISSUE_OTHER,
                    ),
                    split=config.get("split") or config.get("metadata", {}).get("split"),
                    random_seed=config.get("random_seed")
                    or config.get("metadata", {}).get("random_seed"),
                    metadata={**config.get("metadata", {}), **condition.get("metadata", {})},
                )

                instance_id = stable_uuid(
                    "instance",
                    {
                        "persona_id": persona.persona_id,
                        "utterance_id": utterance.utterance_id,
                        "prompt_condition": prompt_condition.value,
                        "context_id": context_metadata.context_id,
                        "experiment": experiment_metadata.model_dump(mode="json"),
                    },
                )

                instances.append(
                    BenchmarkInstance(
                        instance_id=instance_id,
                        instance_metadata=experiment_metadata,
                        persona=persona_with_metadata,
                        utterance=utterance_with_metadata,
                        prompt_condition=prompt_condition,
                        conversation_history=conversation_history,
                        context_metadata=context_metadata,
                        metadata={
                            **config.get("metadata", {}),
                            "domain": utterance.metadata.get("domain"),
                            "difficulty": utterance.metadata.get("difficulty"),
                            "difficulty_score": utterance_metadata.difficulty_score,
                            "difficulty_bin": utterance_metadata.difficulty_bin,
                            "persona_type": persona_metadata.persona_type,
                            "persona_security": persona_metadata.persona_security.value,
                            "at_issue_status": utterance_metadata.at_issue_status.value,
                            "belief_framing": utterance_metadata.belief_framing.value,
                            "belief_owner": utterance_metadata.belief_owner.value,
                            "context_length_turns": context_metadata.context_length_turns,
                            "context_length_tokens": context_metadata.context_length_tokens,
                            "context_length_bucket": context_metadata.context_length_bucket,
                            "context_question_count": context_metadata.context_question_count,
                            "persona_context_turns": persona_conversation_history,
                            "prompt_condition_turns": condition.get(
                                "conversation_history",
                                [],
                            ),
                        },
                    )
                )

    return instances


def build_artifacts(config: dict) -> list[BenchmarkArtifact]:
    artifacts: list[BenchmarkArtifact] = []
    for instance in build_instances(config):
        artifacts.append(
            BenchmarkArtifact(
                artifact_id=instance.instance_id,
                instance_metadata=instance.instance_metadata,
                persona=instance.persona,
                utterance=instance.utterance,
                context=instance.conversation_history,
                context_metadata=instance.context_metadata,
                responses=[],
                metadata=instance.metadata,
            )
        )
    return artifacts
