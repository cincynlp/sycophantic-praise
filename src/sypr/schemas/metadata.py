from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# ENUMS
# ============================================================


class PromptCondition(str, Enum):
    UNPROMPTED = "unprompted"
    ASKED_FOR_EVAL = "asked_for_eval"
    ASKED_FOR_REASSURANCE = "asked_for_reassurance"
    ASKED_WHAT_DO_YOU_THINK = "asked_what_do_you_think"
    SUPPORTIVE_CONTEXT = "supportive_context"
    AT_ISSUE_SELF = "at_issue_self"
    AT_ISSUE_OTHER = "at_issue_other"
    HIGH_STAKES = "high_stakes"
    IDENTITY_INVOLVED = "identity_involved"
    OTHER = "other"


class PersonaSecurity(str, Enum):
    SECURE = "secure"
    NEUTRAL = "neutral"
    INSECURE = "insecure"
    UNKNOWN = "unknown"


class BeliefFraming(str, Enum):
    FIRST_PERSON_BELIEF = "first_person_belief"
    THIRD_PERSON_BELIEF = "third_person_belief"
    REPORTED_BELIEF = "reported_belief"
    DIRECT_ASSERTION = "direct_assertion"
    QUESTION_ONLY = "question_only"
    NONE = "none"
    OTHER = "other"


class AtIssueStatus(str, Enum):
    AT_ISSUE = "at_issue"
    NOT_AT_ISSUE = "not_at_issue"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


class BeliefOwner(str, Enum):
    USER = "user"
    THIRD_PARTY = "third_party"
    NONE = "none"
    UNKNOWN = "unknown"


class ContextType(str, Enum):
    NONE = "none"
    EXPLICIT_PERSONA = "explicit_persona"
    NATURALISTIC_DIALOGUE = "naturalistic_dialogue"
    USER_ASKS_QUESTIONS = "user_asks_questions"
    USER_ANSWERS_QUESTIONS = "user_answers_questions"
    MIXED_QUESTIONS_CONTROL = "mixed_questions_control"
    OTHER = "other"


# ============================================================
# EXPERIMENT METADATA
# ============================================================


class FrozenMetadataModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class ExperimentMetadata(FrozenMetadataModel):
    """
    Global experimental condition metadata.
    """

    experiment_id: str | None = None
    condition_id: str | None = None

    prompt_condition: PromptCondition = PromptCondition.UNPROMPTED

    is_prompted_for_evaluation: bool = False
    is_prompted_for_reassurance: bool = False
    is_prompted_at_issue_self: bool = False
    is_prompted_at_issue_other: bool = False

    risk_level: str | None = None
    high_stakes: bool = False

    split: str | None = None
    random_seed: int | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# PERSONA METADATA
# ============================================================


class PersonaMetadata(FrozenMetadataModel):
    """
    Persona and ability prior metadata.
    """

    persona_id: str

    persona_type: str | None = None
    persona_source: str | None = None

    persona_security: PersonaSecurity = PersonaSecurity.UNKNOWN

    persona_self_confidence_score: float | None = None
    persona_validation_seeking: bool | None = None
    persona_validation_tendency: float | None = None

    persona_text: str | None = None

    expected_ability_static: dict[str, float | None] = Field(
        default_factory=dict
    )
    expected_ability_static_std: dict[str, float | None] = Field(
        default_factory=dict
    )

    expected_ability_from_context_ground_truth: dict[
        str, float | None
    ] = Field(default_factory=dict)
    expected_ability_from_context_ground_truth_std: dict[
        str, float | None
    ] = Field(default_factory=dict)

    expected_ability_from_context_llm_estimated: dict[
        str, float | None
    ] = Field(default_factory=dict)
    expected_ability_from_context_llm_estimated_std: dict[
        str, float | None
    ] = Field(default_factory=dict)

    expected_ability_variance: float | dict[str, float | None] | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("persona_self_confidence_score", "persona_validation_tendency")
    @classmethod
    def validate_optional_unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("score must be in [0,1]")
        return value


# ============================================================
# UTTERANCE METADATA
# ============================================================


class UtteranceMetadata(FrozenMetadataModel):
    """
    Metadata about the target user utterance/question.
    """

    utterance_id: str

    utterance_source: str | None = None
    utterance_text: str | None = None

    domain: str | None = None
    capability: str | None = None
    subdomain: str | None = None

    difficulty_source: str | None = None

    difficulty: str | None = None
    difficulty_score: float | None = None
    difficulty_bin: int | None = None

    human_difficulty: float | None = None
    irt_difficulty: float | None = None

    ground_truth_value: float | None = None
    ground_truth_correctness: bool | None = None

    value_dimensions: dict[str, float | None] = Field(
        default_factory=dict
    )

    llm_rated_value: float | None = None
    llm_rated_correctness: bool | None = None

    llm_rated_value_dimensions: dict[str, float | None] = Field(
        default_factory=dict
    )

    belief_framing: BeliefFraming = BeliefFraming.NONE
    at_issue_status: AtIssueStatus = AtIssueStatus.UNKNOWN
    belief_owner: BeliefOwner = BeliefOwner.UNKNOWN

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ground_truth_value", "llm_rated_value")
    @classmethod
    def validate_optional_unit_interval(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("score must be in [0,1]")
        return value


# ============================================================
# CONTEXT METADATA
# ============================================================


class ContextMetadata(FrozenMetadataModel):
    """
    Conversation/context metadata.
    """

    context_id: str | None = None

    context_type: ContextType = ContextType.NONE

    context_text: str | None = None

    context_length_turns: int = 0
    context_length_tokens: int | None = None
    context_length_bucket: str | None = None

    context_question_count: int = 0

    context_difficulty_bins: list[int] = Field(default_factory=list)

    context_difficulty_mean: float | None = None
    context_difficulty_std: float | None = None

    context_user_accuracy: float | None = None
    context_model_accuracy: float | None = None

    context_user_correct_count: int | None = None
    context_user_incorrect_count: int | None = None

    context_domains: list[str] = Field(default_factory=list)
    context_capabilities: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# SURPRISAL METADATA
# ============================================================


class SurprisalMetadata(FrozenMetadataModel):
    """
    Implicit surprisal / token probability metadata.
    """

    surprisal_model: str | None = None

    answer_token_logprob_without_persona: float | None = None
    answer_token_logprob_with_persona: float | None = None

    answer_surprisal_without_persona: float | None = None
    answer_surprisal_with_persona: float | None = None

    persona_surprisal_delta: float | None = None
    response_surprisal: float | None = None
    response_surprisal_with_persona: float | None = None
    response_surprisal_without_persona: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# RESPONSE METADATA
# ============================================================


class ResponseMetadata(FrozenMetadataModel):
    """
    Metadata for a generated model response.
    """

    response_id: str
    instance_id: str

    model_name: str
    provider: str | None = None

    response_text: str | None = None

    response_length_tokens: int | None = None
    response_length_chars: int | None = None
    token_logprobs_available: bool = False
    logits_available: bool = False

    generation_config: dict[str, Any] = Field(default_factory=dict)

    surprisal: SurprisalMetadata = Field(
        default_factory=SurprisalMetadata
    )

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# JUDGE METADATA
# ============================================================


class JudgeMetadata(FrozenMetadataModel):
    """
    Metadata about the judge configuration/output.
    """

    judge_id: str | None = None

    judge_model: str | None = None
    judge_provider: str | None = None

    judge_setup_name: str | None = None

    exemplar_scale: int | None = None

    n_calls: int = 1
    aggregation: str | None = None

    latency_sec: float | None = None

    raw_outputs: list[dict[str, Any]] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# METRIC INPUTS
# ============================================================


class MetricInputs(FrozenMetadataModel):
    """
    Raw variables needed to recompute many
    sycophantic-praise metric formulations.
    """

    praise_count: int = 0

    praise_targets: dict[str, int] = Field(default_factory=dict)

    praise_magnitude_mean: float | None = None
    praise_magnitude_max: float | None = None
    praise_magnitude_sum: float | None = None

    person_praise_score: float | None = None
    process_praise_score: float | None = None
    outcome_praise_score: float | None = None

    target_weighted_praise: float | None = None

    utterance_value: float | None = None
    expected_value_given_persona: float | None = None

    expected_value_gap: float | None = None

    ground_truth_value: float | None = None
    llm_rated_value: float | None = None

    ground_truth_correctness: bool | None = None
    llm_rated_correctness: bool | None = None

    expected_ability: dict[str, float | None] = Field(
        default_factory=dict
    )

    llm_rated_user_ability: dict[str, float | None] = Field(
        default_factory=dict
    )
    surprisal: SurprisalMetadata = Field(default_factory=SurprisalMetadata)

    domain: str | None = None
    capability: str | None = None

    difficulty_score: float | None = None
    difficulty_bin: int | None = None

    risk_level: str | None = None

    overpraise_score: float | None = None
    miscalibrated_praise_score: float | None = None
    sycophantic_praise_score: float | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================
# MASTER ANALYSIS RECORD
# ============================================================


class AnalysisRecord(FrozenMetadataModel):
    """
    Durable unified analysis row.

    This should contain enough information to:
    - rerun metrics
    - produce plots
    - reproduce experiments
    - compare judge setups
    """

    record_id: str

    instance_id: str

    response_id: str | None = None

    sentence_id: str | None = None
    sentence_index: int | None = None

    sentence_text: str | None = None

    experiment: ExperimentMetadata

    persona: PersonaMetadata

    utterance: UtteranceMetadata

    context: ContextMetadata = Field(
        default_factory=ContextMetadata
    )

    response: ResponseMetadata | None = None

    judge: JudgeMetadata | None = None

    metric_inputs: MetricInputs = Field(
        default_factory=MetricInputs
    )

    metadata: dict[str, Any] = Field(default_factory=dict)
