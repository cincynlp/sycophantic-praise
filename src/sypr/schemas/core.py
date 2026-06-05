from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sypr.schemas.metadata import (
    ContextMetadata,
    ExperimentMetadata,
    JudgeMetadata,
    PersonaMetadata,
    ResponseMetadata,
    UtteranceMetadata,
)


class FrozenSchemaModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PraiseTarget(str, Enum):
    EFFORT = "effort"
    UTTERANCE = "utterance"
    INDIVIDUAL = "individual"


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


class Persona(FrozenSchemaModel):
    persona_id: str
    context: str
    expected_abilities: dict[str, float]
    persona_metadata: PersonaMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expected_abilities")
    @classmethod
    def validate_expected_abilities(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("expected_abilities must not be empty")
        for key, score in value.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"expected_abilities[{key}] must be in [0,1]")
        return value


class Utterance(FrozenSchemaModel):
    utterance_id: str
    text: str
    value: dict[str, float]
    relevance: dict[str, float]
    utterance_metadata: UtteranceMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("value must not be empty")
        for key, score in value.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"value[{key}] must be in [0,1]")
        return value

    @field_validator("relevance")
    @classmethod
    def validate_relevance(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("relevance must not be empty")
        if sum(value.values()) <= 0:
            raise ValueError("relevance weights must sum to > 0")
        for key, score in value.items():
            if score < 0.0:
                raise ValueError(f"relevance[{key}] must be >= 0")
        return value

    @model_validator(mode="after")
    def validate_dimensions(self) -> "Utterance":
        missing = set(self.value) - set(self.relevance)
        if missing:
            raise ValueError(f"Missing relevance values for dimensions: {sorted(missing)}")
        return self


class BenchmarkInstance(FrozenSchemaModel):
    instance_id: str
    instance_metadata: ExperimentMetadata = Field(default_factory=ExperimentMetadata)
    persona: Persona
    utterance: Utterance
    prompt_condition: PromptCondition = PromptCondition.UNPROMPTED
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    context_metadata: ContextMetadata = Field(default_factory=ContextMetadata)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("conversation_history")
    @classmethod
    def validate_history(cls, value: list[dict[str, str]]) -> list[dict[str, str]]:
        for turn in value:
            if "role" not in turn or "content" not in turn:
                raise ValueError("conversation_history turns need 'role' and 'content'")
        return value


class BenchmarkArtifact(FrozenSchemaModel):
    artifact_id: str
    instance_metadata: ExperimentMetadata
    persona: Persona
    utterance: Utterance
    context: list[dict[str, str]] = Field(default_factory=list)
    context_metadata: ContextMetadata = Field(default_factory=ContextMetadata)
    responses: list["ModelResponse"] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationConfig(FrozenSchemaModel):
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 512
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(FrozenSchemaModel):
    instance_id: str
    response_id: str
    model_name: str
    response_text: str
    generation_config: GenerationConfig
    response_metadata: ResponseMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PraiseInstance(FrozenSchemaModel):
    text: str
    target: PraiseTarget
    magnitude: float
    start_char: int | None = None
    end_char: int | None = None
    confidence: float | None = None

    @field_validator("magnitude")
    @classmethod
    def validate_magnitude(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("magnitude must be in [0,1]")
        return value

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        return value

class JudgeConfig(FrozenSchemaModel):
    provider: Literal[
        "huggingface",
        "hf",
        "openai",
        "azure",
        "anthropic_foundry",
        "anthropic-foundry",
        "azure_anthropic",
        "azure-anthropic",
        "claude_foundry",
        "claude-foundry",
    ] = "huggingface"
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    temperature: float = 0.0
    max_tokens: int = 700
    n_calls: int = 1
    aggregation: Literal["none", "union_mean", "union_max"] = "none"
    judge_setup_name: str = "sentence_joint_exemplar"
    exemplars_path: str | None = "data/processed/praise_intensity_exemplars.json"
    exemplar_scale: int | None = None
    azure_base_url: str | None = None
    request_timeout: float | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)

class RawJudgePraiseInstance(FrozenSchemaModel):
    text: str
    target: Literal["effort", "utterance", "individual"]
    magnitude: float

    @field_validator("magnitude")
    @classmethod
    def validate_magnitude(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("magnitude must be in [0,1]")
        return value


class RawJudgeResponse(FrozenSchemaModel):
    praise_instances: list[RawJudgePraiseInstance] = Field(default_factory=list)

class JudgeOutput(FrozenSchemaModel):
    instance_id: str
    response_id: str
    judge_model: str
    praise_instances: list[PraiseInstance] = Field(default_factory=list)
    raw_judge_text: str | None = None
    judge_metadata: JudgeMetadata | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TargetRegime(FrozenSchemaModel):
    alpha: float
    beta_0: float
    beta_V: float
    beta_delta: float
    penalty_weight: float

    @field_validator("alpha", "penalty_weight")
    @classmethod
    def validate_nonnegative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("alpha and penalty_weight must be non-negative")
        return value


class MetricRegime(FrozenSchemaModel):
    name: str
    effort: TargetRegime
    utterance: TargetRegime
    individual: TargetRegime
    stakes_multiplier: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def for_target(self, target: PraiseTarget) -> TargetRegime:
        if target == PraiseTarget.EFFORT:
            return self.effort
        if target == PraiseTarget.UTTERANCE:
            return self.utterance
        if target == PraiseTarget.INDIVIDUAL:
            return self.individual
        raise ValueError(f"Unknown target: {target}")


class PraiseTotals(FrozenSchemaModel):
    effort: float = 0.0
    utterance: float = 0.0
    individual: float = 0.0


class MetricSubscores(FrozenSchemaModel):
    observed: PraiseTotals
    warranted: PraiseTotals
    excess: PraiseTotals


class ScoreOutput(FrozenSchemaModel):
    instance_id: str
    response_id: str
    regime_name: str
    actual_value: float
    expected_value: float
    delta: float
    subscores: MetricSubscores
    sypr_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
