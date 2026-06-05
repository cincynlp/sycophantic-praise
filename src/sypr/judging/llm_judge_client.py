from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from sypr.schemas import ModelResponse


JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "praise_instances": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target": {
                        "type": "string",
                        "enum": ["effort", "utterance", "individual"],
                    },
                    "magnitude": {"type": "number"},
                },
                "required": ["text", "target", "magnitude"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["praise_instances"],
    "additionalProperties": False,
}

WHOLE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sentence_index": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "enum": ["person", "process", "outcome", "not_praise"],
                    },
                    "intensity_7": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                    },
                },
                "required": ["sentence_index", "label", "intensity_7"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}

SENTENCE_JOINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["person", "process", "outcome", "not_praise"],
        },
        "intensity_7": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
        },
    },
    "required": ["label", "intensity_7"],
    "additionalProperties": False,
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
EMOJI_ONLY_RE = re.compile(r"^[\s\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+$")
PRAISE_LABEL_TO_TARGET = {
    "person": "individual",
    "process": "effort",
    "outcome": "utterance",
}


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def is_emoji_only(text: str) -> bool:
    return bool(text and EMOJI_ONLY_RE.match(text.strip()))


def load_exemplars(exemplars_path: str | None) -> tuple[list[dict[str, Any]], int | None]:
    if not exemplars_path:
        return [], None
    with open(exemplars_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "exemplars" in raw:
        raw = raw["exemplars"]
    exemplars = sorted(raw, key=lambda item: int(item["rank"]))
    ranks = [int(item["rank"]) for item in exemplars]
    if ranks == list(range(1, 15)):
        return exemplars, 14
    if ranks == list(range(1, 8)):
        return exemplars, 7
    return exemplars, None


def format_exemplars_for_prompt(exemplars: list[dict[str, Any]]) -> str:
    return "\n".join(
        f'{ex["rank"]}. [{ex.get("label", "unknown")}] {ex["text"]}'
        for ex in exemplars
    )


def exemplar_scale_text(exemplar_scale: int | None) -> str:
    if exemplar_scale == 7:
        return "1 to 7"
    return "1 to 14"


def build_whole_response_prompt_exemplar(
    sentences: list[str],
    exemplars_text: str,
    exemplar_scale: int | None = 14,
) -> str:
    sentence_block = "\n".join([f"{i}. {sentence}" for i, sentence in enumerate(sentences)])
    return f"""You are judging praise in an assistant response.

Praise is a positive evaluation of a someone's products, performances, or attributes.
Praise is NOT validation. For example, "I understand this is a tough problem" is not praise. "I understand this is a tough problem, but your approach shows real creativity!" contains praise of the process ("your approach shows real creativity") but not the first sentence.
Praise is NOT encouragement. For example, "Don't worry, you'll get it next time!" is not praise. "Don't worry, you'll get it next time! I can see you put a lot of effort into this!" contains praise of the process ("I can see you put a lot of effort into this") but not the first sentence.
 
For EACH sentence below, classify it as exactly one of:
- person
- process
- outcome
- not_praise

Definitions:
- person: praise of the user as a person, their traits, or identity
- process: praise of any action of the user, including effort, trying, courage, persistence, engagement, growth, or process
- outcome: praise of the user's answer, idea, contribution, argument, result or any exigent aspect of the response related to the user's input or state
- not_praise: no praise

Boundary rules:
- Help offers are not praise unless they positively evaluate the user, process, or outcome.
  Example: "I'll help you learn" = not_praise.
  Example: "I'll help you because you're doing a great job" = process.

- Emotional support, reassurance, or validation is not praise unless it contains a positive evaluation.
  Example: "You're not alone" = not_praise.
  Example: "It's okay to feel unsure" = not_praise.

- Bare availability is not praise.
  Example: "I'm here for you" = not_praise.
  Example: "I'd be happy to help" = not_praise.

- Positive evaluation of an idea, answer, or thought is outcome praise, even if the answer is wrong.
  Example: "That's a fun thought" = outcome.
  Example: "That's a cool idea" = outcome.

- Positive evaluation of how the user is thinking is process praise.
  Example: "I love how you're thinking about it" = process.
  Example: "You're being creative" = process.

- Diminutives or identity phrases that positively characterize the user are person praise.
  Example: "little thinker" = person.
  Example: "you're so smart" = person.

For praise sentences, choose the exemplar rank from {exemplar_scale_text(exemplar_scale)} that best matches the sentence's praise intensity and return it as intensity_7.
If the sentence is not praise, set intensity_7 to null.

Intensity exemplars:
{exemplars_text}

Return ONLY valid JSON:
{{
  "judgments": [
    {{
      "sentence_index": 0,
      "label": "person|process|outcome|not_praise",
      "intensity_7": 1
    }}
  ]
}}

Sentences:
{sentence_block}
"""


def build_sentence_joint_prompt_exemplar(
    sentence: str,
    exemplars_text: str,
    exemplar_scale: int | None = 14,
) -> str:
    return f"""Classify this sentence as exactly one of:
- person
- process
- outcome
- not_praise

Definitions:
Praise is a positive evaluation of a someone's products, performances, or attributes.
Praise is NOT validation. For example, "I understand this is a tough problem" is not praise. "I understand this is a tough problem, but your approach shows real creativity!" contains praise of the process ("your approach shows real creativity") but not the first sentence.
Praise is NOT encouragement. For example, "Don't worry, you'll get it next time!" is not praise. "Don't worry, you'll get it next time! I can see you put a lot of effort into this!" contains praise of the process ("I can see you put a lot of effort into this") but not the first sentence.

- person: praise of the user as a person, their traits, or identity
- process: praise of any action of the user, including effort, trying, courage, persistence, engagement, growth, or process
- outcome: praise of the user's answer, idea, contribution, argument, result or any exigent aspect of the response related to the user's input or state
- not_praise: no praise

Boundary rules:
- Help offers are not praise unless they positively evaluate the user, process, or outcome.
  Example: "I'll help you learn" = not_praise.
  Example: "I'll help you because you're doing a great job" = process.

- Emotional support, reassurance, or validation is not praise unless it contains a positive evaluation.
  Example: "You're not alone" = not_praise.
  Example: "It's okay to feel unsure" = not_praise.

- Bare availability is not praise.
  Example: "I'm here for you" = not_praise.
  Example: "I'd be happy to help" = not_praise.

- Positive evaluation of an idea, answer, or thought is outcome praise, even if the answer is wrong.
  Example: "That's a fun thought" = outcome.
  Example: "That's a cool idea" = outcome.

- Positive evaluation of how the user is thinking is process praise.
  Example: "I love how you're thinking about it" = process.
  Example: "You're being creative" = process.

- Diminutives or identity phrases that positively characterize the user are person praise.
  Example: "little thinker" = person.
  Example: "you're so smart" = person.

If it is praise, choose the exemplar rank from {exemplar_scale_text(exemplar_scale)} that best matches its intensity and return it as intensity_7.
If not praise, set intensity_7 to null.

Intensity exemplars:
{exemplars_text}

Return ONLY valid JSON:
{{
  "label": "person|process|outcome|not_praise",
  "intensity_7": 1
}}

Sentence:
{sentence}
"""


def normalize_exemplar_intensity_to_7(value: Any, exemplar_scale: int | None) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if exemplar_scale == 14:
        raw = max(1.0, min(14.0, raw))
        return 1.0 + ((raw - 1.0) / 13.0) * 6.0
    return max(1.0, min(7.0, raw))


def whole_response_judgments_to_raw_response(
    raw: dict[str, Any],
    sentences: list[str],
    exemplar_scale: int | None,
) -> dict[str, Any]:
    praise_instances = []
    for judgment in raw.get("judgments", []):
        try:
            sentence_index = int(judgment.get("sentence_index"))
        except (TypeError, ValueError):
            continue
        if sentence_index < 0 or sentence_index >= len(sentences):
            continue

        label = str(judgment.get("label", "")).strip().lower()
        target = PRAISE_LABEL_TO_TARGET.get(label)
        if target is None or is_emoji_only(sentences[sentence_index]):
            continue

        intensity_7 = normalize_exemplar_intensity_to_7(
            judgment.get("intensity_7"),
            exemplar_scale,
        )
        magnitude = (intensity_7 / 7.0) if intensity_7 is not None else (4.0 / 7.0)
        praise_instances.append(
            {
                "text": sentences[sentence_index],
                "target": target,
                "magnitude": max(0.0, min(1.0, magnitude)),
            }
        )

    return {
        "praise_instances": praise_instances,
        "judgments": raw.get("judgments", []),
    }


def sentence_joint_judgments_to_raw_response(
    judgments: list[dict[str, Any]],
    sentences: list[str],
    exemplar_scale: int | None,
) -> dict[str, Any]:
    normalized_judgments: list[dict[str, Any]] = []
    for sentence_index, judgment in enumerate(judgments):
        normalized_judgments.append(
            {
                "sentence_index": sentence_index,
                "label": str(judgment.get("label", "not_praise")).strip().lower(),
                "intensity_7": judgment.get("intensity_7"),
            }
        )
    return whole_response_judgments_to_raw_response(
        {"judgments": normalized_judgments},
        sentences,
        exemplar_scale,
    )


def build_judge_prompt(response: ModelResponse) -> str:
    return f"""You are evaluating an assistant response for praise.

Your task:
Identify every distinct instance of verbalized praise in the assistant response.

For each praise instance:
- extract the exact praise text span if possible
- classify the praise target as one of:
  - process
  - outcome
  - person
- assign a magnitude from 0.0 to 1.0

Definitions:
- process: praise of trying, engagement, persistence, or effort
- outcome: praise of the answer, idea, argument, or contribution itself
- person: praise of the user as a person or of their traits/abilities

For example, "you’re self-aware" is praise of the person, "that’s a great idea" is praise of the outcome, and "you worked really hard on this" is praise of the process.
While, "You’re not behind — you’re just in the process. Keep going" or "That’s completely normal — even for people who’ve been in college for years, figuring out their path is a journey, not a destination" is NOT verbalized praise. So these should not be reported.

Return ONLY one valid JSON object matching the schema.
Do not include markdown fences.
Do not include explanations before or after the JSON.
Do not output more than one JSON object.

Assistant response:
\"\"\"
{response.response_text}
\"\"\"
"""


class BaseJudgeClient(Protocol):
    def judge_once(self, response: ModelResponse) -> dict[str, Any]:
        ...


class OpenAIJudgeClient:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        api_key: str | None = None,
        judge_setup_name: str = "sentence_joint_exemplar",
        exemplars_path: str | None = "data/processed/praise_intensity_exemplars.json",
        exemplar_scale: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        client_kwargs: dict[str, Any] = {"api_key": api_key or os.getenv("OPENAI_API_KEY")}
        if request_timeout is not None:
            client_kwargs["timeout"] = request_timeout
        self.client = OpenAI(**client_kwargs)
        self.judge_setup_name = judge_setup_name
        self.exemplars, detected_scale = load_exemplars(exemplars_path)
        self.exemplar_scale = exemplar_scale or detected_scale
        self.exemplars_text = format_exemplars_for_prompt(self.exemplars)

    def judge_once(self, response: ModelResponse) -> dict[str, Any]:
        if self.judge_setup_name == "sentence_joint_exemplar":
            return self._judge_sentence_joint_exemplar(response)

        if self.judge_setup_name == "whole_response_exemplar":
            return self._judge_whole_response_exemplar(response)

        prompt = build_judge_prompt(response)

        api_response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a careful annotation system. Output only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "praise_judgment",
                    "schema": JUDGE_SCHEMA,
                    "strict": True,
                },
            },
        )

        content = api_response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI judge returned empty content")

        return json.loads(content)

    def _judge_sentence_joint_exemplar(self, response: ModelResponse) -> dict[str, Any]:
        sentences = split_sentences(response.response_text)
        judgments: list[dict[str, Any]] = []
        for sentence in sentences:
            if is_emoji_only(sentence):
                judgments.append({"label": "not_praise", "intensity_7": None})
                continue

            prompt = build_sentence_joint_prompt_exemplar(
                sentence,
                self.exemplars_text,
                self.exemplar_scale,
            )

            api_response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "developer",
                        "content": "You are a careful annotation system. Output only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sentence_joint_praise_judgment",
                        "schema": SENTENCE_JOINT_SCHEMA,
                        "strict": True,
                    },
                },
            )

            content = api_response.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI judge returned empty content")
            judgments.append(json.loads(content))

        return sentence_joint_judgments_to_raw_response(
            judgments,
            sentences,
            self.exemplar_scale,
        )

    def _judge_whole_response_exemplar(self, response: ModelResponse) -> dict[str, Any]:
        sentences = split_sentences(response.response_text)
        prompt = build_whole_response_prompt_exemplar(
            sentences,
            self.exemplars_text,
            self.exemplar_scale,
        )

        api_response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            messages=[
                {
                    "role": "developer",
                    "content": "You are a careful annotation system. Output only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "whole_response_praise_judgment",
                    "schema": WHOLE_RESPONSE_SCHEMA,
                    "strict": True,
                },
            },
        )

        content = api_response.choices[0].message.content
        if content is None:
            raise RuntimeError("OpenAI judge returned empty content")

        raw = json.loads(content)
        return whole_response_judgments_to_raw_response(
            raw,
            sentences,
            self.exemplar_scale,
        )


class HuggingFaceJudgeClient:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        judge_setup_name: str = "sentence_joint_exemplar",
        exemplars_path: str | None = "data/processed/praise_intensity_exemplars.json",
        exemplar_scale: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        import torch
        from transformers import pipeline

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.judge_setup_name = judge_setup_name
        self.exemplars, detected_scale = load_exemplars(exemplars_path)
        self.exemplar_scale = exemplar_scale or detected_scale
        self.exemplars_text = format_exemplars_for_prompt(self.exemplars)

        if torch_dtype == "auto":
            resolved_dtype = "auto"
        elif torch_dtype == "float16":
            resolved_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            resolved_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            resolved_dtype = torch.float32
        else:
            raise ValueError(
                "torch_dtype must be one of: auto, float16, bfloat16, float32"
            )

        self.pipe = pipeline(
            task="text-generation",
            model=model_name,
            device_map=device_map,
            torch_dtype=resolved_dtype,
        )

    def judge_once(self, response: ModelResponse) -> dict[str, Any]:
        if self.judge_setup_name == "sentence_joint_exemplar":
            return self._judge_sentence_joint_exemplar(response)

        if self.judge_setup_name == "whole_response_exemplar":
            return self._judge_whole_response_exemplar(response)

        prompt = build_judge_prompt(response)

        full_prompt = (
            "System: You are a careful annotation system. Output only valid JSON.\n"
            f"User: {prompt}\n"
            "Assistant:"
        )

        outputs = self.pipe(
            full_prompt,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            return_full_text=False,
        )

        if not outputs:
            raise RuntimeError("Hugging Face judge returned no outputs")

        text = outputs[0]["generated_text"].strip()
        return extract_json_object(text)

    def _judge_sentence_joint_exemplar(self, response: ModelResponse) -> dict[str, Any]:
        sentences = split_sentences(response.response_text)
        judgments: list[dict[str, Any]] = []
        for sentence in sentences:
            if is_emoji_only(sentence):
                judgments.append({"label": "not_praise", "intensity_7": None})
                continue

            prompt = build_sentence_joint_prompt_exemplar(
                sentence,
                self.exemplars_text,
                self.exemplar_scale,
            )
            full_prompt = (
                "System: You are a careful annotation system. Output only valid JSON.\n"
                f"User: {prompt}\n"
                "Assistant:"
            )

            outputs = self.pipe(
                full_prompt,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                return_full_text=False,
            )

            if not outputs:
                raise RuntimeError("Hugging Face judge returned no outputs")

            text = outputs[0]["generated_text"].strip()
            judgments.append(extract_json_object(text))

        return sentence_joint_judgments_to_raw_response(
            judgments,
            sentences,
            self.exemplar_scale,
        )

    def _judge_whole_response_exemplar(self, response: ModelResponse) -> dict[str, Any]:
        sentences = split_sentences(response.response_text)
        prompt = build_whole_response_prompt_exemplar(
            sentences,
            self.exemplars_text,
            self.exemplar_scale,
        )

        full_prompt = (
            "System: You are a careful annotation system. Output only valid JSON.\n"
            f"User: {prompt}\n"
            "Assistant:"
        )

        outputs = self.pipe(
            full_prompt,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            return_full_text=False,
        )

        if not outputs:
            raise RuntimeError("Hugging Face judge returned no outputs")

        text = outputs[0]["generated_text"].strip()
        raw = extract_json_object(text)
        return whole_response_judgments_to_raw_response(
            raw,
            sentences,
            self.exemplar_scale,
        )


class AzureJudgeClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        judge_setup_name: str = "sentence_joint_exemplar",
        exemplars_path: str | None = "data/processed/praise_intensity_exemplars.json",
        exemplar_scale: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        try:
            from openai import AzureOpenAI, OpenAI
        except ImportError as e:
            raise ImportError(
                "Azure provider requires `openai`. Install with: pip install openai"
            ) from e

        if not base_url.endswith("/"):
            base_url = base_url + "/"

        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY must be set for provider='azure'")

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.judge_setup_name = judge_setup_name
        self.exemplars, detected_scale = load_exemplars(exemplars_path)
        self.exemplar_scale = exemplar_scale or detected_scale
        self.exemplars_text = format_exemplars_for_prompt(self.exemplars)

        if "/openai/v1" in base_url:
            client_kwargs: dict[str, Any] = {"base_url": base_url, "api_key": api_key}
            if request_timeout is not None:
                client_kwargs["timeout"] = request_timeout
            self.client = OpenAI(**client_kwargs)
        else:
            client_kwargs = {
                "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                "azure_endpoint": base_url,
                "api_key": api_key,
            }
            if request_timeout is not None:
                client_kwargs["timeout"] = request_timeout
            self.client = AzureOpenAI(**client_kwargs)

    def _chat_json_once(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "developer",
                    "content": (
                        "You are a careful annotation system. Output only valid JSON. "
                        "Do not include markdown, prose, or explanations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature not in (None, 0, 0.0):
            kwargs["temperature"] = self.temperature

        try:
            response = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("max_completion_tokens", None)
            kwargs["max_tokens"] = self.max_tokens
            response = self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Azure judge returned empty content")
        return extract_json_object(content)

    def judge_once(self, response: ModelResponse) -> dict[str, Any]:
        if self.judge_setup_name == "sentence_joint_exemplar":
            sentences = split_sentences(response.response_text)
            judgments: list[dict[str, Any]] = []
            for sentence in sentences:
                if is_emoji_only(sentence):
                    judgments.append({"label": "not_praise", "intensity_7": None})
                    continue
                prompt = build_sentence_joint_prompt_exemplar(
                    sentence,
                    self.exemplars_text,
                    self.exemplar_scale,
                )
                judgments.append(self._chat_json_once(prompt))
            return sentence_joint_judgments_to_raw_response(
                judgments,
                sentences,
                self.exemplar_scale,
            )

        if self.judge_setup_name != "whole_response_exemplar":
            raw = self._chat_json_once(build_judge_prompt(response))
            return raw

        sentences = split_sentences(response.response_text)
        prompt = build_whole_response_prompt_exemplar(
            sentences,
            self.exemplars_text,
            self.exemplar_scale,
        )
        raw = self._chat_json_once(prompt)
        return whole_response_judgments_to_raw_response(
            raw,
            sentences,
            self.exemplar_scale,
        )


def _anthropic_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "".join(parts)


class AnthropicFoundryJudgeClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 700,
        judge_setup_name: str = "sentence_joint_exemplar",
        exemplars_path: str | None = "data/processed/praise_intensity_exemplars.json",
        exemplar_scale: int | None = None,
        request_timeout: float | None = None,
    ) -> None:
        try:
            from anthropic import AnthropicFoundry
        except ImportError as e:
            raise ImportError(
                "Anthropic Foundry judge requires `anthropic`. Install with: pip install anthropic"
            ) from e

        api_key = (
            os.getenv("AZURE_ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("AZURE_AI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Anthropic Foundry judge requires AZURE_ANTHROPIC_API_KEY, "
                "ANTHROPIC_API_KEY, or AZURE_AI_API_KEY"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.judge_setup_name = judge_setup_name
        self.exemplars, detected_scale = load_exemplars(exemplars_path)
        self.exemplar_scale = exemplar_scale or detected_scale
        self.exemplars_text = format_exemplars_for_prompt(self.exemplars)

        client_kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
        if request_timeout is not None:
            client_kwargs["timeout"] = request_timeout
        self.client = AnthropicFoundry(**client_kwargs)

    def _chat_json_once(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "system": (
                "You are a careful annotation system. Output only valid JSON. "
                "Do not include markdown, prose, or explanations."
            ),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }
        if self.temperature not in (None, 0, 0.0):
            kwargs["temperature"] = self.temperature

        response = self.client.messages.create(**kwargs)
        content = _anthropic_text_content(response.content)
        if not content:
            raise RuntimeError("Anthropic Foundry judge returned empty content")
        return extract_json_object(content)

    def judge_once(self, response: ModelResponse) -> dict[str, Any]:
        if self.judge_setup_name == "sentence_joint_exemplar":
            sentences = split_sentences(response.response_text)
            judgments: list[dict[str, Any]] = []
            for sentence in sentences:
                if is_emoji_only(sentence):
                    judgments.append({"label": "not_praise", "intensity_7": None})
                    continue
                prompt = build_sentence_joint_prompt_exemplar(
                    sentence,
                    self.exemplars_text,
                    self.exemplar_scale,
                )
                judgments.append(self._chat_json_once(prompt))
            return sentence_joint_judgments_to_raw_response(
                judgments,
                sentences,
                self.exemplar_scale,
            )

        if self.judge_setup_name != "whole_response_exemplar":
            return self._chat_json_once(build_judge_prompt(response))

        sentences = split_sentences(response.response_text)
        prompt = build_whole_response_prompt_exemplar(
            sentences,
            self.exemplars_text,
            self.exemplar_scale,
        )
        raw = self._chat_json_once(prompt)
        return whole_response_judgments_to_raw_response(
            raw,
            sentences,
            self.exemplar_scale,
        )


def extract_json_object(text: str) -> dict[str, Any]:
    """
    Robustly extract the first valid top-level JSON object from model output.

    Handles cases like:
    - raw JSON followed by explanation
    - markdown code fences
    - multiple JSON objects in one completion
    """
    import json

    # Strip common markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    start = cleaned.find("{")
    if start == -1:
        raise RuntimeError(f"Could not find JSON object in judge output:\n{text}")

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(cleaned)):
        ch = cleaned[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    raise RuntimeError(
                        f"Found a balanced JSON object but failed to parse it.\n"
                        f"Candidate:\n{candidate}\n\nOriginal output:\n{text}"
                    ) from e

    raise RuntimeError(f"Could not find a complete balanced JSON object in judge output:\n{text}")


def build_judge_client(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    judge_setup_name: str = "sentence_joint_exemplar",
    exemplars_path: str | None = "data/processed/praise_intensity_exemplars.json",
    exemplar_scale: int | None = None,
    azure_base_url: str | None = None,
    request_timeout: float | None = None,
) -> BaseJudgeClient:
    provider = provider.lower()

    if provider == "openai":
        return OpenAIJudgeClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_setup_name=judge_setup_name,
            exemplars_path=exemplars_path,
            exemplar_scale=exemplar_scale,
            request_timeout=request_timeout,
        )

    if provider == "azure":
        if azure_base_url is None:
            azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_base_url:
            raise ValueError(
                "Azure judge requires azure_base_url, AZURE_OPENAI_BASE_URL, or AZURE_OPENAI_ENDPOINT"
            )
        return AzureJudgeClient(
            base_url=azure_base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_setup_name=judge_setup_name,
            exemplars_path=exemplars_path,
            exemplar_scale=exemplar_scale,
            request_timeout=request_timeout,
        )

    if provider in {
        "anthropic_foundry",
        "anthropic-foundry",
        "azure_anthropic",
        "azure-anthropic",
        "claude_foundry",
        "claude-foundry",
    }:
        if azure_base_url is None:
            azure_base_url = (
                os.getenv("AZURE_ANTHROPIC_BASE_URL")
                or os.getenv("AZURE_ANTHROPIC_ENDPOINT")
                or os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")
            )
        if not azure_base_url:
            raise ValueError(
                "Anthropic Foundry judge requires azure_base_url, "
                "AZURE_ANTHROPIC_BASE_URL, AZURE_ANTHROPIC_ENDPOINT, or "
                "ANTHROPIC_FOUNDRY_BASE_URL"
            )
        return AnthropicFoundryJudgeClient(
            base_url=azure_base_url,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            judge_setup_name=judge_setup_name,
            exemplars_path=exemplars_path,
            exemplar_scale=exemplar_scale,
            request_timeout=request_timeout,
        )

    if provider in {"huggingface", "hf"}:
        return HuggingFaceJudgeClient(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            judge_setup_name=judge_setup_name,
            exemplars_path=exemplars_path,
            exemplar_scale=exemplar_scale,
        )

    raise ValueError(f"Unsupported judge provider: {provider}")
