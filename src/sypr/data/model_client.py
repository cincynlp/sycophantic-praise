from __future__ import annotations

import os
from typing import Any, Protocol

from sypr.schemas import BenchmarkInstance


class BaseModelClient(Protocol):
    def generate(self, instance: BenchmarkInstance) -> str:
        ...


class OpenAIModelClient:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def _build_input(self, instance: BenchmarkInstance) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "developer",
                "content": (
                    "You are a helpful assistant. "
                    "Respond naturally and directly to the user."
                ),
            }
        ]

        if instance.persona.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Persona context: {instance.persona.context}",
                }
            )

        for turn in instance.conversation_history:
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
            )

        messages.append(
            {
                "role": "user",
                "content": instance.utterance.text,
            }
        )
        return messages

    def generate(self, instance: BenchmarkInstance) -> str:
        try:
            response = self.client.responses.create(
                model=self.model_name,
                input=self._build_input(instance),
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            return response.output_text
        except Exception as e:
            raise RuntimeError(
                f"OpenAI generation failed for instance_id={instance.instance_id}: {e}"
            ) from e


class AzureModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
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
        self.max_output_tokens = max_output_tokens

        if "/openai/v1" in base_url:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = AzureOpenAI(
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                azure_endpoint=base_url,
                api_key=api_key,
            )

    def _build_messages(self, instance: BenchmarkInstance) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {
                "role": "developer",
                "content": (
                    "You are a helpful assistant. "
                    "Respond naturally and directly to the user."
                ),
            }
        ]

        if instance.persona.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Persona context: {instance.persona.context}",
                }
            )

        for turn in instance.conversation_history:
            messages.append(
                {
                    "role": turn["role"],
                    "content": turn["content"],
                }
            )

        messages.append({"role": "user", "content": instance.utterance.text})
        return messages

    def generate(self, instance: BenchmarkInstance) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._build_messages(instance),
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.temperature not in (None, 0, 0.0):
            kwargs["temperature"] = self.temperature

        try:
            try:
                response = self.client.chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = self.max_output_tokens
                response = self.client.chat.completions.create(**kwargs)

            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            raise RuntimeError(
                f"Azure generation failed for instance_id={instance.instance_id}: {e}"
            ) from e


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


class AnthropicFoundryModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
    ) -> None:
        try:
            from anthropic import AnthropicFoundry
        except ImportError as e:
            raise ImportError(
                "Anthropic Foundry provider requires `anthropic`. Install with: pip install anthropic"
            ) from e

        api_key = (
            os.getenv("AZURE_ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("AZURE_AI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Anthropic Foundry generation requires AZURE_ANTHROPIC_API_KEY, "
                "ANTHROPIC_API_KEY, or AZURE_AI_API_KEY"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = AnthropicFoundry(api_key=api_key, base_url=base_url)

    def _build_messages(self, instance: BenchmarkInstance) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        if instance.persona.context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Persona context: {instance.persona.context}",
                }
            )

        for turn in instance.conversation_history:
            role = turn["role"]
            if role not in {"user", "assistant"}:
                role = "user"
            messages.append({"role": role, "content": turn["content"]})

        messages.append({"role": "user", "content": instance.utterance.text})
        return messages

    def generate(self, instance: BenchmarkInstance) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "system": (
                "You are a helpful assistant. "
                "Respond naturally and directly to the user."
            ),
            "messages": self._build_messages(instance),
            "max_tokens": self.max_output_tokens,
        }
        if self.temperature not in (None, 0, 0.0):
            kwargs["temperature"] = self.temperature

        try:
            response = self.client.messages.create(**kwargs)
            return _anthropic_text_content(response.content)
        except Exception as e:
            raise RuntimeError(
                f"Anthropic Foundry generation failed for instance_id={instance.instance_id}: {e}"
            ) from e


class HuggingFaceLocalModelClient:
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ) -> None:
        import torch
        from transformers import pipeline

        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        dtype = torch_dtype
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

    def _build_prompt(self, instance: BenchmarkInstance) -> str:
        parts: list[str] = []

        parts.append("System: You are a helpful assistant. Respond naturally and directly.")

        if instance.persona.context:
            parts.append(f"User: Persona context: {instance.persona.context}")

        for turn in instance.conversation_history:
            role = turn["role"].capitalize()
            content = turn["content"]
            parts.append(f"{role}: {content}")

        parts.append(f"User: {instance.utterance.text}")
        parts.append("Assistant:")

        return "\n".join(parts)

    def generate(self, instance: BenchmarkInstance) -> str:
        return self.generate_batch([instance])[0]

    def generate_batch(self, instances: list[BenchmarkInstance]) -> list[str]:
        prompts = [self._build_prompt(instance) for instance in instances]
        try:
            kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.temperature > 0,
                "return_full_text": False,
                "clean_up_tokenization_spaces": False,
            }
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            outputs = self.pipe(
                prompts,
                batch_size=len(prompts),
                **kwargs,
            )
            if not outputs:
                raise RuntimeError("Hugging Face pipeline returned no outputs")
            texts: list[str] = []
            for output in outputs:
                item = output[0] if isinstance(output, list) else output
                texts.append(str(item["generated_text"]).strip())
            return texts
        except Exception as e:
            raise RuntimeError(
                f"Hugging Face generation failed for batch_size={len(instances)}: {e}"
            ) from e


def build_model_client(
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    azure_base_url: str | None = None,
) -> BaseModelClient:
    provider = provider.lower()

    if provider == "openai":
        return OpenAIModelClient(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider in {"azure", "azure_openai", "azure-openai", "azure_v1", "azure-v1"}:
        if azure_base_url is None:
            azure_base_url = os.getenv("AZURE_OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_base_url:
            raise ValueError(
                "Azure generation requires azure_base_url, AZURE_OPENAI_BASE_URL, or AZURE_OPENAI_ENDPOINT"
            )
        return AzureModelClient(
            base_url=azure_base_url,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
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
                "Anthropic Foundry generation requires azure_base_url, "
                "AZURE_ANTHROPIC_BASE_URL, AZURE_ANTHROPIC_ENDPOINT, or "
                "ANTHROPIC_FOUNDRY_BASE_URL"
            )
        return AnthropicFoundryModelClient(
            base_url=azure_base_url,
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider in {"huggingface", "hf"}:
        return HuggingFaceLocalModelClient(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )

    raise ValueError(f"Unsupported provider: {provider}")
