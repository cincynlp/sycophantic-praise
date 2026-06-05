from __future__ import annotations

import concurrent.futures
import json
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from sypr.data.model_client import BaseModelClient, build_model_client
from sypr.data.build_benchmark import stable_uuid
from sypr.schemas import BenchmarkInstance, GenerationConfig, ModelResponse, ResponseMetadata

T = TypeVar("T")


def _sleep_before_retry(
    *,
    attempt_idx: int,
    retry_initial_delay: float,
    retry_max_delay: float,
) -> None:
    delay = min(retry_max_delay, retry_initial_delay * (2 ** max(0, attempt_idx - 1)))
    jitter = random.uniform(0.0, delay * 0.1) if delay > 0 else 0.0
    time.sleep(delay + jitter)


def call_with_retries(
    fn: Callable[[], T],
    *,
    retry_attempts: int = 4,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
) -> T:
    retry_attempts = max(1, retry_attempts)
    last_error: Exception | None = None

    for attempt_idx in range(1, retry_attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt_idx >= retry_attempts:
                break
            _sleep_before_retry(
                attempt_idx=attempt_idx,
                retry_initial_delay=retry_initial_delay,
                retry_max_delay=retry_max_delay,
            )

    assert last_error is not None
    raise last_error


def generation_config(
    *,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    device_map: str = "auto",
    torch_dtype: str = "auto",
) -> GenerationConfig:
    config = GenerationConfig(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        metadata={
            "provider": provider,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
        },
    )
    return config


def generate_response_for_instance(
    *,
    instance: BenchmarkInstance,
    client: BaseModelClient,
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    response_index: int = 0,
) -> ModelResponse:
    response_text = client.generate(instance)
    return model_response_from_text(
        instance=instance,
        response_text=response_text,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        response_index=response_index,
    )


def model_response_from_text(
    *,
    instance: BenchmarkInstance,
    response_text: str,
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    response_index: int = 0,
) -> ModelResponse:
    config = generation_config(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    response_id = stable_uuid(
        "response",
        {
            "instance_id": instance.instance_id,
            "model_name": model_name,
            "provider": provider,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_index": response_index,
        },
    )
    return ModelResponse(
        instance_id=instance.instance_id,
        response_id=response_id,
        model_name=model_name,
        response_text=response_text,
        generation_config=config,
        response_metadata=ResponseMetadata(
            response_id=response_id,
            instance_id=instance.instance_id,
            model_name=model_name,
            provider=provider,
            response_text=response_text,
            response_length_chars=len(response_text),
            response_length_tokens=len(response_text.split()),
            generation_config=config.model_dump(mode="json"),
            metadata={"response_index": response_index},
        ),
        metadata={
            "provider": provider,
            "response_index": response_index,
            "context_length_bucket": instance.context_metadata.context_length_bucket,
        },
    )


def generate_response_for_instance_with_retries(
    *,
    retry_attempts: int = 4,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    **kwargs,
) -> ModelResponse:
    return call_with_retries(
        lambda: generate_response_for_instance(**kwargs),
        retry_attempts=retry_attempts,
        retry_initial_delay=retry_initial_delay,
        retry_max_delay=retry_max_delay,
    )


def generate_responses(
    instances: list[BenchmarkInstance],
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    azure_base_url: str | None = None,
    max_workers: int = 1,
    retry_attempts: int = 4,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    runtime_save_path: str | Path | None = None,
) -> list[ModelResponse]:
    existing_by_instance_id: dict[str, ModelResponse] = {}
    written_instance_ids: set[str] = set()
    write_lock = threading.Lock()
    if runtime_save_path is not None:
        for response in read_existing_responses(runtime_save_path):
            existing_by_instance_id[response.instance_id] = response
        written_instance_ids = set(existing_by_instance_id)

    max_workers = max(1, max_workers)

    responses: list[ModelResponse | None] = [None] * len(instances)
    to_generate: list[tuple[int, BenchmarkInstance]] = []
    for idx, instance in enumerate(instances):
        existing = existing_by_instance_id.get(instance.instance_id)
        if existing is not None:
            responses[idx] = existing
        else:
            to_generate.append((idx, instance))

    if not to_generate:
        return [response for response in responses if response is not None]

    client = build_model_client(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        azure_base_url=azure_base_url,
    )

    def record_response(response: ModelResponse) -> None:
        if runtime_save_path is None:
            return
        with write_lock:
            if response.instance_id in written_instance_ids:
                return
            append_response_jsonl(runtime_save_path, response)
            written_instance_ids.add(response.instance_id)

    def generate_one(idx: int, instance: BenchmarkInstance) -> tuple[int, ModelResponse]:
        response = generate_response_for_instance_with_retries(
            instance=instance,
            client=client,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            response_index=idx,
            retry_attempts=retry_attempts,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
        )
        record_response(response)
        return idx, response

    if max_workers == 1:
        for idx, instance in to_generate:
            result_idx, response = generate_one(idx, instance)
            responses[result_idx] = response
        return [response for response in responses if response is not None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_one, idx, instance): idx
            for idx, instance in to_generate
        }
        for future in concurrent.futures.as_completed(futures):
            result_idx, response = future.result()
            responses[result_idx] = response

    return [response for response in responses if response is not None]


def read_existing_response_ids(path: str | Path) -> set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()

    ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = row.get("instance_id")
            if isinstance(instance_id, str) and instance_id:
                ids.add(instance_id)
    return ids


def read_existing_responses(path: str | Path) -> list[ModelResponse]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    responses = []
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                responses.append(ModelResponse(**json.loads(line)))
    return responses


def append_response_jsonl(path: str | Path, response: ModelResponse) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(response.model_dump(mode="json"), ensure_ascii=False) + "\n")
        f.flush()
