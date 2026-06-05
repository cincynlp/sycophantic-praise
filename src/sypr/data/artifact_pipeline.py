from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

from sypr.data.generate import (
    generate_response_for_instance_with_retries,
    model_response_from_text,
)
from sypr.data.io import read_jsonl, write_jsonl
from sypr.data.model_client import build_model_client
from sypr.schemas import BenchmarkArtifact, BenchmarkInstance, ModelResponse


def artifact_to_instance(artifact: BenchmarkArtifact) -> BenchmarkInstance:
    return BenchmarkInstance(
        instance_id=artifact.artifact_id,
        instance_metadata=artifact.instance_metadata,
        persona=artifact.persona,
        utterance=artifact.utterance,
        prompt_condition=artifact.instance_metadata.prompt_condition,
        conversation_history=artifact.context,
        context_metadata=artifact.context_metadata,
        metadata=artifact.metadata,
    )


def response_exists(
    artifact: BenchmarkArtifact,
    *,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> bool:
    for response in artifact.responses:
        metadata = response.metadata
        config = response.generation_config
        if (
            response.model_name == model_name
            and metadata.get("provider") == provider
            and config.temperature == temperature
            and config.max_tokens == max_tokens
        ):
            return True
    return False


def generate_responses_for_artifacts(
    artifacts: list[BenchmarkArtifact],
    *,
    provider: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    azure_base_url: str | None = None,
    max_workers: int = 1,
    batch_size: int = 1,
    retry_attempts: int = 4,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 30.0,
    runtime_save_path: str | Path | None = None,
    skip_existing: bool = True,
) -> list[BenchmarkArtifact]:
    existing_by_artifact_id = read_existing_artifacts_by_id(runtime_save_path) if runtime_save_path else {}
    written_artifact_ids = set(existing_by_artifact_id)
    write_lock = threading.Lock()
    max_workers = max(1, max_workers)
    batch_size = max(1, batch_size)

    generated: list[BenchmarkArtifact | None] = [None] * len(artifacts)
    to_generate: list[tuple[int, BenchmarkArtifact]] = []
    for idx, artifact in enumerate(artifacts):
        existing = existing_by_artifact_id.get(artifact.artifact_id)
        if skip_existing and existing is not None and response_exists(
            existing,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            generated[idx] = existing
        elif skip_existing and response_exists(
            artifact,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            generated[idx] = artifact
        else:
            to_generate.append((idx, artifact))

    if not to_generate:
        return [artifact for artifact in generated if artifact is not None]

    client = build_model_client(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        azure_base_url=azure_base_url,
    )

    def record_artifact(artifact: BenchmarkArtifact) -> None:
        if runtime_save_path is None:
            return
        with write_lock:
            if artifact.artifact_id in written_artifact_ids:
                return
            append_artifact_jsonl(runtime_save_path, artifact)
            written_artifact_ids.add(artifact.artifact_id)

    def generate_one(idx: int, artifact: BenchmarkArtifact) -> tuple[int, BenchmarkArtifact]:
        instance = artifact_to_instance(artifact)
        response = generate_response_for_instance_with_retries(
            instance=instance,
            client=client,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            response_index=len(artifact.responses),
            retry_attempts=retry_attempts,
            retry_initial_delay=retry_initial_delay,
            retry_max_delay=retry_max_delay,
        )
        updated = artifact.model_copy(update={"responses": [*artifact.responses, response]})
        record_artifact(updated)
        return idx, updated

    def generate_batch(batch: list[tuple[int, BenchmarkArtifact]]) -> list[tuple[int, BenchmarkArtifact]]:
        if provider not in {"huggingface", "hf"} or len(batch) == 1:
            return [generate_one(idx, artifact) for idx, artifact in batch]

        instances = [artifact_to_instance(artifact) for _, artifact in batch]
        if not hasattr(client, "generate_batch"):
            return [generate_one(idx, artifact) for idx, artifact in batch]

        texts = client.generate_batch(instances)
        results = []
        for (idx, artifact), instance, text in zip(batch, instances, texts):
            response = model_response_from_text(
                instance=instance,
                response_text=text,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                device_map=device_map,
                torch_dtype=torch_dtype,
                response_index=len(artifact.responses),
            )
            updated = artifact.model_copy(update={"responses": [*artifact.responses, response]})
            record_artifact(updated)
            results.append((idx, updated))
        return results

    batches = [to_generate[i : i + batch_size] for i in range(0, len(to_generate), batch_size)]
    if max_workers == 1:
        for batch in batches:
            for idx, artifact in generate_batch(batch):
                generated[idx] = artifact
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(generate_batch, batch) for batch in batches]
            for future in concurrent.futures.as_completed(futures):
                for idx, artifact in future.result():
                    generated[idx] = artifact

    return [artifact for artifact in generated if artifact is not None]


def flatten_artifact_responses(artifacts: list[BenchmarkArtifact]) -> list[ModelResponse]:
    responses: list[ModelResponse] = []
    for artifact in artifacts:
        responses.extend(artifact.responses)
    return responses


def read_artifacts(path: str | Path) -> list[BenchmarkArtifact]:
    return [BenchmarkArtifact(**row) for row in read_jsonl(path)]


def write_artifacts(path: str | Path, artifacts: list[BenchmarkArtifact]) -> None:
    write_jsonl(path, [artifact.model_dump(mode="json") for artifact in artifacts])


def append_artifact_jsonl(path: str | Path, artifact: BenchmarkArtifact) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False) + "\n")
        f.flush()


def read_existing_artifacts_by_id(path: str | Path | None) -> dict[str, BenchmarkArtifact]:
    if path is None:
        return {}
    output = Path(path)
    if not output.exists():
        return {}

    artifacts: dict[str, BenchmarkArtifact] = {}
    with output.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                artifact = BenchmarkArtifact(**json.loads(stripped))
            except Exception:
                continue
            artifacts[artifact.artifact_id] = artifact
    return artifacts
