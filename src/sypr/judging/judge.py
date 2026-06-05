from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

from sypr.judging.aggregate import (
    aggregate_judge_calls,
    build_judge_output,
    normalize_raw_judge_response,
    raw_to_praise_instances,
)
from sypr.judging.llm_judge_client import build_judge_client
from sypr.data.io import read_jsonl
from sypr.schemas import JudgeConfig, JudgeOutput, ModelResponse

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime.
    tqdm = None


def _progress(iterable, *, desc: str, total: int | None = None):
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total)


def _judge_output_matches_config(output: JudgeOutput, judge_config: JudgeConfig) -> bool:
    metadata = output.metadata or {}
    judge_metadata = output.judge_metadata
    provider = (
        judge_metadata.judge_provider
        if judge_metadata is not None
        else metadata.get("provider")
    )
    setup = (
        judge_metadata.judge_setup_name
        if judge_metadata is not None
        else metadata.get("judge_setup_name")
    )
    n_calls = (
        judge_metadata.n_calls
        if judge_metadata is not None
        else metadata.get("n_calls", 1)
    )
    aggregation = (
        judge_metadata.aggregation
        if judge_metadata is not None
        else metadata.get("aggregation")
    )
    exemplar_scale = (
        judge_metadata.exemplar_scale
        if judge_metadata is not None
        else metadata.get("exemplar_scale")
    )
    return (
        output.judge_model == judge_config.model_name
        and provider == judge_config.provider
        and setup == judge_config.judge_setup_name
        and n_calls == judge_config.n_calls
        and aggregation == judge_config.aggregation
        and exemplar_scale == judge_config.exemplar_scale
    )


def _load_existing_judgments_by_response_id(
    path: str | Path | None,
    judge_config: JudgeConfig,
) -> dict[str, JudgeOutput]:
    if path is None or not Path(path).exists():
        return {}
    existing: dict[str, JudgeOutput] = {}
    for row in read_jsonl(path):
        try:
            output = JudgeOutput(**row)
        except Exception:
            continue
        if _judge_output_matches_config(output, judge_config):
            existing[output.response_id] = output
    return existing


def _append_judge_output(path: str | Path, output: JudgeOutput) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(output.model_dump(mode="json"), ensure_ascii=False) + "\n")
        f.flush()


def _api_error_details(exc: Exception) -> dict[str, object]:
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    response = getattr(exc, "response", None)
    if request_id is None and response is not None:
        request_id = getattr(response, "headers", {}).get("x-request-id")
    code = getattr(exc, "code", None)
    error_type = getattr(exc, "type", None)
    body = getattr(exc, "body", None)
    return {
        "exception_type": type(exc).__name__,
        "status_code": status_code,
        "request_id": request_id,
        "code": code,
        "error_type": error_type,
        "body": body,
        "message": str(exc),
    }


def _error_hint(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    if status_code in {401, 403} or "api key" in message or "unauthorized" in message:
        return "Check AZURE_OPENAI_API_KEY, endpoint permissions, and deployment access."
    if status_code == 404 or "deployment" in message or "not found" in message:
        return "Check --model-name matches the Azure deployment name and --azure-base-url points at the right resource."
    if status_code == 429 or "rate limit" in message or "too many requests" in message:
        return "Likely rate limited. Lower --max-workers, wait, or check Azure quota for this deployment."
    if status_code in {500, 502, 503, 504} or "timeout" in message:
        return "Transient service/network error. Rerun the command; completed judgments will be skipped."
    if "json" in message:
        return "Judge returned malformed/empty JSON. Try increasing --max-tokens or rerun the failed responses."
    return "Rerun with a lower --max-workers if this happened during parallel judging."


def _append_judge_error(path: str | Path, row: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def judge_responses(
    responses: list[ModelResponse],
    judge_config: JudgeConfig,
    *,
    runtime_save_path: str | Path | None = None,
    skip_existing: bool = True,
    max_workers: int = 1,
    error_save_path: str | Path | None = None,
    stop_on_error: bool = True,
) -> list[JudgeOutput]:
    def build_client():
        return build_judge_client(
            provider=judge_config.provider,
            model_name=judge_config.model_name,
            temperature=judge_config.temperature,
            max_tokens=judge_config.max_tokens,
            device_map=judge_config.device_map,
            torch_dtype=judge_config.torch_dtype,
            judge_setup_name=judge_config.judge_setup_name,
            exemplars_path=judge_config.exemplars_path,
            exemplar_scale=judge_config.exemplar_scale,
            azure_base_url=judge_config.azure_base_url,
            request_timeout=judge_config.request_timeout,
        )

    thread_local = threading.local()

    def client_for_thread():
        client = getattr(thread_local, "client", None)
        if client is None:
            client = build_client()
            thread_local.client = client
        return client

    def judge_one(response: ModelResponse) -> JudgeOutput:
        client = client_for_thread()
        raw_calls = []
        parsed_calls = []

        try:
            for call_index in range(judge_config.n_calls):
                try:
                    raw = client.judge_once(response)
                except Exception as exc:
                    details = _api_error_details(exc)
                    raise RuntimeError(
                        "Judge API call failed "
                        f"response_id={response.response_id} "
                        f"instance_id={response.instance_id} "
                        f"model={judge_config.model_name} "
                        f"provider={judge_config.provider} "
                        f"setup={judge_config.judge_setup_name} "
                        f"call_index={call_index + 1}/{judge_config.n_calls} "
                        f"status_code={details.get('status_code')} "
                        f"request_id={details.get('request_id')} "
                        f"hint={_error_hint(exc)} "
                        f"error={exc}"
                    ) from exc
                raw_calls.append(raw)
                try:
                    parsed = normalize_raw_judge_response(raw)
                    parsed_calls.append(raw_to_praise_instances(parsed))
                except Exception as exc:
                    raise RuntimeError(
                        "Judge response parsing failed "
                        f"response_id={response.response_id} "
                        f"instance_id={response.instance_id} "
                        f"model={judge_config.model_name} "
                        f"setup={judge_config.judge_setup_name} "
                        f"raw={raw!r} "
                        f"hint={_error_hint(exc)} "
                        f"error={exc}"
                    ) from exc
        except Exception:
            raise

        aggregated = aggregate_judge_calls(parsed_calls, judge_config.aggregation)

        return build_judge_output(
            instance_id=response.instance_id,
            response_id=response.response_id,
            judge_model=judge_config.model_name,
            praise_instances=aggregated,
            metadata={
                "provider": judge_config.provider,
                "judge_setup_name": judge_config.judge_setup_name,
                "exemplars_path": judge_config.exemplars_path,
                "exemplar_scale": judge_config.exemplar_scale,
                "n_calls": judge_config.n_calls,
                "aggregation": judge_config.aggregation,
                "raw_calls": raw_calls,
            },
        )

    max_workers = max(1, max_workers)
    outputs: list[JudgeOutput | None] = [None] * len(responses)
    existing_by_response_id = _load_existing_judgments_by_response_id(
        runtime_save_path,
        judge_config,
    )
    to_judge: list[tuple[int, ModelResponse]] = []

    for idx, response in enumerate(responses):
        existing = existing_by_response_id.get(response.response_id)
        if skip_existing and existing is not None:
            outputs[idx] = existing
            continue
        to_judge.append((idx, response))

    print(
        "Judge startup: "
        f"responses={len(responses)} "
        f"already_done={len(responses) - len(to_judge)} "
        f"to_judge={len(to_judge)} "
        f"provider={judge_config.provider} "
        f"model={judge_config.model_name} "
        f"setup={judge_config.judge_setup_name} "
        f"max_workers={max_workers} "
        f"request_timeout={judge_config.request_timeout}"
    )

    if not to_judge:
        return [output for output in outputs if output is not None]

    write_lock = threading.Lock()
    error_lock = threading.Lock()

    def record_output(idx: int, output: JudgeOutput) -> None:
        if runtime_save_path is not None:
            with write_lock:
                if output.response_id not in existing_by_response_id:
                    _append_judge_output(runtime_save_path, output)
                    existing_by_response_id[output.response_id] = output
        outputs[idx] = output

    def record_error(idx: int, response: ModelResponse, exc: Exception) -> None:
        error_row = {
            "response_id": response.response_id,
            "instance_id": response.instance_id,
            "judge_model": judge_config.model_name,
            "judge_provider": judge_config.provider,
            "judge_setup_name": judge_config.judge_setup_name,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "hint": _error_hint(exc.__cause__ or exc),
            "cause": _api_error_details(exc.__cause__) if exc.__cause__ else None,
        }
        if error_save_path is not None:
            with error_lock:
                _append_judge_error(error_save_path, error_row)
        if stop_on_error:
            raise exc

    if max_workers == 1:
        for idx, response in _progress(
            to_judge,
            desc="Judging responses",
            total=len(to_judge),
        ):
            try:
                record_output(idx, judge_one(response))
            except Exception as exc:
                record_error(idx, response, exc)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(judge_one, response): idx
                for idx, response in to_judge
            }
            for future in _progress(
                concurrent.futures.as_completed(futures),
                desc="Judging responses",
                total=len(futures),
            ):
                idx = futures[future]
                response = responses[idx]
                try:
                    record_output(idx, future.result())
                except Exception as exc:
                    record_error(idx, response, exc)

    return [output for output in outputs if output is not None]
