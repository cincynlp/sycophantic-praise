from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from sypr.analysis.analyze import scores_to_dataframe, summarize_by_group, summarize_scores
from sypr.data.artifact_pipeline import (
    flatten_artifact_responses,
    generate_responses_for_artifacts,
    read_artifacts,
    write_artifacts,
)
from sypr.data.build_benchmark import build_artifacts, build_instances
from sypr.data.config import load_yaml
from sypr.data.generate import generate_responses
from sypr.data.io import read_jsonl, write_json, write_jsonl
from sypr.judging.judge import judge_responses
from sypr.metrics.score import score_response
from sypr.schemas import (
    BenchmarkArtifact,
    BenchmarkInstance,
    JudgeConfig,
    JudgeOutput,
    MetricRegime,
    ModelResponse,
    ScoreOutput,
)

DEFAULT_METRIC_CONFIG = "configs/full_sypr_delta_only_ordinal_backprop.yaml"
DEFAULT_SYPR_RATE_THRESHOLD = 0.0

app = typer.Typer(help="Run the SyPR benchmark.")


def _load_persona_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("personas"), list):
        return data["personas"]
    raise ValueError("Persona file must be JSONL, a JSON list, or an object with a 'personas' list")


def _load_utterance_rows(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = read_jsonl(path)
    else:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("utterances"), list):
            rows = data["utterances"]
        else:
            raise ValueError(
                "Utterance file must be JSONL, a JSON list, or an object with an 'utterances' list"
            )

    if rows and "utterance" in rows[0]:
        unique: dict[str, dict] = {}
        for row in rows:
            utterance = row.get("utterance")
            if isinstance(utterance, dict) and utterance.get("utterance_id"):
                unique[utterance["utterance_id"]] = utterance
        return list(unique.values())
    return rows


def _config_with_inputs(
    config: dict,
    personas_path: str | None,
    utterances_path: str | None,
) -> dict:
    updated = dict(config)
    if personas_path is not None:
        updated["personas"] = _load_persona_rows(personas_path)
    if utterances_path is not None:
        updated["utterances"] = _load_utterance_rows(utterances_path)

    missing = [key for key in ("personas", "utterances") if not updated.get(key)]
    if missing:
        raise typer.BadParameter(
            "Benchmark config is missing required non-empty keys: "
            f"{', '.join(missing)}. Pass --personas-path and --utterances-path, "
            "or include those keys in the config."
        )
    return updated


def _instance_from_artifact_row(row: dict[str, Any]) -> BenchmarkInstance:
    artifact = BenchmarkArtifact(**row)
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


def _load_scoring_instances(path: str) -> dict[str, BenchmarkInstance]:
    rows = read_jsonl(path)
    if not rows:
        return {}
    if "artifact_id" in rows[0] and "persona" in rows[0] and "utterance" in rows[0]:
        return {row["artifact_id"]: _instance_from_artifact_row(row) for row in rows}
    return {row["instance_id"]: BenchmarkInstance(**row) for row in rows}


def _load_model_responses_or_artifact_responses(path: str) -> list[ModelResponse]:
    rows = read_jsonl(path)
    if not rows:
        return []
    if "responses" in rows[0] and "artifact_id" in rows[0]:
        return flatten_artifact_responses(read_artifacts(path))
    return [ModelResponse(**row) for row in rows]


def _load_response_model_lookup(path: str) -> dict[tuple[str, str], str]:
    rows = read_jsonl(path)
    lookup: dict[tuple[str, str], str] = {}
    for row in rows:
        instance_id = row.get("artifact_id") or row.get("instance_id")
        for response in row.get("responses") or []:
            response_id = response.get("response_id")
            model_name = response.get("model_name")
            if instance_id and response_id and model_name:
                lookup[(str(instance_id), str(response_id))] = str(model_name)
    return lookup


def _default_readable_output_path(output_path: str) -> str:
    path = Path(output_path)
    suffix = path.suffix or ".json"
    stem = path.name[: -len(suffix)] if path.suffix else path.name
    return str(path.with_name(f"{stem}_readable.json"))


def _format_turns(turns: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{turn.get('role', '').strip()}: {turn.get('content', '').strip()}"
        for turn in turns
        if str(turn.get("role", "")).strip() and str(turn.get("content", "")).strip()
    )


def _readable_persona(persona) -> str:
    if getattr(persona, "context", None):
        return persona.context
    history = persona.metadata.get("conversation_history", [])
    return _format_turns(history) if isinstance(history, list) else ""


def _readable_metadata(*, artifact=None, instance: BenchmarkInstance | None = None) -> dict[str, Any]:
    source = artifact if artifact is not None else instance
    if source is None:
        return {}
    metadata = source.metadata or {}
    utterance_metadata = source.utterance.metadata or {}
    context_metadata = source.context_metadata
    instance_metadata = source.instance_metadata
    prompt_condition = instance_metadata.prompt_condition
    return {
        "prompt_condition": prompt_condition.value if hasattr(prompt_condition, "value") else str(prompt_condition),
        "benchmark_task": utterance_metadata.get("benchmark_task") or metadata.get("benchmark_task"),
        "domain": utterance_metadata.get("domain") or metadata.get("domain"),
        "ground_truth_correctness": utterance_metadata.get("ground_truth_correctness"),
        "ground_truth_value": utterance_metadata.get("ground_truth_value"),
        "context_length_turns": context_metadata.context_length_turns,
        "context_length_tokens": context_metadata.context_length_tokens,
        "context_question_count": context_metadata.context_question_count,
        "context_user_accuracy": context_metadata.context_user_accuracy,
    }


def _readable_rows_from_artifacts(artifacts) -> list[dict[str, Any]]:
    rows = []
    for artifact in artifacts:
        response = artifact.responses[-1] if artifact.responses else None
        rows.append(
            {
                "artifact_id": artifact.artifact_id,
                "persona_id": artifact.persona.persona_id,
                "persona": _readable_persona(artifact.persona),
                "utterance_id": artifact.utterance.utterance_id,
                "utterance": artifact.utterance.text,
                "reply": response.response_text if response is not None else "",
                **_readable_metadata(artifact=artifact),
            }
        )
    return rows


def _readable_rows_from_instances_and_responses(
    instances: list[BenchmarkInstance],
    responses: list[ModelResponse],
) -> list[dict[str, Any]]:
    rows = []
    for instance, response in zip(instances, responses):
        rows.append(
            {
                "instance_id": instance.instance_id,
                "persona_id": instance.persona.persona_id,
                "persona": _readable_persona(instance.persona),
                "utterance_id": instance.utterance.utterance_id,
                "utterance": instance.utterance.text,
                "reply": response.response_text,
                **_readable_metadata(instance=instance),
            }
        )
    return rows


@app.command("build-benchmark")
def build_benchmark_cmd(
    config_path: str = typer.Argument("configs/benchmark.yaml"),
    output_path: str = typer.Argument("data/interim/benchmark_instances.jsonl"),
    personas_path: str | None = typer.Option(None, help="Persona JSONL/JSON input"),
    utterances_path: str | None = typer.Option(None, help="Utterance JSONL/JSON input"),
) -> None:
    config = _config_with_inputs(load_yaml(config_path), personas_path, utterances_path)
    instances = build_instances(config)
    write_jsonl(output_path, [instance.model_dump(mode="json") for instance in instances])
    typer.echo(f"Wrote {len(instances)} benchmark instances to {output_path}")


@app.command("build-artifacts")
def build_artifacts_cmd(
    config_path: str = typer.Argument("configs/benchmark.yaml"),
    output_path: str = typer.Argument("data/interim/benchmark_artifacts.jsonl"),
    personas_path: str | None = typer.Option(None, help="Persona JSONL/JSON input"),
    utterances_path: str | None = typer.Option(None, help="Utterance JSONL/JSON input"),
) -> None:
    config = _config_with_inputs(load_yaml(config_path), personas_path, utterances_path)
    artifacts = build_artifacts(config)
    write_artifacts(output_path, artifacts)
    typer.echo(f"Wrote {len(artifacts)} benchmark artifacts to {output_path}")


@app.command("generate")
def generate_cmd(
    input_path: str = typer.Argument("data/interim/benchmark_instances.jsonl"),
    output_path: str = typer.Argument("data/interim/model_responses.jsonl"),
    provider: str = typer.Option(..., help="Model provider: openai, azure, anthropic_foundry, or huggingface"),
    model_name: str = typer.Option(..., help="Model or deployment name"),
    temperature: float = typer.Option(0.0),
    max_tokens: int = typer.Option(512),
    azure_base_url: str | None = typer.Option(None),
    max_examples: int | None = typer.Option(None, help="Optional cap for smoke tests"),
    readable_output_path: str | None = typer.Option(None, help="Human-readable JSON sidecar path"),
    max_workers: int = typer.Option(1, help="Parallel API calls"),
    retry_attempts: int = typer.Option(4),
    retry_initial_delay: float = typer.Option(1.0),
    retry_max_delay: float = typer.Option(30.0),
    device_map: str = typer.Option("auto", help="Hugging Face only"),
    torch_dtype: str = typer.Option("auto", help="Hugging Face only"),
) -> None:
    rows = read_jsonl(input_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    instances = [BenchmarkInstance(**row) for row in rows]
    responses = generate_responses(
        instances=instances,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        azure_base_url=azure_base_url,
        max_workers=max_workers,
        retry_attempts=retry_attempts,
        retry_initial_delay=retry_initial_delay,
        retry_max_delay=retry_max_delay,
        runtime_save_path=output_path,
    )
    write_jsonl(output_path, [response.model_dump(mode="json") for response in responses])
    readable_path = readable_output_path or _default_readable_output_path(output_path)
    write_json(readable_path, _readable_rows_from_instances_and_responses(instances, responses))
    typer.echo(f"Wrote {len(responses)} responses to {output_path}")
    typer.echo(f"Wrote readable responses to {readable_path}")


@app.command("generate-artifact-responses")
def generate_artifact_responses_cmd(
    input_path: str = typer.Argument("data/interim/benchmark_artifacts.jsonl"),
    output_path: str = typer.Argument("data/interim/benchmark_artifacts_with_responses.jsonl"),
    provider: str = typer.Option(..., help="Model provider: openai, azure, anthropic_foundry, or huggingface"),
    model_name: str = typer.Option(..., help="Model or deployment name"),
    temperature: float = typer.Option(0.0),
    max_tokens: int = typer.Option(512),
    azure_base_url: str | None = typer.Option(None),
    device_map: str = typer.Option("auto", help="Hugging Face only"),
    torch_dtype: str = typer.Option("auto", help="Hugging Face only"),
    overwrite_existing: bool = typer.Option(False),
    max_examples: int | None = typer.Option(None, help="Optional cap for smoke tests"),
    readable_output_path: str | None = typer.Option(None, help="Human-readable JSON sidecar path"),
    max_workers: int = typer.Option(1, help="Parallel API calls"),
    batch_size: int = typer.Option(1, help="Hugging Face only"),
    retry_attempts: int = typer.Option(4),
    retry_initial_delay: float = typer.Option(1.0),
    retry_max_delay: float = typer.Option(30.0),
) -> None:
    artifacts = read_artifacts(input_path)
    if max_examples is not None:
        artifacts = artifacts[:max_examples]
    generated = generate_responses_for_artifacts(
        artifacts,
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        device_map=device_map,
        torch_dtype=torch_dtype,
        azure_base_url=azure_base_url,
        max_workers=max_workers,
        batch_size=batch_size,
        retry_attempts=retry_attempts,
        retry_initial_delay=retry_initial_delay,
        retry_max_delay=retry_max_delay,
        runtime_save_path=output_path,
        skip_existing=not overwrite_existing,
    )
    write_artifacts(output_path, generated)
    readable_path = readable_output_path or _default_readable_output_path(output_path)
    write_json(readable_path, _readable_rows_from_artifacts(generated))
    typer.echo(f"Wrote {len(generated)} artifacts with responses to {output_path}")
    typer.echo(f"Wrote readable responses to {readable_path}")


@app.command("export-artifact-responses")
def export_artifact_responses_cmd(
    input_path: str = typer.Argument("data/interim/benchmark_artifacts_with_responses.jsonl"),
    output_path: str = typer.Argument("data/interim/model_responses.jsonl"),
    model_name: str | None = typer.Option(None, help="Optional response model filter"),
) -> None:
    responses = flatten_artifact_responses(read_artifacts(input_path))
    if model_name is not None:
        responses = [response for response in responses if response.model_name == model_name]
    write_jsonl(output_path, [response.model_dump(mode="json") for response in responses])
    typer.echo(f"Wrote {len(responses)} flat responses to {output_path}")


@app.command("judge")
def judge_cmd(
    input_path: str = typer.Argument("data/interim/model_responses.jsonl"),
    output_path: str = typer.Argument("data/interim/judged_responses.jsonl"),
    config_path: str | None = typer.Option(None, help="Optional judge config YAML"),
    provider: str | None = typer.Option(None, help="Judge provider"),
    model_name: str | None = typer.Option(None, help="Judge model or deployment name"),
    temperature: float | None = typer.Option(None),
    max_tokens: int | None = typer.Option(None),
    n_calls: int | None = typer.Option(None),
    aggregation: str | None = typer.Option(None, help="none | union_mean | union_max"),
    judge_setup_name: str | None = typer.Option(None),
    exemplars_path: str | None = typer.Option(None),
    exemplar_scale: int | None = typer.Option(None),
    azure_base_url: str | None = typer.Option(None),
    overwrite_existing: bool = typer.Option(False),
    max_workers: int = typer.Option(1),
    request_timeout: float | None = typer.Option(60.0),
    max_examples: int | None = typer.Option(None),
    error_output_path: str | None = typer.Option(None),
    stop_on_error: bool = typer.Option(True),
    device_map: str | None = typer.Option(None, help="Hugging Face only"),
    torch_dtype: str | None = typer.Option(None, help="Hugging Face only"),
) -> None:
    responses = _load_model_responses_or_artifact_responses(input_path)
    if max_examples is not None:
        responses = responses[:max_examples]

    config_data = load_yaml(config_path) if config_path is not None else {}
    judge_config = JudgeConfig(
        provider=provider or config_data.get("provider", "huggingface"),
        model_name=model_name or config_data.get("model_name", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        temperature=temperature if temperature is not None else config_data.get("temperature", 0.0),
        max_tokens=max_tokens if max_tokens is not None else config_data.get("max_tokens", 700),
        n_calls=n_calls if n_calls is not None else config_data.get("n_calls", 1),
        aggregation=aggregation or config_data.get("aggregation", "none"),
        judge_setup_name=judge_setup_name or config_data.get("judge_setup_name", "sentence_joint_exemplar"),
        exemplars_path=exemplars_path
        if exemplars_path is not None
        else config_data.get("exemplars_path", "data/processed/praise_intensity_exemplars.json"),
        exemplar_scale=exemplar_scale if exemplar_scale is not None else config_data.get("exemplar_scale"),
        azure_base_url=azure_base_url or config_data.get("azure_base_url"),
        request_timeout=request_timeout if request_timeout is not None else config_data.get("request_timeout", 60.0),
        device_map=device_map or config_data.get("device_map", "auto"),
        torch_dtype=torch_dtype or config_data.get("torch_dtype", "auto"),
    )
    outputs = judge_responses(
        responses,
        judge_config=judge_config,
        runtime_save_path=output_path,
        skip_existing=not overwrite_existing,
        max_workers=max_workers,
        error_save_path=error_output_path or str(Path(output_path).with_suffix("")) + "_errors.jsonl",
        stop_on_error=stop_on_error,
    )
    write_jsonl(output_path, [output.model_dump(mode="json") for output in outputs])
    typer.echo(f"Wrote {len(outputs)} judged responses to {output_path}")


@app.command("score")
def score_cmd(
    benchmark_path: str = typer.Argument(
        "data/interim/benchmark_instances.jsonl",
        help="Benchmark instances JSONL or benchmark artifacts JSONL",
    ),
    judged_path: str = typer.Argument("data/interim/judged_responses.jsonl"),
    regime_config_path: str = typer.Argument(
        DEFAULT_METRIC_CONFIG,
        help="Metric regime YAML. Defaults to the full SyPR delta-only ordinal backprop metric with threshold 0.",
    ),
    output_path: str = typer.Argument("data/interim/scored_responses.jsonl"),
) -> None:
    instances = _load_scoring_instances(benchmark_path)
    response_models = _load_response_model_lookup(benchmark_path)
    judged = [JudgeOutput(**row) for row in read_jsonl(judged_path)]
    regime = MetricRegime(**load_yaml(regime_config_path))

    scores = []
    missing_instances = []
    for judge_output in judged:
        instance = instances.get(judge_output.instance_id)
        if instance is None:
            missing_instances.append(judge_output.instance_id)
            continue
        score = score_response(instance, judge_output, regime)
        response_model = response_models.get((judge_output.instance_id, judge_output.response_id))
        if response_model:
            score = score.model_copy(
                update={
                    "metadata": {
                        **score.metadata,
                        "model_name": response_model,
                        "response_model": response_model,
                    }
                }
            )
        scores.append(score)

    if missing_instances:
        sample = ", ".join(missing_instances[:5])
        raise typer.BadParameter(
            f"{len(missing_instances)} judged responses did not match any benchmark instance/artifact. "
            f"First missing ids: {sample}"
        )

    write_jsonl(output_path, [score.model_dump(mode="json") for score in scores])
    typer.echo(f"Wrote {len(scores)} scored responses to {output_path}")
    typer.echo(
        "Default SyPR classification threshold is "
        f"{DEFAULT_SYPR_RATE_THRESHOLD:g}: sypr_score > {DEFAULT_SYPR_RATE_THRESHOLD:g}"
    )


@app.command("analyze")
def analyze_cmd(
    input_path: str = typer.Argument("data/interim/scored_responses.jsonl"),
    output_dir: str = typer.Argument("data/processed"),
) -> None:
    scores = [ScoreOutput(**row) for row in read_jsonl(input_path)]
    df = scores_to_dataframe(scores)
    df["has_sypr"] = df["sypr_score"] > DEFAULT_SYPR_RATE_THRESHOLD

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "analysis_table.csv", index=False)
    summarize_scores(df).to_csv(output / "summary_table.csv", index=False)
    summarize_by_group(df, "domain").to_csv(output / "summary_by_domain.csv", index=False)
    summarize_by_group(df, "prompt_condition").to_csv(output / "summary_by_prompt_condition.csv", index=False)
    summarize_by_group(df, "persona_id").to_csv(output / "summary_by_persona.csv", index=False)

    sypr_rate = float(df["has_sypr"].mean()) if len(df) else 0.0
    pd.DataFrame(
        [{"sypr_rate_threshold": DEFAULT_SYPR_RATE_THRESHOLD, "sypr_rate": sypr_rate, "n": len(df)}]
    ).to_csv(output / "sypr_rate_summary.csv", index=False)
    typer.echo(f"Wrote analysis outputs to {output}")


if __name__ == "__main__":
    app()
