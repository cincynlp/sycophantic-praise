from __future__ import annotations

from collections import defaultdict

from sypr.schemas import (
    JudgeMetadata,
    JudgeOutput,
    PraiseInstance,
    PraiseTarget,
    RawJudgeResponse,
)


def normalize_raw_judge_response(raw: dict) -> RawJudgeResponse:
    return RawJudgeResponse(**raw)


def raw_to_praise_instances(raw: RawJudgeResponse) -> list[PraiseInstance]:
    items: list[PraiseInstance] = []

    for inst in raw.praise_instances:
        items.append(
            PraiseInstance(
                text=inst.text.strip(),
                target=PraiseTarget(inst.target),
                magnitude=inst.magnitude,
                confidence=None,
            )
        )
    return items


def aggregate_judge_calls(
    calls: list[list[PraiseInstance]],
    mode: str,
) -> list[PraiseInstance]:
    if mode == "none":
        if len(calls) != 1:
            raise ValueError("aggregation='none' requires exactly one judge call")
        return calls[0]

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)

    for call in calls:
        for item in call:
            key = (item.text.strip().lower(), item.target.value)
            grouped[key].append(item.magnitude)

    aggregated: list[PraiseInstance] = []
    for (text_key, target_str), magnitudes in grouped.items():
        if mode == "union_mean":
            magnitude = sum(magnitudes) / len(magnitudes)
        elif mode == "union_max":
            magnitude = max(magnitudes)
        else:
            raise ValueError(f"Unknown aggregation mode: {mode}")

        aggregated.append(
            PraiseInstance(
                text=text_key,
                target=PraiseTarget(target_str),
                magnitude=magnitude,
                confidence=len(magnitudes) / len(calls) if calls else None,
            )
        )

    return aggregated


def build_judge_output(
    instance_id: str,
    response_id: str,
    judge_model: str,
    praise_instances: list[PraiseInstance],
    metadata: dict | None = None,
) -> JudgeOutput:
    metadata = metadata or {}
    return JudgeOutput(
        instance_id=instance_id,
        response_id=response_id,
        judge_model=judge_model,
        praise_instances=praise_instances,
        raw_judge_text=None,
        judge_metadata=JudgeMetadata(
            judge_model=judge_model,
            judge_provider=metadata.get("provider"),
            judge_setup_name=metadata.get("judge_setup_name"),
            exemplar_scale=metadata.get("exemplar_scale"),
            n_calls=metadata.get("n_calls", 1),
            aggregation=metadata.get("aggregation"),
            raw_outputs=metadata.get("raw_calls", []),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in {"provider", "n_calls", "aggregation", "raw_calls"}
            },
        ),
        metadata=metadata,
    )
