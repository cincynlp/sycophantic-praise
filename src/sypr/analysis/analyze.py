from __future__ import annotations

import pandas as pd

from sypr.schemas import ScoreOutput


def scores_to_dataframe(scores: list[ScoreOutput]) -> pd.DataFrame:
    rows = []
    for s in scores:
        rows.append(
            {
                "instance_id": s.instance_id,
                "response_id": s.response_id,
                "regime_name": s.regime_name,
                "actual_value": s.actual_value,
                "expected_value": s.expected_value,
                "delta": s.delta,
                "P_effort": s.subscores.observed.effort,
                "P_utterance": s.subscores.observed.utterance,
                "P_individual": s.subscores.observed.individual,
                "W_effort": s.subscores.warranted.effort,
                "W_utterance": s.subscores.warranted.utterance,
                "W_individual": s.subscores.warranted.individual,
                "X_effort": s.subscores.excess.effort,
                "X_utterance": s.subscores.excess.utterance,
                "X_individual": s.subscores.excess.individual,
                "sypr_score": s.sypr_score,
                "model_name": s.metadata.get("model_name") or s.metadata.get("response_model"),
                "domain": s.metadata.get("domain"),
                "difficulty": s.metadata.get("difficulty"),
                "difficulty_score": s.metadata.get("difficulty_score"),
                "difficulty_bin": s.metadata.get("difficulty_bin"),
                "prompt_condition": s.metadata.get("prompt_condition"),
                "persona_id": s.metadata.get("persona_id"),
                "persona_type": s.metadata.get("persona_type"),
                "persona_security": s.metadata.get("persona_security"),
                "at_issue_status": s.metadata.get("at_issue_status"),
                "belief_framing": s.metadata.get("belief_framing"),
                "context_length_turns": s.metadata.get("context_length_turns"),
                "context_length_tokens": s.metadata.get("context_length_tokens"),
                "expected_value_source": s.metadata.get("expected_value_source"),
            }
        )
    return pd.DataFrame(rows)


def summarize_scores(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "n": len(df),
                "mean_sypr_score": df["sypr_score"].mean(),
                "mean_P_effort": df["P_effort"].mean(),
                "mean_P_utterance": df["P_utterance"].mean(),
                "mean_P_individual": df["P_individual"].mean(),
                "mean_X_effort": df["X_effort"].mean(),
                "mean_X_utterance": df["X_utterance"].mean(),
                "mean_X_individual": df["X_individual"].mean(),
            }
        ]
    )


def summarize_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=("sypr_score", "size"),
            mean_sypr_score=("sypr_score", "mean"),
            mean_P_effort=("P_effort", "mean"),
            mean_P_utterance=("P_utterance", "mean"),
            mean_P_individual=("P_individual", "mean"),
            mean_X_effort=("X_effort", "mean"),
            mean_X_utterance=("X_utterance", "mean"),
            mean_X_individual=("X_individual", "mean"),
        )
        .reset_index()
    )
    return grouped
