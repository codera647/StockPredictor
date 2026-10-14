import json
from datetime import date

import numpy as np
import pandas as pd

from stock_predictor.config import PipelineError


def clean_actions(actions: pd.DataFrame, symbols: set[str]) -> tuple[pd.DataFrame, dict]:
    """Normalize event revisions without treating an acquirer's deal as a share-unit change."""
    records = []
    duplicate = int(actions.duplicated(["event_id", "record_version"]).sum())
    actions = actions.drop_duplicates(["event_id", "record_version"])
    conflicts = set(actions.loc[actions.duplicated("event_id", keep=False), "event_id"])
    for row in actions.itertuples(index=False):
        payload = json.loads(row.payload_json)
        kind = row.action_type
        day = payload.get("ex_date") or payload.get("effective_date") or payload.get("process_date")
        ratio, reason = None, None
        if row.event_id in conflicts:
            reason = "conflicting_action_revisions"
        if kind in ("forward_splits", "reverse_splits"):
            affected = [payload.get("symbol")]
            try:
                ratio = float(payload["new_rate"]) / float(payload["old_rate"])
                if not np.isfinite(ratio) or ratio <= 0:
                    raise ValueError("invalid ratio")
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                ratio, reason = None, "invalid_split_ratio"
        elif kind == "cash_dividends":
            affected = [payload.get("symbol")]
        elif kind in ("cash_mergers", "stock_mergers", "stock_and_cash_mergers"):
            affected = [payload.get("acquiree_symbol")]
            reason = reason or "unsupported_terminal_action"
        elif kind == "spin_offs":
            affected = [payload.get("source_symbol"), payload.get("new_symbol")]
            reason = reason or "unsupported_spinoff"
        elif kind == "name_changes":
            affected = [payload.get("old_symbol"), payload.get("new_symbol")]
            reason = reason or "identity_boundary_requires_review"
        else:
            affected = json.loads(row.symbols_json)
            reason = reason or "unsupported_action_type"
        try:
            day = date.fromisoformat(day).isoformat()
        except (TypeError, ValueError):
            day = None
            reason = "missing_action_date"
        for symbol in sorted(set(affected) & symbols):
            records.append(
                {
                    "symbol": symbol,
                    "event_id": row.event_id,
                    "action_type": kind,
                    "effective_session_date": day,
                    "split_ratio": ratio,
                    "process_date": row.process_date,
                    "observed_at": row.retrieved_at,
                    "announcement_at": None,
                    "historical_availability": "reconstructed_effective_date",
                    "quality_status": "requires_review" if reason else "typed_source_record",
                    "quality_reason": reason,
                    "record_version": row.record_version,
                    "raw_snapshot_key": row.raw_snapshot_key,
                }
            )
    columns = [
        "symbol",
        "event_id",
        "action_type",
        "effective_session_date",
        "split_ratio",
        "process_date",
        "observed_at",
        "announcement_at",
        "historical_availability",
        "quality_status",
        "quality_reason",
        "record_version",
        "raw_snapshot_key",
    ]
    return pd.DataFrame(records, columns=columns), {
        "identical_duplicates": duplicate,
        "conflicting_event_ids": len(conflicts),
    }


def clean_bars(
    raw: pd.DataFrame,
    actions: pd.DataFrame,
    reference: pd.DataFrame,
    sessions: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    if raw.empty or sessions.empty:
        raise PipelineError("Daily dataset needs actual bars and exchange sessions")
    if set(raw.feed) != {"sip"} or set(raw.adjustment_basis) != {"raw"}:
        raise PipelineError("Mixed feeds or adjusted bars are incompatible with this dataset")
    input_rows = len(raw)
    if sessions.session_date.duplicated().any() or sessions.session_date.tolist() != sorted(
        sessions.session_date
    ):
        raise PipelineError("Exchange session keys must be unique and chronological")
    if not set(raw.session_date).issubset(set(sessions.session_date)):
        raise PipelineError("Bars outside the archived exchange calendar require review")
    keys = ["instrument_id", "session_date", "source", "feed", "bar_type"]
    duplicate = int(raw.duplicated(keys + ["record_version"]).sum())
    raw = raw.drop_duplicates(keys + ["record_version"])
    conflict_keys = raw.loc[raw.duplicated(keys, keep=False), keys].drop_duplicates()
    raw = raw.merge(conflict_keys.assign(conflicting_revision=True), on=keys, how="left")
    # All conflicting snapshots remain in the original archive; no last-revision-wins fitting.
    raw = raw.sort_values("retrieved_at").drop_duplicates(keys, keep="first")
    reference = reference.drop_duplicates()
    if reference.duplicated(["symbol", "session_date"]).any():
        raise PipelineError("Conflicting split-reference observations require review")
    calendar_dates = sessions.session_date.tolist()
    calendar = sessions.set_index("session_date")
    panels, reviews = [], []
    for symbol, observed in raw.groupby("symbol", sort=True):
        observed = observed.set_index("session_date").reindex(calendar_dates)
        original = observed.copy()
        for name in (
            "symbol",
            "instrument_id",
            "exchange",
            "currency",
            "source",
            "feed",
            "bar_type",
        ):
            values = raw.loc[raw.symbol == symbol, name].dropna().unique()
            if len(values) != 1:
                raise PipelineError(f"Ambiguous {name} for {symbol}")
            observed[name] = values[0]
        observed["session_date"] = calendar_dates
        observed["session_index"] = np.arange(len(calendar_dates))
        observed["session_close_at"] = pd.to_datetime(calendar["close"], utc=True).to_numpy()
        observed["cutoff_at"] = pd.to_datetime(observed.session_close_at, utc=True) + pd.Timedelta(
            minutes=30
        )
        observed["missing_bar"] = original.record_version.isna().to_numpy()
        prices = observed[["open_raw", "high_raw", "low_raw", "close_raw"]].apply(
            pd.to_numeric, errors="coerce"
        )
        volume = pd.to_numeric(observed.volume_raw, errors="coerce")
        valid = (
            np.isfinite(prices).all(axis=1)
            & prices.gt(0).all(axis=1)
            & np.isfinite(volume)
            & volume.ge(0)
            & volume.eq(np.floor(volume))
            & prices.high_raw.ge(prices[["open_raw", "close_raw"]].max(axis=1))
            & prices.low_raw.le(prices[["open_raw", "close_raw"]].min(axis=1))
            & prices.high_raw.ge(prices.low_raw)
            & observed.quality_status.eq("valid_observation")
            & ~observed.conflicting_revision.fillna(False).astype(bool)
        ).to_numpy(copy=True)
        factors = np.ones(len(observed), dtype=float)
        action_observed_ns = np.full(len(observed), pd.NaT.value, dtype=np.int64)
        boundary = np.zeros(len(observed), dtype=bool)
        cash_event = np.zeros(len(observed), dtype=bool)
        for action in actions.loc[actions.symbol == symbol].itertuples(index=False):
            if not action.effective_session_date:
                valid[:] = False
                reviews.append(
                    {"symbol": symbol, "reason": "undated_action", "event_id": action.event_id}
                )
                continue
            idx = int(np.searchsorted(calendar_dates, action.effective_session_date))
            if idx >= len(observed):
                continue
            stamp = pd.to_datetime(action.observed_at, utc=True, errors="coerce")
            if pd.notna(stamp):
                action_observed_ns[idx:] = np.maximum(action_observed_ns[idx:], stamp.value)
            exact = calendar_dates[idx] == action.effective_session_date
            if (
                action.action_type == "cash_dividends"
                and action.quality_status == "typed_source_record"
            ):
                cash_event[idx] = True
            elif (
                action.action_type in ("forward_splits", "reverse_splits")
                and action.quality_status == "typed_source_record"
                and exact
            ):
                factors[idx:] *= action.split_ratio
            else:
                boundary[idx] = True
                reviews.append(
                    {
                        "symbol": symbol,
                        "session_date": calendar_dates[idx],
                        "reason": action.quality_reason or "non_session_split_date",
                        "event_id": action.event_id,
                    }
                )
        observed["split_factor_prefix"] = factors
        observed["action_observed_at_basis"] = pd.to_datetime(action_observed_ns, utc=True)
        for name in ("open", "high", "low", "close"):
            observed[name + "_basis"] = prices[name + "_raw"].to_numpy() * factors
        observed["volume_basis"] = volume.to_numpy() / factors
        audit = observed.reset_index(drop=True).merge(
            reference.loc[reference.symbol == symbol],
            on=["symbol", "session_date"],
            how="left",
            validate="one_to_one",
        )
        reconciled = np.ones(len(observed), dtype=bool)
        for name, letter in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")):
            ratio = audit[name + "_raw"] / pd.to_numeric(
                audit[letter + "_split_reference"], errors="coerce"
            )
            finite = np.isfinite(ratio.to_numpy()) & ratio.gt(0).to_numpy()
            if not finite.any():
                reconciled[:] = False
                continue
            anchor = np.flatnonzero(finite)[-1]
            expected = factors[anchor] / factors
            reconciled &= finite & np.isclose(
                ratio.to_numpy() / ratio.iloc[anchor],
                expected,
                rtol=config["split_ratio_tolerance"],
                atol=1e-6,
            )
        jump = (
            np.log(observed.close_basis / observed.close_basis.shift(1))
            .abs()
            .gt(config["suspicious_log_jump"])
            .fillna(False)
            .to_numpy()
        )
        unresolved_jump = jump & ~cash_event & ~boundary
        valid &= reconciled & ~boundary & ~unresolved_jump
        observed["unsupported_action_boundary"] = boundary
        observed["unresolved_jump"] = unresolved_jump
        observed["split_reference_reconciled"] = reconciled
        observed["clean_valid"] = valid
        observed["source_quality_status"] = observed.quality_status
        observed["quality_status"] = np.where(valid, "validated_research_bar", "quarantined")
        observed["action_verification_status"] = np.where(
            reconciled, "provider_reconciled_research", "unresolved_adjustment"
        )
        observed["observed_available_at"] = observed.available_at
        observed["available_at_reconstructed"] = observed.cutoff_at
        observed["availability_method"] = config["historical_availability_policy"]
        observed["identity_verification_status"] = config["identity_policy"]
        observed["quality_reason_codes"] = [
            json.dumps(
                [
                    name
                    for name, mask in (
                        ("missing_bar", observed.missing_bar.to_numpy()),
                        (
                            "invalid_ohlcv_or_revision",
                            ~valid & reconciled & ~boundary & ~unresolved_jump,
                        ),
                        ("unresolved_adjustment", ~reconciled),
                        ("unsupported_action", boundary),
                        ("unresolved_jump", unresolved_jump),
                    )
                    if mask[i]
                ]
            )
            for i in range(len(observed))
        ]
        panels.append(observed.reset_index(drop=True))
    panel = pd.concat(panels, ignore_index=True)
    return panel, {
        "input_rows": input_rows,
        "identical_duplicates": duplicate,
        "conflicting_session_keys": len(conflict_keys),
        "calendar_aligned_rows": len(panel),
        "missing_sessions": int(panel.missing_bar.sum()),
        "valid_rows": int(panel.clean_valid.sum()),
        "unresolved_split_reference_rows": int((~panel.split_reference_reconciled).sum()),
        "unresolved_jump_rows": int(panel.unresolved_jump.sum()),
        "action_reviews": reviews,
    }
