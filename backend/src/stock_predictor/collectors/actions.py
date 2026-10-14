import json

from stock_predictor.storage.archive import digest, json_bytes


def action_rows(data: dict, envelope: dict, ref: dict) -> list[dict]:
    """Retain every action type/field. Process date is not announcement availability."""
    rows = []
    for action_type, records in (data["corporate_actions"] or {}).items():
        for record in records:
            symbols = sorted(
                {
                    value
                    for key, value in record.items()
                    if (key == "symbol" or key.endswith("_symbol")) and isinstance(value, str)
                }
            )
            rows.append(
                {
                    "event_id": str(record.get("id", digest(json_bytes(record)))),
                    "action_type": action_type,
                    "symbols_json": json.dumps(symbols),
                    "process_date": record.get("process_date"),
                    "effective_date": record.get("ex_date") or record.get("effective_date"),
                    "announcement_at": None,
                    "available_at": envelope["retrieved_at"],
                    "retrieved_at": envelope["retrieved_at"],
                    "availability_method": "observed_retrieval",
                    "record_version": digest(json_bytes(record)),
                    "verification_status": "unverified",
                    "response_hash": ref["sha256"],
                    "raw_snapshot_key": ref["key"],
                    "payload_json": json.dumps(record, sort_keys=True),
                }
            )
    return rows
