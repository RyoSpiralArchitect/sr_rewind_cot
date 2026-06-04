from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cases (
    case_hash TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    stance_pattern TEXT NOT NULL,
    metaphor_mode TEXT NOT NULL,
    vehicle_spec TEXT NOT NULL,
    domain_distance TEXT NOT NULL,
    risk_domain TEXT NOT NULL,
    case_json TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    run_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    case_hash TEXT NOT NULL,
    stance_pattern TEXT NOT NULL,
    metaphor_mode TEXT NOT NULL,
    vehicle_spec TEXT NOT NULL,
    domain_distance TEXT NOT NULL,
    mapping_visibility TEXT NOT NULL,
    control_arm TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    temperature REAL NOT NULL,
    sample_index INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    generated_text TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audits (
    audit_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    audit_type TEXT NOT NULL,
    judge_provider TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judge_index INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(run_id, audit_type, judge_provider, judge_model, judge_index, prompt_version)
);

CREATE TABLE IF NOT EXISTS quality_pairs (
    pair_id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL,
    case_hash TEXT NOT NULL,
    metaphor_mode TEXT NOT NULL,
    vehicle_spec TEXT NOT NULL,
    mapping_visibility TEXT NOT NULL,
    control_arm TEXT NOT NULL,
    text_a_run_id TEXT NOT NULL,
    text_b_run_id TEXT NOT NULL,
    judge_provider TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judge_index INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(text_a_run_id, text_b_run_id, judge_provider, judge_model, judge_index, prompt_version)
);
"""


class HarnessDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_SQL)
        self._ensure_current_columns()
        self.conn.commit()

    def _ensure_current_columns(self) -> None:
        """Best-effort migration for older DBs that predate current metadata columns."""
        def columns(table: str) -> set[str]:
            return {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}

        gen_cols = columns("generations")
        if "case_hash" not in gen_cols:
            self.conn.execute("ALTER TABLE generations ADD COLUMN case_hash TEXT NOT NULL DEFAULT 'legacy_case_hash'")
        if "metaphor_mode" not in gen_cols:
            self.conn.execute("ALTER TABLE generations ADD COLUMN metaphor_mode TEXT NOT NULL DEFAULT 'structural'")
        if "vehicle_spec" not in gen_cols:
            self.conn.execute("ALTER TABLE generations ADD COLUMN vehicle_spec TEXT NOT NULL DEFAULT 'constrained'")
        if "mapping_visibility" not in gen_cols:
            self.conn.execute("ALTER TABLE generations ADD COLUMN mapping_visibility TEXT NOT NULL DEFAULT 'legacy'")

        case_cols = columns("cases")
        if "metaphor_mode" not in case_cols:
            self.conn.execute("ALTER TABLE cases ADD COLUMN metaphor_mode TEXT NOT NULL DEFAULT 'structural'")
        if "vehicle_spec" not in case_cols:
            self.conn.execute("ALTER TABLE cases ADD COLUMN vehicle_spec TEXT NOT NULL DEFAULT 'constrained'")

        qp_cols = columns("quality_pairs")
        if "case_hash" not in qp_cols:
            self.conn.execute("ALTER TABLE quality_pairs ADD COLUMN case_hash TEXT NOT NULL DEFAULT 'legacy_case_hash'")
        if "metaphor_mode" not in qp_cols:
            self.conn.execute("ALTER TABLE quality_pairs ADD COLUMN metaphor_mode TEXT NOT NULL DEFAULT 'structural'")
        if "vehicle_spec" not in qp_cols:
            self.conn.execute("ALTER TABLE quality_pairs ADD COLUMN vehicle_spec TEXT NOT NULL DEFAULT 'constrained'")
        if "mapping_visibility" not in qp_cols:
            self.conn.execute("ALTER TABLE quality_pairs ADD COLUMN mapping_visibility TEXT NOT NULL DEFAULT 'legacy'")

    def close(self) -> None:
        self.conn.close()

    def upsert_case(self, case_dict: dict[str, Any], case_hash: str) -> None:
        mapping = case_dict.get("mapping", {})
        self.conn.execute(
            """
            INSERT INTO cases(case_hash, case_id, stance_pattern, metaphor_mode, vehicle_spec, domain_distance, risk_domain, case_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(case_hash) DO UPDATE SET
              case_id=excluded.case_id,
              stance_pattern=excluded.stance_pattern,
              metaphor_mode=excluded.metaphor_mode,
              vehicle_spec=excluded.vehicle_spec,
              domain_distance=excluded.domain_distance,
              risk_domain=excluded.risk_domain,
              case_json=excluded.case_json,
              updated_at=excluded.updated_at
            """,
            (
                case_hash,
                case_dict["case_id"],
                case_dict["stance_pattern"],
                case_dict.get("metaphor_mode", "structural"),
                case_dict.get("vehicle_spec", "constrained"),
                mapping.get("domain_distance", "near"),
                case_dict.get("risk_domain", "benign"),
                json.dumps(case_dict, ensure_ascii=False, sort_keys=True),
                time.time(),
            ),
        )
        self.conn.commit()

    def generation_exists(self, run_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM generations WHERE run_id=?", (run_id,)).fetchone()
        return row is not None

    def insert_generation(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO generations(
                run_id, case_id, case_hash, stance_pattern, metaphor_mode, vehicle_spec, domain_distance, mapping_visibility, control_arm,
                provider, model, temperature, sample_index, prompt_version,
                generated_text, raw_response, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["run_id"], row["case_id"], row["case_hash"], row["stance_pattern"],
                row.get("metaphor_mode", "structural"), row.get("vehicle_spec", "constrained"), row["domain_distance"],
                row["mapping_visibility"], row["control_arm"], row["provider"], row["model"], row["temperature"],
                row["sample_index"], row["prompt_version"], row["generated_text"], row["raw_response"],
                row.get("created_at", time.time()),
            ),
        )
        self.conn.commit()

    def audit_exists(self, audit_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM audits WHERE audit_id=?", (audit_id,)).fetchone()
        return row is not None

    def insert_audit(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO audits(
                audit_id, run_id, audit_type, judge_provider, judge_model, judge_index,
                prompt_version, raw_response, parsed_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["audit_id"], row["run_id"], row["audit_type"], row["judge_provider"], row["judge_model"],
                row["judge_index"], row["prompt_version"], row["raw_response"],
                json.dumps(row["parsed_json"], ensure_ascii=False, sort_keys=True), row.get("created_at", time.time()),
            ),
        )
        self.conn.commit()

    def quality_pair_exists(self, pair_id: str) -> bool:
        row = self.conn.execute("SELECT 1 FROM quality_pairs WHERE pair_id=?", (pair_id,)).fetchone()
        return row is not None

    def insert_quality_pair(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO quality_pairs(
                pair_id, case_id, case_hash, metaphor_mode, vehicle_spec, mapping_visibility, control_arm, text_a_run_id, text_b_run_id,
                judge_provider, judge_model, judge_index, prompt_version,
                raw_response, parsed_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["pair_id"], row["case_id"], row["case_hash"],
                row.get("metaphor_mode", "structural"), row.get("vehicle_spec", "constrained"),
                row["mapping_visibility"], row["control_arm"],
                row["text_a_run_id"], row["text_b_run_id"], row["judge_provider"], row["judge_model"],
                row["judge_index"], row["prompt_version"], row["raw_response"],
                json.dumps(row["parsed_json"], ensure_ascii=False, sort_keys=True), row.get("created_at", time.time()),
            ),
        )
        self.conn.commit()

    def fetch_generations(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            """
            SELECT * FROM generations
            ORDER BY metaphor_mode, vehicle_spec, case_id, case_hash, mapping_visibility, control_arm, provider, temperature, sample_index
            """
        ))

    def fetch_generation(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM generations WHERE run_id=?", (run_id,)).fetchone()

    def fetch_cases_by_hash(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT case_hash, case_json FROM cases").fetchall()
        return {r["case_hash"]: json.loads(r["case_json"]) for r in rows}

    def fetch_cases(self) -> dict[str, dict[str, Any]]:
        """Compatibility helper: returns latest row per case_id where possible."""
        rows = self.conn.execute("SELECT case_id, case_json, updated_at FROM cases ORDER BY updated_at").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for r in rows:
            out[r["case_id"]] = json.loads(r["case_json"])
        return out

    def fetch_audits(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM audits"))

    def fetch_quality_pairs(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM quality_pairs"))
