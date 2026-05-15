import hashlib
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple
import xml.etree.ElementTree as ET

import boto3
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

load_dotenv(".env")

DB_PATH = "streamlit_poc.db"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".doc", ".docx"}
APP_CONFIG = {
    "hour_tolerance": 0.01,
    "confidence_threshold": 0.8,
    "trusted_streak_threshold": 3,
    "critical_fields": ["employee_name", "vendor", "company"],
    # Used only for pre-call cost preview (output size is unknown until the model responds).
    "llm_assumed_output_tokens": 1200,
}
# Bump this when extraction/autofill mapping logic changes so cached autofill is recalculated.
AUTOFILL_LOGIC_VERSION = "2026-04-17-v3"
FIXED_AWS_REGION = "us-east-1"
FIXED_BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

LLM_PREVIEW_DEFAULT_FOOTNOTE = (
    "Out column shows assumed output tokens from Settings (Assumed output tokens). "
    "For multiple files, the Total row sums input tokens, assumed output tokens, and estimated charge for rows that extracted successfully."
)

# Static column headers for the Step 2 usage table (not configurable in Settings).
LLM_PREVIEW_TABLE_LABELS = {
    "col_in": "In (tokens)",
    "col_out": "Out (est. tokens)",
    "col_rate": "Rate/M (in|out)",
    "col_total": "Total (charge)",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mask_secret(value: str, keep_last: int = 4) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if len(raw) <= keep_last:
        return "*" * len(raw)
    return ("*" * (len(raw) - keep_last)) + raw[-keep_last:]


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        create table if not exists trusted_streaks (
            employee_name text not null,
            vendor text not null,
            company text not null,
            template_hash text not null,
            pattern_key text not null,
            streak integer not null default 0,
            primary key (employee_name, vendor, company, template_hash, pattern_key)
        )
        """
    )
    conn.execute(
        """
        create table if not exists validation_history (
            id integer primary key autoincrement,
            created_at text not null,
            employee_id text,
            employee_name text,
            streak_value integer default 0,
            template_hash text,
            pattern_key text,
            submission_hash text,
            step1_json text not null,
            extracted_json text not null,
            comparison_json text not null,
            decision text not null,
            approval_type text not null,
            reasons_json text not null,
            reason_codes_json text,
            aws_meta_json text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists app_settings (
            key text primary key,
            value text not null
        )
        """
    )
    conn.execute(
        """
        create table if not exists learning_approvals (
            employee_id text not null,
            vendor text not null,
            company text not null,
            template_hash text not null,
            pattern_key text not null,
            mismatch_signature text not null,
            last_answer text not null default '',
            last_question text not null default '',
            streak_count integer not null default 0,
            updated_at text not null,
            primary key (employee_id, vendor, company, template_hash, pattern_key, mismatch_signature)
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("pragma table_info(validation_history)").fetchall()}
    if "employee_name" not in existing_cols:
        conn.execute("alter table validation_history add column employee_name text")
    if "employee_id" not in existing_cols:
        conn.execute("alter table validation_history add column employee_id text")
    if "streak_value" not in existing_cols:
        conn.execute("alter table validation_history add column streak_value integer default 0")
    if "template_hash" not in existing_cols:
        conn.execute("alter table validation_history add column template_hash text")
    if "pattern_key" not in existing_cols:
        conn.execute("alter table validation_history add column pattern_key text")
    if "submission_hash" not in existing_cols:
        conn.execute("alter table validation_history add column submission_hash text")
    if "reason_codes_json" not in existing_cols:
        conn.execute("alter table validation_history add column reason_codes_json text")
    learning_cols = {row[1] for row in conn.execute("pragma table_info(learning_approvals)").fetchall()}
    if "last_question" not in learning_cols:
        conn.execute("alter table learning_approvals add column last_question text default ''")
    conn.execute(
        """
        insert into app_settings(key, value)
        values('trusted_streak_threshold', ?)
        on conflict(key) do nothing
        """,
        (str(APP_CONFIG["trusted_streak_threshold"]),),
    )
    conn.commit()
    conn.close()


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def canonical_person_name(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\s+", " ", raw)
    # Last, First [Middle] -> First [Middle] Last
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) >= 2:
            raw = f"{parts[1]} {parts[0]}"
    raw = re.sub(r"[^A-Za-z\s'-]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return normalize_text(raw)


def format_person_name_display(value: str) -> str:
    canon = canonical_person_name(value)
    if not canon:
        return ""
    return " ".join(tok.capitalize() for tok in canon.split())


def is_probable_person_name(value: str) -> bool:
    v = normalize_text(value or "")
    if not v:
        return False
    tokens = [t for t in re.split(r"[,\s]+", v) if t.strip()]
    if len(tokens) < 2:
        return False
    corporate_markers = {
        "inc",
        "llc",
        "corp",
        "corporation",
        "company",
        "co",
        "solutions",
        "systems",
        "group",
        "technologies",
        "tech",
        "ltd",
    }
    if any(t in corporate_markers for t in tokens):
        return False
    weekday_month_markers = {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }
    if any(t in weekday_month_markers for t in tokens):
        return False
    return True


def has_explicit_approver_context(text: str) -> bool:
    t = normalize_text(text or "")
    if not t:
        return False
    return bool(
        re.search(
            r"\b(approved\s+by|timesheet\s+approver|approver|signature|signed\s+by)\b",
            t,
            flags=re.IGNORECASE,
        )
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            raise
        return json.loads(match.group(0))


def default_normalized(error: str = "") -> Dict[str, Any]:
    payload = {
        "employee_name": "",
        "vendor": "",
        "company": "",
        "duration": "",
        "period_start": "",
        "period_end": "",
        "approved": "",
        "approver_name": "",
        "day_hours": [],
        "total_hours": None,
        "confidence": 0.0,
        "confidence_breakdown": {},
        "headers": [],
        "table_columns": [],
    }
    if error:
        payload["error"] = error
    return payload


def user_friendly_error(error_text: str) -> str:
    text = (error_text or "").strip()
    lowered = text.lower()
    if "unsupporteddocumentexception" in lowered or "unsupported format" in lowered:
        return "Uploaded file is not a supported timesheet document. Please upload a valid PDF/JPG/PNG/TXT/DOC/DOCX timesheet."
    if "password-protected pdf" in lowered:
        return "This PDF is password protected. Please upload an unlocked timesheet file."
    if "corrupted" in lowered or "unreadable" in lowered:
        return "The uploaded file looks unreadable or corrupted. Please upload a clear timesheet file."
    if (
        "expiredtoken" in lowered
        or "accessdenied" in lowered
        or "invalidsignatureexception" in lowered
        or "unrecognizedclientexception" in lowered
        or "unable to locate credentials" in lowered
        or "security token included in the request is invalid" in lowered
    ):
        return "AWS credentials are missing/invalid/expired. Please open Navigation -> Settings and update AWS configuration."
    if "textract_failed" in lowered:
        return "Could not read text from this file. If AWS configuration is set, verify it in Navigation -> Settings, then retry."
    if "bedrock_normalization_failed" in lowered:
        return "The file was read, but Bedrock normalization failed. Please check Bedrock Model ID/AWS credentials in Navigation -> Settings."
    return text or "Unable to process this file. Please upload a valid timesheet document."


def infer_approved_from_text(text: str) -> str:
    raw = (text or "")
    lowered = raw.lower()
    # Explicit negatives => treat as not approved (force manual review later).
    negative_patterns = [
        r"\bnot\s+approved\b",
        r"\bnot\s+approval\b",
        r"\brejected\b",
        r"\bdeclined\b",
        r"\bpending\b",
        r"\bpending\s+approval\b",
    ]
    for pattern in negative_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return ""

    # Placeholder signature labels (without an actual sign/approval event)
    # should not be treated as approved.
    if (
        "for verification purposes only" in lowered
        and ("supervisor/manager signature" in lowered or "manager signature" in lowered)
    ):
        return ""

    # Signature found => approved (as long as not explicitly rejected/pending above).
    signature_patterns = [
        r"\bsignature\b",
        r"\bsigned\b",
        r"\bsign\s*[:\-]?\s*\b",
        r"\bapproved\s+by\b",
    ]
    for pattern in signature_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return "approved"

    approved_patterns = [
        r"\bstatus\s*[:\-]?\s*approved\b",
        r"\bapproved\s+status\b",
        r"\bstatus\b[\s\S]{0,20}\bapproved\b",
        r"\bapproved\b",
    ]
    for pattern in approved_patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return "approved"
    return ""


def infer_approver_name_from_text(text: str) -> str:
    raw = (text or "")
    if not raw.strip():
        return ""

    def clean_name(value: str) -> str:
        v = re.sub(r"\s+", " ", (value or "")).strip(" :,-\t\r\n")
        v = re.sub(r"[^A-Za-z ,.'-]", "", v).strip()
        if len(v) < 3 or len(v) > 60:
            return ""
        lowered = normalize_text(v)
        weekday_words = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
        month_words = {
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        }
        blocked = [
            "approved",
            "approval",
            "status",
            "system generated automatic approval",
            "signature",
            "yes",
            "no",
            "time period end date",
            "week end date",
            "invoice date",
            "contract id",
            "project",
            "total hours",
        ]
        if lowered in blocked or lowered.startswith("approved") or " date" in lowered or "period" in lowered:
            return ""
        tokens = [t for t in re.split(r"[,\s]+", lowered) if t.strip()]
        # Reject date-like strings such as "Monday, January".
        if any(t in weekday_words for t in tokens) or any(t in month_words for t in tokens):
            return ""
        # Keep only human-like names (at least 2 tokens, unless comma format).
        token_count = len([t for t in re.split(r"[,\s]+", v) if t.strip()])
        if token_count < 2:
            return ""
        if not is_probable_person_name(v):
            return ""
        return v

    # Inline pattern: "Approved By: Firstname Lastname"
    m = re.search(r"\bapproved\s*by\s*[:\-]?\s*([A-Za-z][A-Za-z ,.'-]{2,60})", raw, flags=re.IGNORECASE)
    if m:
        nm = clean_name(m.group(1))
        if nm:
            return nm

    # Header + next line pattern from OCR line-by-line output.
    lines = [ln.strip() for ln in raw.splitlines()]
    for i, ln in enumerate(lines):
        if re.search(r"\bapproved\s*by\b", ln, flags=re.IGNORECASE):
            trailing = re.sub(r"(?i)^.*\bapproved\s*by\b\s*[:\-]?\s*", "", ln).strip()
            nm = clean_name(trailing)
            if nm:
                return nm
            for j in range(i + 1, min(i + 4, len(lines))):
                cand = clean_name(lines[j])
                if cand:
                    return cand
    return ""


def infer_company_from_text(text: str) -> str:
    raw = (text or "")
    if not raw.strip():
        return ""
    lowered = normalize_text(raw)
    # Common client label variants seen in timesheet templates.
    if "capital one" in lowered or "capitalone" in lowered:
        return "CapitalOne"
    return ""


def textract_extract(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    cfg = get_runtime_aws_config()
    client_kwargs: Dict[str, Any] = {"region_name": cfg["aws_region"]}
    if cfg.get("aws_access_key_id") and cfg.get("aws_secret_access_key"):
        client_kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        client_kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
        if cfg.get("aws_session_token"):
            client_kwargs["aws_session_token"] = cfg["aws_session_token"]
    client = boto3.client("textract", **client_kwargs)
    resp = client.analyze_document(Document={"Bytes": file_bytes}, FeatureTypes=["FORMS", "TABLES"])
    lines = [b.get("Text", "") for b in resp.get("Blocks", []) if b.get("BlockType") == "LINE"]
    return {"raw": resp, "text": "\n".join(lines), "filename": filename}


def extract_text_from_pdf_local(file_bytes: bytes) -> str:
    """
    Fallback PDF text extractor when Textract AnalyzeDocument rejects a PDF format.
    Uses pypdf text extraction from pages; returns concatenated text.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    out: List[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt.strip():
            out.append(txt)
    return "\n".join(out).strip()


def extract_text_from_txt_local(file_bytes: bytes) -> str:
    for enc in ["utf-8", "utf-16", "latin-1"]:
        try:
            return file_bytes.decode(enc, errors="ignore").strip()
        except Exception:
            continue
    return ""


def extract_text_from_docx_local(file_bytes: bytes) -> str:
    """
    Lightweight DOCX text extraction without adding dependencies.
    Reads word/document.xml and joins text runs.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                return ""
            xml_bytes = zf.read("word/document.xml")
        root = ET.fromstring(xml_bytes)
        text_parts: List[str] = []
        for node in root.iter():
            # DOCX namespaces vary; keep suffix-only matching for robustness.
            if str(node.tag).endswith("}t") and node.text:
                text_parts.append(node.text)
            elif str(node.tag).endswith("}p"):
                text_parts.append("\n")
        return re.sub(r"\n{3,}", "\n\n", "".join(text_parts)).strip()
    except Exception:
        return ""


def extract_text_from_doc_local(file_bytes: bytes) -> str:
    """
    Best-effort extraction for legacy .doc binary files.
    Pulls printable text segments; suitable for local testing files.
    """
    try:
        raw = file_bytes.decode("latin-1", errors="ignore")
    except Exception:
        return ""
    chunks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\s:/,._\-]{3,}", raw)
    cleaned = [c.strip() for c in chunks if len(c.strip()) >= 4]
    return "\n".join(cleaned[:400]).strip()


def prevalidate_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    ext = os.path.splitext((filename or "").lower())[1]
    issues: List[str] = []
    details: Dict[str, Any] = {"supported_format": ext in SUPPORTED_EXTENSIONS}
    if ext not in SUPPORTED_EXTENSIONS:
        issues.append(f"Unsupported format: {ext or 'unknown'}")
        return {"failed": True, "issues": issues, "details": details}

    if ext in {".txt", ".doc", ".docx"}:
        details["document_type"] = ext
        if not file_bytes:
            issues.append("Empty document")
    elif ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            details["pdf_page_count"] = len(reader.pages)
            details["password_protected"] = bool(reader.is_encrypted)
            if reader.is_encrypted:
                # Some PDFs are marked encrypted but are viewable with an empty user password.
                # Treat those as readable so processing can continue.
                try:
                    decrypt_result = reader.decrypt("")
                except Exception:
                    decrypt_result = 0
                details["pdf_decrypt_attempt_empty_password"] = int(decrypt_result) if isinstance(decrypt_result, int) else 0
                if not decrypt_result:
                    issues.append("Password-protected PDF")
        except Exception as exc:
            issues.append(f"Corrupted or unreadable PDF: {exc}")
    else:
        try:
            img = Image.open(io.BytesIO(file_bytes))
            img.verify()
            details["image_format"] = img.format
        except UnidentifiedImageError:
            issues.append("Corrupted or unreadable image")
        except Exception as exc:
            issues.append(f"Unreadable image: {exc}")

    return {"failed": len(issues) > 0, "issues": issues, "details": details}


def validate_image_quality(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    ext = os.path.splitext((filename or "").lower())[1]
    if ext not in {".png", ".jpg", ".jpeg"}:
        return {"issues": [], "metrics": {}, "flagged": False}
    issues: List[str] = []
    metrics: Dict[str, Any] = {}
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("L")
        width, height = img.size
        metrics["width"] = width
        metrics["height"] = height
        # Low-resolution check
        if width < 800 or height < 600:
            issues.append("Low-resolution image")
        # Basic blur heuristic using grayscale gradient energy
        pixels = list(img.getdata())
        if pixels:
            # Lightweight rough sharpness estimate without extra dependencies
            # (higher is sharper)
            diffs = []
            row_len = width
            for y in range(1, height):
                base = y * row_len
                prev = (y - 1) * row_len
                for x in range(1, width):
                    idx = base + x
                    diffs.append(abs(pixels[idx] - pixels[idx - 1]) + abs(pixels[idx] - pixels[prev + x]))
            sharpness = sum(diffs) / max(len(diffs), 1)
            metrics["sharpness_score"] = round(sharpness, 2)
            if sharpness < 8:
                issues.append("Blurry image")
        # Cropped/cut-off heuristic: extreme aspect ratio or very small edge margins
        aspect_ratio = width / max(height, 1)
        metrics["aspect_ratio"] = round(aspect_ratio, 3)
        if aspect_ratio > 2.5 or aspect_ratio < 0.35:
            issues.append("Possible cropped or partial page")
    except Exception as exc:
        issues.append(f"Image quality check failed: {exc}")
    return {"issues": issues, "metrics": metrics, "flagged": len(issues) > 0}


def build_confidence_breakdown(textract_raw: Dict[str, Any], normalized: Dict[str, Any]) -> Dict[str, Any]:
    blocks = textract_raw.get("Blocks", []) if isinstance(textract_raw, dict) else []
    line_blocks = [b for b in blocks if b.get("BlockType") == "LINE"]
    cell_blocks = [b for b in blocks if b.get("BlockType") == "CELL"]
    field_conf: Dict[str, float] = {}

    def match_conf(value: str) -> float:
        needle = normalize_text(value)
        if not needle:
            return 0.0
        best = 0.0
        for line in line_blocks:
            text = normalize_text(line.get("Text", ""))
            if needle and needle in text:
                best = max(best, safe_float(line.get("Confidence", 0.0), 0.0))
        return best

    for field in ["employee_name", "vendor", "company", "period_start", "period_end", "approved"]:
        field_conf[field] = match_conf(str(normalized.get(field, "")))

    # Cell confidence by row (for day-wise table confidence)
    row_conf_map: Dict[int, List[float]] = {}
    for cell in cell_blocks:
        row_idx = int(cell.get("RowIndex", 0))
        row_conf_map.setdefault(row_idx, []).append(safe_float(cell.get("Confidence", 0.0), 0.0))
    row_confidence = {
        str(row): round(sum(vals) / max(len(vals), 1), 2) for row, vals in row_conf_map.items() if row > 0
    }

    critical_vals = [v for k, v in field_conf.items() if k in {"employee_name", "vendor", "company"} and v > 0]
    critical_conf = round(sum(critical_vals) / max(len(critical_vals), 1), 2) if critical_vals else 0.0
    day_conf_vals = list(row_confidence.values())
    day_conf = round(sum(day_conf_vals) / max(len(day_conf_vals), 1), 2) if day_conf_vals else 0.0
    aggregate = round((critical_conf * 0.6) + (day_conf * 0.4), 2) if (critical_conf or day_conf) else 0.0
    return {
        "field_confidence": field_conf,
        "table_row_confidence": row_confidence,
        "critical_fields_confidence": critical_conf,
        "day_hours_confidence": day_conf,
        "aggregate_confidence": aggregate,
    }


def build_bedrock_normalization_prompt(text: str, hints: Dict[str, Any] | None = None) -> str:
    """Full user prompt sent to Bedrock for timesheet JSON normalization (must stay in sync with normalize_with_bedrock)."""
    hints = hints or {}
    hint_duration = hints.get("duration", "")
    hint_period_start = hints.get("period_start", "")
    hint_period_end = hints.get("period_end", "")
    return f"""
Extract timesheet fields and return JSON only.
Keys: employee_name, vendor, company, duration, period_start, period_end, approved, approver_name, day_hours, total_hours, confidence, headers, table_columns.
day_hours must be [{"{"}"date":"YYYY-MM-DD","hours":number{"}"}].
approved should contain text from the timesheet approval/signature area, e.g. yes/approved/signature, if present.
approver_name should contain signer/approver readable name when present, else empty.
Important:
- Map hours to the exact date headers from the timesheet table (no left/right shifting).
- Keep zero values exactly as shown.
- Use hints only if the document doesn't clearly specify duration/period.
Hints:
- expected_duration: {hint_duration}
- expected_period_start: {hint_period_start}
- expected_period_end: {hint_period_end}
Use empty string/array for unknown values.
Text:
{text}
"""


def count_prompt_tokens_estimate(prompt: str) -> int:
    """Approximate tokenizer count (Claude-style text; not an official Anthropic meter)."""
    raw = prompt or ""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(raw)))
    except Exception:
        return max(1, len(raw.encode("utf-8")) // 4)


def bedrock_claude_price_per_million_usd(model_id: str) -> Tuple[float, float, str]:
    """
    On-demand USD per 1M tokens (input, output) for common Bedrock Anthropic IDs.
    Match longer/more specific substrings first. Unknown models use Sonnet-like defaults.
    """
    mid = (model_id or "").lower()
    ordered: List[Tuple[str, float, float]] = [
        ("anthropic.claude-3-opus", 15.0, 75.0),
        ("claude-3-opus", 15.0, 75.0),
        ("anthropic.claude-3-5-sonnet", 3.0, 15.0),
        ("claude-3-5-sonnet", 3.0, 15.0),
        ("anthropic.claude-3-sonnet", 3.0, 15.0),
        ("claude-3-sonnet", 3.0, 15.0),
        ("anthropic.claude-3-haiku", 0.25, 1.25),
        ("claude-3-haiku", 0.25, 1.25),
        ("anthropic.claude-sonnet-4", 3.0, 15.0),
        ("anthropic.claude-3-7-sonnet", 3.0, 15.0),
    ]
    for needle, inp_m, out_m in ordered:
        if needle in mid:
            return inp_m, out_m, needle
    return 3.0, 15.0, "default_pricing_profile"


def resolve_llm_unit_pricing(model_id: str) -> Tuple[float, float, str, str]:
    """
    Returns (input_per_million_tokens, output_per_million_tokens, profile_label, charge_unit).
    charge_unit is \"USD\" (public table or custom dollars per 1M) or \"credits\" (custom internal credits per 1M).
    """
    mode = normalize_text(get_setting_text("llm_pricing_mode", "auto")).lower() or "auto"
    if mode in {"custom_usd", "custom", "usd"}:
        inp = get_setting_float("llm_custom_input_usd_per_mtok", 3.0)
        out = get_setting_float("llm_custom_output_usd_per_mtok", 15.0)
        return max(0.0, inp), max(0.0, out), "custom_usd_per_mtok", "USD"
    if mode in {"custom_credits", "credits", "credit"}:
        inp = get_setting_float("llm_custom_input_credits_per_mtok", 100.0)
        out = get_setting_float("llm_custom_output_credits_per_mtok", 500.0)
        return max(0.0, inp), max(0.0, out), "custom_credits_per_mtok", "credits"
    in_per_m, out_per_m, profile = bedrock_claude_price_per_million_usd(model_id)
    return in_per_m, out_per_m, profile, "USD"


def estimate_bedrock_llm_cost(
    model_id: str, prompt: str, assumed_output_tokens: int | None = None
) -> Dict[str, Any]:
    assumed_out = int(assumed_output_tokens if assumed_output_tokens is not None else APP_CONFIG["llm_assumed_output_tokens"])
    assumed_out = max(0, assumed_out)
    input_tok = count_prompt_tokens_estimate(prompt)
    in_per_m, out_per_m, profile, unit = resolve_llm_unit_pricing(model_id)
    in_cost = input_tok * in_per_m / 1_000_000.0
    out_cost = assumed_out * out_per_m / 1_000_000.0
    total_c = in_cost + out_cost
    note_auto = (
        "Approximate charges using the built-in On-Demand style table for the selected model ID family; "
        "tokens are estimated (tiktoken / byte fallback), not provider meters."
    )
    note_custom_usd = "Charges use your custom USD amounts per 1 million input/output tokens from Settings."
    note_custom_cr = "Charges use your custom internal credits per 1 million input/output tokens from Settings."
    if profile == "custom_usd_per_mtok":
        note = note_custom_usd
    elif profile == "custom_credits_per_mtok":
        note = note_custom_cr
    else:
        note = note_auto
    out: Dict[str, Any] = {
        "model_id": model_id,
        "pricing_profile": profile,
        "pricing_mode_resolved": get_setting_text("llm_pricing_mode", "auto").strip() or "auto",
        "charge_unit": unit,
        "input_tokens_est": input_tok,
        "output_tokens_assumed": assumed_out,
        "input_rate_per_mtok": in_per_m,
        "output_rate_per_mtok": out_per_m,
        "input_charge_est": round(in_cost, 6),
        "output_charge_est": round(out_cost, 6),
        "total_charge_est": round(total_c, 6),
        "input_usd_est": round(in_cost, 6) if unit == "USD" else 0.0,
        "output_usd_est": round(out_cost, 6) if unit == "USD" else 0.0,
        "total_usd_est": round(total_c, 6) if unit == "USD" else 0.0,
        "usd_per_mtok_input": in_per_m if unit == "USD" else None,
        "usd_per_mtok_output": out_per_m if unit == "USD" else None,
        "input_credits_est": round(in_cost, 6) if unit == "credits" else None,
        "output_credits_est": round(out_cost, 6) if unit == "credits" else None,
        "total_credits_est": round(total_c, 6) if unit == "credits" else None,
        "credits_per_mtok_input": in_per_m if unit == "credits" else None,
        "credits_per_mtok_output": out_per_m if unit == "credits" else None,
        "pricing_note": note,
    }
    return out


def _streamlit_script_active() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _merge_usage_from_llm_response(cost_preview: Dict[str, Any], response_metadata: Any) -> None:
    if not isinstance(cost_preview, dict) or not isinstance(response_metadata, dict):
        return
    usage = response_metadata.get("usage")
    if not isinstance(usage, dict):
        usage = response_metadata
    inp = usage.get("input_tokens")
    if inp is None:
        inp = usage.get("prompt_tokens")
    out = usage.get("output_tokens")
    if out is None:
        out = usage.get("completion_tokens")
    if inp is None and isinstance(response_metadata.get("input_tokens"), int):
        inp = response_metadata.get("input_tokens")
    if out is None and isinstance(response_metadata.get("output_tokens"), int):
        out = response_metadata.get("output_tokens")
    if inp is not None:
        try:
            cost_preview["actual_input_tokens"] = int(inp)
        except (TypeError, ValueError):
            pass
    if out is not None:
        try:
            cost_preview["actual_output_tokens"] = int(out)
        except (TypeError, ValueError):
            pass
    model_id = str(cost_preview.get("model_id", "") or "")
    in_per_m, out_per_m, _, unit = resolve_llm_unit_pricing(model_id)
    ai = cost_preview.get("actual_input_tokens")
    ao = cost_preview.get("actual_output_tokens")
    if isinstance(ai, int) and isinstance(ao, int):
        total_actual = ai * in_per_m / 1_000_000.0 + ao * out_per_m / 1_000_000.0
        cost_preview["actual_total_charge_est"] = round(total_actual, 6)
        if unit == "USD":
            cost_preview["actual_total_usd_est"] = round(total_actual, 6)
        else:
            cost_preview["actual_total_credits_est"] = round(total_actual, 6)


def normalize_with_bedrock(
    text: str,
    hints: Dict[str, Any] | None = None,
    *,
    source_label: str = "",
    cost_preview_holder: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = get_runtime_aws_config()
    model_id = cfg["bedrock_model_id"]
    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID missing. Set it in .env")
    client_kwargs: Dict[str, Any] = {"region_name": cfg["aws_region"]}
    if cfg.get("aws_access_key_id") and cfg.get("aws_secret_access_key"):
        client_kwargs["aws_access_key_id"] = cfg["aws_access_key_id"]
        client_kwargs["aws_secret_access_key"] = cfg["aws_secret_access_key"]
        if cfg.get("aws_session_token"):
            client_kwargs["aws_session_token"] = cfg["aws_session_token"]
    bedrock_runtime_client = boto3.client("bedrock-runtime", **client_kwargs)
    llm = ChatBedrock(
        model_id=model_id,
        region_name=cfg["aws_region"],
        client=bedrock_runtime_client,
        model_kwargs={"temperature": 0},
    )
    prompt = build_bedrock_normalization_prompt(text, hints)
    preview = estimate_bedrock_llm_cost(model_id, prompt)
    if cost_preview_holder is not None:
        cost_preview_holder.clear()
        cost_preview_holder.update(preview)
    msg = llm.invoke(prompt)
    if cost_preview_holder is not None:
        _merge_usage_from_llm_response(cost_preview_holder, getattr(msg, "response_metadata", None) or {})
    out = msg.content
    if isinstance(out, list):
        out = "".join(str(x) for x in out)
    return extract_json_from_text(str(out))


MONTH_NAME_TO_NUM = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def _parse_date_any(value: str, default_year: int | None = None) -> date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    raw = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    raw = raw.replace("\\", "/").replace(".", "/")
    raw = re.sub(r"\s+", " ", raw)

    # ISO / YYYY/MM/DD
    m = re.match(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$", raw)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    # MM/DD/YYYY or DD/MM/YYYY (we'll treat first as month if > 12 forces day/month)
    m = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$", raw)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if a > 12 and b <= 12:
            d, mo = a, b
        elif b > 12 and a <= 12:
            mo, d = a, b
        else:
            mo, d = a, b
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    # "Dec 25 2025" or "Dec 25"
    m = re.match(r"^\s*([A-Za-z]{3,9})\s+(\d{1,2})(?:\s+(\d{4}))?\s*$", raw)
    if m:
        mo = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if not mo:
            return None
        d = int(m.group(2))
        y = int(m.group(3)) if m.group(3) else default_year
        if not y:
            return None
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    # "25 Dec 2025" or "25 Dec"
    m = re.match(r"^\s*(\d{1,2})\s+([A-Za-z]{3,9})(?:\s+(\d{4}))?\s*$", raw)
    if m:
        d = int(m.group(1))
        mo = MONTH_NAME_TO_NUM.get(m.group(2).lower())
        if not mo:
            return None
        y = int(m.group(3)) if m.group(3) else default_year
        if not y:
            return None
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    return None


def extract_explicit_dates_from_text(text: str) -> set[str]:
    raw = str(text or "")
    if not raw.strip():
        return set()
    out: set[str] = set()
    patterns = [
        r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2}(?:\s+\d{4})?\b",
        r"\b\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{4})?\b",
    ]
    for pat in patterns:
        for token in re.findall(pat, raw):
            dt = _parse_date_any(str(token))
            if dt:
                out.add(dt.isoformat())
    return out


def _extract_candidate_periods_from_text(text: str, default_year: int | None) -> List[Tuple[date, date]]:
    t = (text or "")
    if not t:
        return []
    t = re.sub(r"\s+", " ", t)
    candidates: List[Tuple[date, date]] = []

    # YYYY/MM/DD - YYYY/MM/DD
    for m in re.finditer(
        r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})\s*(?:to|through|thru|[-–—])\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
        t,
        flags=re.IGNORECASE,
    ):
        a = _parse_date_any(m.group(1), default_year)
        b = _parse_date_any(m.group(2), default_year)
        if a and b:
            candidates.append((min(a, b), max(a, b)))

    # Dec 25 2025 - Jan 4 2026 (years optional)
    month_names = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    for m in re.finditer(
        rf"({month_names})\s*(\d{{1,2}})(?:\s*(\d{{4}}))?\s*(?:to|through|thru|[-–—])\s*({month_names})\s*(\d{{1,2}})(?:\s*(\d{{4}}))?",
        t,
        flags=re.IGNORECASE,
    ):
        left = f"{m.group(1)} {m.group(2)} {m.group(3) or ''}".strip()
        right = f"{m.group(4)} {m.group(5)} {m.group(6) or ''}".strip()
        a = _parse_date_any(left, default_year)
        b = _parse_date_any(right, default_year)
        if a and b:
            candidates.append((min(a, b), max(a, b)))

    # Dec 25 - 4 (implicit month on right; infer month/year from left and default_year)
    for m in re.finditer(
        rf"({month_names})\s*(\d{{1,2}})\s*(?:to|through|thru|[-–—])\s*(\d{{1,2}})\b",
        t,
        flags=re.IGNORECASE,
    ):
        mo = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        if not mo or not default_year:
            continue
        d1 = int(m.group(2))
        d2 = int(m.group(3))
        a = _parse_date_any(f"{m.group(1)} {d1} {default_year}", default_year)
        if not a:
            continue
        # Assume right side is same month unless it wraps (e.g., 29-4 suggests next month)
        right_year = default_year
        right_month = mo
        if d2 < d1:
            right_month = mo + 1
            if right_month == 13:
                right_month = 1
                right_year += 1
        try:
            b = date(right_year, right_month, d2)
        except ValueError:
            continue
        candidates.append((min(a, b), max(a, b)))

    return candidates


def _infer_month_year_context_from_text(text: str) -> Tuple[Tuple[int, int] | None, Tuple[int, int] | None]:
    """
    Infer a (month, year) context from strings like:
    - "Dec 2025 - Jan 2026"
    - "Dec 25 - Jan 26"  (interpreted as month-year with 2-digit years when near month names)
    Returns ((start_month, start_year), (end_month, end_year)) or (None, None).
    """
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return None, None
    month_names = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

    def parse_year_token(tok: str) -> int | None:
        tok = (tok or "").strip()
        if not tok:
            return None
        if re.fullmatch(r"\d{4}", tok):
            return int(tok)
        if re.fullmatch(r"\d{2}", tok):
            # Use a reasonable pivot: 00-79 => 2000s, 80-99 => 1900s
            yy = int(tok)
            return 2000 + yy if yy <= 79 else 1900 + yy
        return None

    # Pattern: Dec 2025 - Jan 2026 (or with short year)
    m = re.search(
        rf"\b{month_names}\b\s+(\d{{2,4}})\s*(?:to|through|thru|[-–—])\s*\b{month_names}\b\s+(\d{{2,4}})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        m1 = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        y1 = parse_year_token(m.group(2))
        m2 = MONTH_NAME_TO_NUM.get(m.group(3).lower())
        y2 = parse_year_token(m.group(4))
        if m1 and y1 and m2 and y2:
            return (m1, y1), (m2, y2)

    # Pattern: Dec 25 - Jan 26 (ambiguous; treat as month-year if both are 2-digit and look like years)
    m = re.search(
        rf"\b{month_names}\b\s+(\d{{1,2}})\s*(?:to|through|thru|[-–—])\s*\b{month_names}\b\s+(\d{{1,2}})\b",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        m1 = MONTH_NAME_TO_NUM.get(m.group(1).lower())
        a = m.group(2)
        m2 = MONTH_NAME_TO_NUM.get(m.group(3).lower())
        b = m.group(4)
        # Heuristic: if both numbers are >= 20 and <= 99, they're likely years in UI banners.
        if m1 and m2 and a.isdigit() and b.isdigit():
            aa = int(a)
            bb = int(b)
            if 20 <= aa <= 99 and 20 <= bb <= 99:
                y1 = 2000 + aa if aa <= 79 else 1900 + aa
                y2 = 2000 + bb if bb <= 79 else 1900 + bb
                # If months wrap (Dec -> Jan) and years are same, adjust start year.
                if m1 > m2 and y1 == y2:
                    y1 -= 1
                return (m1, y1), (m2, y2)

    return None, None


def _infer_week_window_from_text(text: str, month_year_ctx: Tuple[Tuple[int, int] | None, Tuple[int, int] | None]) -> Tuple[date | None, date | None]:
    """
    Infer the displayed week window like "29 - 4" using month/year context like "Dec 25 - Jan 26".
    Returns (start_date, end_date) if possible.
    """
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return None, None
    start_ctx, end_ctx = month_year_ctx
    if not start_ctx or not end_ctx:
        return None, None
    (start_month, start_year), (end_month, end_year) = start_ctx, end_ctx

    # Find a likely "day range" token (e.g., "29 - 4") - choose the first reasonable one.
    m = re.search(r"\b([0-3]?\d)\s*[-–—]\s*([0-3]?\d)\b", t)
    if not m:
        return None, None
    d1, d2 = int(m.group(1)), int(m.group(2))
    if not (1 <= d1 <= 31 and 1 <= d2 <= 31):
        return None, None

    # If it wraps (29 -> 4) then end is in end_ctx month/year; start is in start_ctx.
    wraps = d2 < d1
    try:
        start_date = date(start_year, start_month, d1)
    except ValueError:
        start_date = None
    try:
        end_date = date(end_year if wraps else start_year, end_month if wraps else start_month, d2)
    except ValueError:
        end_date = None

    if start_date and end_date:
        return start_date, end_date
    return None, None


def _infer_period_from_extraction(normalized: Dict[str, Any], ocr_text: str) -> Tuple[str, str]:
    # Use existing normalized dates if valid ISO.
    existing_start = parse_iso_date_optional(str(normalized.get("period_start", "")))
    existing_end = parse_iso_date_optional(str(normalized.get("period_end", "")))

    # Infer default year from extracted day_hours (already ISO most of the time).
    extracted_dates: List[date] = []
    for row in normalized.get("day_hours", []) or []:
        d = _parse_date_any(str(row.get("date", "")), None) or parse_iso_date_optional(str(row.get("date", "")))
        if d:
            extracted_dates.append(d)
    default_year = None
    if extracted_dates:
        # Prefer year of earliest date in table.
        default_year = min(extracted_dates).year
    elif existing_start:
        default_year = existing_start.year
    elif existing_end:
        default_year = existing_end.year

    # Candidate period ranges from OCR text.
    candidates = _extract_candidate_periods_from_text(ocr_text, default_year)
    # Also infer from month-year banner + week day range ("Dec 25 - Jan 26" + "29 - 4").
    # This is a strong signal in UI-based timesheets (often the only place the year is shown),
    # so it must take priority over potentially-wrong LLM years in day_hours.
    month_year_ctx = _infer_month_year_context_from_text(ocr_text)
    week_start, week_end = _infer_week_window_from_text(ocr_text, month_year_ctx)
    if week_start and week_end:
        return min(week_start, week_end).isoformat(), max(week_start, week_end).isoformat()

    if extracted_dates:
        min_dt = min(extracted_dates)
        max_dt = max(extracted_dates)
        covering = [c for c in candidates if c[0] <= min_dt and c[1] >= max_dt]
        if covering:
            best = sorted(covering, key=lambda x: (x[1] - x[0]).days)[0]
            return best[0].isoformat(), best[1].isoformat()
        # Fallback: use min/max extracted dates.
        return min_dt.isoformat(), max_dt.isoformat()

    # If no day_hours, fall back to best candidate or existing.
    if candidates:
        best = sorted(candidates, key=lambda x: (x[1] - x[0]).days)[0]
        return best[0].isoformat(), best[1].isoformat()

    return (existing_start.isoformat() if existing_start else ""), (existing_end.isoformat() if existing_end else "")


def _normalize_day_hours_dates_inplace(normalized: Dict[str, Any], default_year: int | None) -> None:
    rows = normalized.get("day_hours", [])
    if not isinstance(rows, list):
        return
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = str(row.get("date", "")).strip()
        raw_hours = row.get("hours")
        d = parse_iso_date_optional(raw_date) or _parse_date_any(raw_date, default_year)
        if not d:
            continue
        out.append({"date": d.isoformat(), "hours": safe_float(raw_hours, 0.0)})
    normalized["day_hours"] = out


def _align_day_hours_years_to_period(normalized: Dict[str, Any]) -> None:
    """
    If day_hours years are clearly off (e.g., 2022) but month/day match a known period,
    rewrite them to fit within period_start/period_end window.
    """
    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or ps > pe:
        return
    rows = normalized.get("day_hours", [])
    if not isinstance(rows, list) or not rows:
        return
    fixed: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        d = parse_iso_date_optional(str(row.get("date", "")))
        if not d:
            continue
        # If already in range, keep.
        if ps <= d <= pe:
            fixed.append({"date": d.isoformat(), "hours": safe_float(row.get("hours", 0.0), 0.0)})
            continue
        # Try same month/day with period years.
        candidates = []
        for y in {ps.year, pe.year}:
            try:
                candidates.append(date(y, d.month, d.day))
            except ValueError:
                continue
        chosen = None
        for cd in candidates:
            if ps <= cd <= pe:
                chosen = cd
                break
        if chosen:
            fixed.append({"date": chosen.isoformat(), "hours": safe_float(row.get("hours", 0.0), 0.0)})
        else:
            fixed.append({"date": d.isoformat(), "hours": safe_float(row.get("hours", 0.0), 0.0)})
    normalized["day_hours"] = fixed


def _extract_day_number_from_cell_text(text: str) -> int | None:
    t = normalize_text(text).replace("mon", "").replace("tue", "").replace("wed", "").replace("thu", "").replace("fri", "").replace("sat", "").replace("sun", "")
    # Pull the first plausible day-of-month number.
    m = re.search(r"\b([1-9]|[12]\d|3[01])\b", t)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_float_from_cell_text(text: str) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    # Common hour formats: "8", "8.0", "8.00", sometimes with trailing units.
    m = re.search(r"(-?\d+(?:\.\d+)?)", raw.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _fallback_extract_day_hours_from_textract(textract_raw: Dict[str, Any], ocr_text: str) -> List[Dict[str, Any]]:
    """
    Fallback extractor used when Bedrock normalization fails.
    It attempts to reconstruct the day-wise hours table from Textract TABLE/CELL blocks.
    """
    if not isinstance(textract_raw, dict):
        return []
    blocks = textract_raw.get("Blocks", [])
    if not isinstance(blocks, list):
        return []

    cell_blocks = [b for b in blocks if isinstance(b, dict) and b.get("BlockType") == "CELL"]
    if not cell_blocks:
        return []

    # Build row/col->cell text map for quick scanning.
    row_to_daynums: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # row -> [(col, day_num)]
    row_has_total: Dict[int, bool] = defaultdict(bool)
    cells_by_row: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for c in cell_blocks:
        row_idx = int(c.get("RowIndex", 0) or 0)
        col_idx = int(c.get("ColumnIndex", 0) or 0)
        txt = str(c.get("Text", "") or "")
        if not row_idx or not col_idx:
            continue
        cells_by_row[row_idx].append(c)

        if re.search(r"\btotal\b", txt, flags=re.IGNORECASE):
            row_has_total[row_idx] = True

        dn = _extract_day_number_from_cell_text(txt)
        if dn:
            row_to_daynums[row_idx].append((col_idx, dn))

    # Pick the "header row" that contains the most distinct day-number cells.
    best_header_row = None
    best_header_cnt = 0
    for r, pairs in row_to_daynums.items():
        distinct_days = {d for _, d in pairs}
        if len(distinct_days) > best_header_cnt and len(distinct_days) >= 4:
            best_header_cnt = len(distinct_days)
            best_header_row = r

    if best_header_row is None:
        return []

    header_pairs = sorted(row_to_daynums.get(best_header_row, []), key=lambda x: x[0])
    if len(header_pairs) < 4:
        return []

    # Infer the correct period using OCR week/month banners; this fixes year mismatches.
    tmp_norm = {"day_hours": [], "period_start": "", "period_end": ""}
    ps_s, pe_s = _infer_period_from_extraction(tmp_norm, ocr_text)
    ps = parse_iso_date_optional(ps_s)
    pe = parse_iso_date_optional(pe_s)
    if not ps or not pe or ps > pe:
        return []

    window_dates = date_range(ps, pe)
    window_seq = [d.day for d in window_dates]

    extracted_seq = [dn for _, dn in header_pairs]
    extracted_cols = [col for col, _ in header_pairs]

    # Align extracted day-number sequence to the inferred window by order.
    col_to_date: Dict[int, date] = {}
    matched = False
    for start_offset in range(0, max(1, len(window_seq) - len(extracted_seq) + 1)):
        if window_seq[start_offset : start_offset + len(extracted_seq)] == extracted_seq:
            for i, col in enumerate(extracted_cols):
                idx = start_offset + i
                if idx < len(window_dates):
                    col_to_date[col] = window_dates[idx]
            matched = True
            break

    if not matched:
        # Last resort: map by position to the start of window.
        m = min(len(extracted_cols), len(window_dates))
        for i in range(m):
            col_to_date[extracted_cols[i]] = window_dates[i]

    if len(col_to_date) < 3:
        return []

    totals_by_col: Dict[int, float] = defaultdict(float)

    for c in cell_blocks:
        row_idx = int(c.get("RowIndex", 0) or 0)
        col_idx = int(c.get("ColumnIndex", 0) or 0)
        if not row_idx or not col_idx:
            continue
        if row_idx == best_header_row:
            continue
        if row_has_total.get(row_idx, False):
            continue
        if col_idx not in col_to_date:
            continue
        txt = str(c.get("Text", "") or "")
        val = _extract_float_from_cell_text(txt)
        if val is None:
            continue
        totals_by_col[col_idx] += val

    day_hours: List[Dict[str, Any]] = []
    for col_idx, dt in col_to_date.items():
        day_hours.append({"date": dt.isoformat(), "hours": round(float(totals_by_col.get(col_idx, 0.0)), 2)})
    day_hours.sort(key=lambda x: x["date"])
    return day_hours


_WEEKDAY_TO_NUM = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "mon": 0,
    "tue": 1,
    "wed": 2,
    "thu": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}


def _parse_slash_date_ambiguous(value: str, prefer_us: bool = True) -> date | None:
    """
    Parse dates like 01/04/2026 where month/day can be ambiguous.
    When prefer_us=True, treat as MM/DD/YYYY when both parts <= 12.
    """
    raw = (value or "").strip()
    m = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$", raw)
    if not m:
        return None
    a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if a > 12:
        d, mo = a, b
    elif b > 12:
        mo, d = a, b
    else:
        if prefer_us:
            mo, d = a, b
        else:
            d, mo = a, b
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def _extract_ending_date_from_ocr(ocr_text: str) -> date | None:
    t = ocr_text or ""
    m = re.search(
        r"(?i)\bending\s+date\s*[:#]?\s*(\d{1,2}/\d{1,2}/\d{4})\b",
        t,
    )
    if not m:
        return None
    return _parse_slash_date_ambiguous(m.group(1), prefer_us=True)


def _extract_weekday_hours_from_ocr_block(ocr_text: str, week_start_monday: date) -> Dict[int, float]:
    """
    Parse a simple Mon-Sun block with a trailing 'X.XXh' subtotal per day (see capture.jpg OCR).
    Returns weekday_index (Mon=0) -> hours.
    """
    lines = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
    hours_by_wd: Dict[int, float] = {}
    i = 0
    while i < len(lines):
        wd = lines[i].strip().lower()
        if wd not in _WEEKDAY_TO_NUM:
            i += 1
            continue
        # Look ahead within a small window for an hours token like '8.00h' or '0.00h'
        # Include the weekday line itself; some OCR layouts put "8.00h" on the same line as the weekday.
        window = lines[i : i + 12]
        candidates: List[float] = []
        for ln in window:
            ln_l = ln.strip().lower()
            if ln_l in _WEEKDAY_TO_NUM and ln_l != wd:
                break
            # Only trust explicit hour totals like "8.00h" / "0.00h" (avoid "08 00" time tokens and unrelated numbers).
            # Strict subtotal pattern (avoid matching "7.00" inside "17:00")
            for m in re.finditer(r"\b(\d{1,2}\.\d{2})\s*h\b", ln, flags=re.IGNORECASE):
                val = safe_float(m.group(1), None)
                if val is None:
                    continue
                if val < 0 or val > 24:
                    continue
                candidates.append(float(val))
        best_h: float | None = None
        if candidates:
            # OCR sometimes emits a stray 0.00h before the real subtotal; prefer the last non-zero if any.
            non_zero = [c for c in candidates if c > 0]
            plausible = [c for c in candidates if c <= 16]
            if plausible:
                non_zero_plausible = [c for c in plausible if c > 0]
                best_h = non_zero_plausible[-1] if non_zero_plausible else plausible[-1]
            else:
                best_h = non_zero[-1] if non_zero else candidates[-1]
        if best_h is not None:
            hours_by_wd[_WEEKDAY_TO_NUM[wd]] = float(best_h)
        i += 1
    return hours_by_wd


def _apply_ending_date_week_anchor(normalized: Dict[str, Any], ocr_text: str) -> None:
    """
    When OCR shows 'Ending Date: MM/DD/YYYY' and a weekday hour grid, anchor the week ending on that date
    (typically week-ending Sunday for Mon-Sun grids) and rebuild day_hours + period.
    Overrides clearly wrong LLM years (e.g. 2023) when anchor is far from extracted dates.
    """
    anchor = _extract_ending_date_from_ocr(ocr_text)
    if not anchor:
        return

    # Scope narrowly to "day rows + subtotal hours" style captures (like capture.jpg) to avoid surprising other templates.
    lowered = (ocr_text or "").lower()
    if "subtotal hours" not in lowered:
        return
    if "monday" not in lowered or "sunday" not in lowered:
        return

    rows = normalized.get("day_hours", []) if isinstance(normalized.get("day_hours"), list) else []
    parsed: List[date] = []
    for row in rows:
        if isinstance(row, dict):
            d = parse_iso_date_optional(str(row.get("date", "")))
            if d:
                parsed.append(d)
    if parsed:
        min_d, max_d = min(parsed), max(parsed)
        year_mismatch = anchor.year not in {d.year for d in parsed}
        week_mismatch = abs((max_d - anchor).days) > 7 or abs((min_d - anchor).days) > 7
        far_year = abs((min_d - anchor).days) > 400 or abs((max_d - anchor).days) > 400
        if not (year_mismatch or week_mismatch or far_year):
            return

    hours_by_wd = _extract_weekday_hours_from_ocr_block(ocr_text, week_start_monday=anchor)
    if len(hours_by_wd) < 5:
        return

    # Week window ending on Ending Date: Mon..Sun style grids => Monday is end - 6 days.
    monday = anchor - timedelta(days=6)

    day_hours: List[Dict[str, Any]] = []
    for wd in range(7):
        dt = monday + timedelta(days=wd)
        day_hours.append({"date": dt.isoformat(), "hours": round(float(hours_by_wd.get(wd, 0.0)), 2)})
    day_hours.sort(key=lambda x: x["date"])
    normalized["day_hours"] = day_hours
    normalized["period_start"] = monday.isoformat()
    normalized["period_end"] = anchor.isoformat()
    normalized["total_hours"] = round(sum(safe_float(x.get("hours", 0.0), 0.0) for x in day_hours), 2)


def _extract_project_grid_daily_totals_from_text(normalized: Dict[str, Any], ocr_text: str) -> List[float]:
    """
    For project/task style grids, choose the best candidate daily-total row from OCR text.
    Scoped heuristic to avoid changing unrelated templates.
    """
    headers_text = " ".join(str(x) for x in (normalized.get("headers") or []))
    columns_text = " ".join(str(x) for x in (normalized.get("table_columns") or []))
    scope_text = f"{headers_text} {columns_text}".lower()
    if "project" not in scope_text:
        return []
    if "task" not in scope_text and "description" not in scope_text:
        return []

    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or ps > pe:
        return []

    period_dates = date_range(ps, pe)
    # Typical work-week counts in these reports (Mon-Fri, sometimes +Sat/+Sun)
    expected_lengths = {5, 6, 7, len(period_dates)}
    lines = [ln.strip() for ln in (ocr_text or "").splitlines() if ln.strip()]
    non_work_markers = [
        "unpaid",
        "time off",
        "pto",
        "leave",
        "holiday",
    ]
    def is_non_work_line(text: str) -> bool:
        t = normalize_text(text or "")
        return any(m in t for m in non_work_markers)
    candidates: List[List[float]] = []

    for ln in lines:
        if is_non_work_line(ln):
            continue
        vals = [safe_float(m.group(1), None) for m in re.finditer(r"\b(\d{1,2}(?:\.\d{1,2})?)\b", ln)]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        # Reject lines that include weekly totals like 40/48 as part of row.
        if any(v > 24 for v in vals):
            continue
        if len(vals) not in expected_lengths:
            continue
        # Keep realistic per-day values.
        if any(v < 0 or v > 16 for v in vals):
            continue
        candidates.append([float(v) for v in vals])

    if not candidates:
        # Second-pass fallback: OCR/PDF text can split row values into separate lines.
        # Scan rolling windows of decimal hour tokens and pick the most plausible day-total series.
        filtered_text = "\n".join(ln for ln in lines if not is_non_work_line(ln))
        token_vals = [safe_float(m.group(1), None) for m in re.finditer(r"\b(\d{1,2}\.\d{2})\b", filtered_text)]
        token_vals = [float(v) for v in token_vals if v is not None and 0 <= v <= 16]
        if token_vals:
            # Typical workdays in period (Mon-Fri); fallback to 5..7.
            workday_count = sum(1 for d in period_dates if datetime.strptime(d, "%Y-%m-%d").weekday() < 5)
            lengths = [workday_count] if 5 <= workday_count <= 7 else [5, 6, 7]
            for ln in lengths:
                if ln <= 0 or len(token_vals) < ln:
                    continue
                for i in range(0, len(token_vals) - ln + 1):
                    row = token_vals[i : i + ln]
                    candidates.append(row)
        if not candidates:
            return []

    # Score candidates:
    # - prefer higher length (7 > 6 > 5),
    # - prefer rows close to daily standard (typically around 8),
    # - prefer low spread (stable day totals).
    def score(row: List[float]) -> float:
        n = len(row)
        mean = sum(row) / max(n, 1)
        spread = max(row) - min(row) if row else 0.0
        target_sum = 8.0 * n
        sum_penalty = -abs(sum(row) - target_sum) / 8.0
        around_eight = -abs(mean - 8.0)
        length_bonus = n * 0.3
        spread_penalty = -spread * 0.2
        return length_bonus + around_eight + spread_penalty + sum_penalty

    best = sorted(candidates, key=score, reverse=True)[0]
    return best


def _apply_project_grid_day_totals(normalized: Dict[str, Any], ocr_text: str) -> None:
    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or ps > pe:
        return
    # Keep this heuristic scoped to weekly-like grids; longer periods (e.g. semi-monthly)
    # are handled by other logic and can be distorted by short rolling token windows.
    if (pe - ps).days + 1 > 8:
        return
    row = _extract_project_grid_daily_totals_from_text(normalized, ocr_text)
    if not row:
        return

    period_dates = date_range(ps, pe)
    # Map row values to the first N dates in the period (usual left-to-right table order).
    out: List[Dict[str, Any]] = []
    for i, dt in enumerate(period_dates):
        hrs = row[i] if i < len(row) else 0.0
        out.append({"date": dt, "hours": round(float(hrs), 2)})
    normalized["day_hours"] = out
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def _apply_semi_monthly_weekend_zero_fix(normalized: Dict[str, Any], ocr_text: str) -> None:
    """
    Fix shifted semi-monthly patterns where weekends are incorrectly non-zero and many weekdays are zero.
    Scoped to OCR containing 'semi-monthly' and 15-day-like windows.
    """
    lowered = (ocr_text or "").lower()
    if "semi-monthly" not in lowered and "semi monthly" not in lowered:
        return
    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or ps > pe:
        return
    span = (pe - ps).days + 1
    if span < 14 or span > 16:
        return
    rows = normalized.get("day_hours", [])
    if not isinstance(rows, list) or not rows:
        return

    by_date: Dict[str, float] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        dt = str(r.get("date", ""))
        d = parse_iso_date_optional(dt)
        if not d:
            continue
        by_date[d.isoformat()] = safe_float(r.get("hours", 0.0), 0.0)
    if not by_date:
        return

    period_dates = date_range(ps, pe)
    weekend_non_zero = 0
    weekday_zero = 0
    weekday_non_zero_vals: List[float] = []
    for iso_dt in period_dates:
        d = datetime.strptime(iso_dt, "%Y-%m-%d").date()
        h = by_date.get(iso_dt, 0.0)
        if d.weekday() >= 5:
            if h > 0:
                weekend_non_zero += 1
        else:
            if h <= 0:
                weekday_zero += 1
            else:
                weekday_non_zero_vals.append(h)

    # Trigger only for clearly shifted/implausible patterns.
    if weekend_non_zero < 2 or weekday_zero < 4 or not weekday_non_zero_vals:
        return

    # Use dominant weekday non-zero value (usually 8.0).
    rounded_vals = [round(v, 2) for v in weekday_non_zero_vals]
    dominant = Counter(rounded_vals).most_common(1)[0][0]
    out: List[Dict[str, Any]] = []
    for iso_dt in period_dates:
        d = datetime.strptime(iso_dt, "%Y-%m-%d").date()
        if d.weekday() >= 5:
            hrs = 0.0
        else:
            hrs = float(dominant)
        out.append({"date": iso_dt, "hours": hrs})
    normalized["day_hours"] = out
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def _apply_weekly_mon_fri_alignment_fix(normalized: Dict[str, Any], ocr_text: str = "") -> None:
    """
    Fix weekly Sat..Fri templates where OCR/LLM can shift Monday-Friday hours into weekend columns.
    Trigger only for clear 7-day weekly windows with a Sat/Sun/Mon... header signature.
    """
    headers = [normalize_text(str(x)) for x in (normalized.get("headers") or []) if x is not None]
    table_cols = [str(x) for x in (normalized.get("table_columns") or []) if x is not None]
    if not headers or "total" not in " ".join(headers):
        return
    lowered_ocr = normalize_text(ocr_text or "")
    # If explicit non-working rows are present (e.g., Unpaid Time Off), this
    # normalization can incorrectly turn blank weekdays into 8. Skip in that case.
    if any(tok in lowered_ocr for tok in ["unpaid", "time off", "pto", "leave", "holiday"]):
        return
    # Signature like: Sat Sun Mon Tue Wed Thu Fri TOTAL
    signature = ["sat", "sun", "mon", "tue", "wed", "thu", "fri"]
    joined = " ".join(headers)
    if not all(day in joined for day in signature):
        return

    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or (pe - ps).days != 6:
        return

    rows = normalized.get("day_hours", []) if isinstance(normalized.get("day_hours"), list) else []
    if len(rows) < 7:
        return
    by_date: Dict[str, float] = {}
    for r in rows:
        if isinstance(r, dict):
            d = parse_iso_date_optional(str(r.get("date", "")))
            if d:
                by_date[d.isoformat()] = safe_float(r.get("hours", 0.0), 0.0)
    week_dates = date_range(ps, pe)
    week_vals = [by_date.get(d, 0.0) for d in week_dates]
    non_zero_vals = [round(v, 2) for v in week_vals if v > 0]
    if len(non_zero_vals) != 5:
        return
    dominant = Counter(non_zero_vals).most_common(1)[0][0] if non_zero_vals else 0.0
    if dominant <= 0:
        return
    # Only normalize when total clearly implies Mon-Fri pattern.
    total = safe_float(normalized.get("total_hours", None), None)
    if total is None:
        total = round(sum(week_vals), 2)
    if abs(total - (5.0 * dominant)) > APP_CONFIG["hour_tolerance"]:
        return

    out: List[Dict[str, Any]] = []
    for iso_dt in week_dates:
        d = datetime.strptime(iso_dt, "%Y-%m-%d").date()
        hrs = float(dominant) if d.weekday() < 5 else 0.0
        out.append({"date": iso_dt, "hours": hrs})
    normalized["day_hours"] = out
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def _apply_beeline_total_hours_row_fix(normalized: Dict[str, Any], ocr_text: str) -> None:
    """
    Parse Beeline-style weekly grids where OCR includes:
    - date headers: Sat 12/13 ... Fri 12/19
    - totals row: TOTAL HOURS 0 0 8 8 8 8 8 40
    Use date-cell aligned values (first 7 numbers) as day_hours.
    """
    raw = (ocr_text or "")
    lowered = normalize_text(raw)
    if "total hours" not in lowered or "pay code(required)" not in lowered:
        return

    # Capture Sat..Fri date headers.
    m_dates = re.search(
        r"Sat\s+(\d{1,2}/\d{1,2})\s+Sun\s+(\d{1,2}/\d{1,2})\s+Mon\s+(\d{1,2}/\d{1,2})\s+Tue\s+(\d{1,2}/\d{1,2})\s+Wed\s+(\d{1,2}/\d{1,2})\s+Thu\s+(\d{1,2}/\d{1,2})\s+Fri\s+(\d{1,2}/\d{1,2})",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_dates:
        return
    mmdd_list = [m_dates.group(i) for i in range(1, 8)]

    # Determine year from "Dec 13 - Dec 19, 2025" style token or existing period.
    inferred_year = None
    m_year = re.search(r"[A-Za-z]{3,9}\s+\d{1,2}\s*-\s*[A-Za-z]{3,9}\s+\d{1,2},\s*(\d{4})", raw)
    if m_year:
        inferred_year = int(m_year.group(1))
    if not inferred_year:
        ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
        pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
        inferred_year = ps.year if ps else (pe.year if pe else None)
    if not inferred_year:
        return

    # Parse TOTAL HOURS row: first 7 are per-day values, last one is weekly total.
    m_vals = re.search(
        r"TOTAL\s+HOURS\s+((?:\d+(?:\.\d+)?\s+){7})(\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_vals:
        return
    day_tokens = re.findall(r"\d+(?:\.\d+)?", m_vals.group(1))
    if len(day_tokens) != 7:
        return
    day_values = [safe_float(x, 0.0) for x in day_tokens]

    out: List[Dict[str, Any]] = []
    parsed_dates: List[date] = []
    for mmdd, hrs in zip(mmdd_list, day_values):
        dt = _parse_date_any(f"{mmdd}/{inferred_year}", inferred_year)
        if not dt:
            return
        parsed_dates.append(dt)
        out.append({"date": dt.isoformat(), "hours": round(float(hrs), 2)})

    if not out:
        return
    out.sort(key=lambda x: x["date"])
    normalized["day_hours"] = out
    normalized["period_start"] = min(parsed_dates).isoformat()
    normalized["period_end"] = max(parsed_dates).isoformat()
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def _apply_capitalone_total_hours_row_fix(normalized: Dict[str, Any], ocr_text: str) -> None:
    """
    Parse CapitalOne-style weekly grids where OCR includes:
    - weekday header row: Sat Sun Mon Tue Wed Thu Fri TOTAL
    - date row: 12/27 12/28 12/29 12/30 12/31 01/01 01/02
    - totals row: TOTAL HOURS 0 0 8 8 8 0 8 32
    """
    raw = (ocr_text or "")
    lowered = normalize_text(raw)
    if "total hours" not in lowered or "pay code" not in lowered:
        return
    if "capitalone" not in lowered and "capital one" not in lowered:
        return

    # Date row usually appears after Pay Code section with 7 mm/dd tokens.
    m_date_row = re.search(
        r"Pay\s*Code\s*\(required\)\s+((?:\d{1,2}/\d{1,2}\s+){6}\d{1,2}/\d{1,2})",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_date_row:
        return
    mmdd_list = re.findall(r"\d{1,2}/\d{1,2}", m_date_row.group(1))
    if len(mmdd_list) != 7:
        return

    # Parse TOTAL HOURS row: first 7 are per-day values, last one is weekly total.
    m_vals = re.search(
        r"TOTAL\s+HOURS\s+((?:\d+(?:\.\d+)?\s+){7})(\d+(?:\.\d+)?)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m_vals:
        return
    day_tokens = re.findall(r"\d+(?:\.\d+)?", m_vals.group(1))
    if len(day_tokens) != 7:
        return
    day_values = [safe_float(x, 0.0) for x in day_tokens]

    # Infer year from date-range banner like "Dec 27 Jan 02, 2026".
    inferred_year = None
    m_year = re.search(r"[A-Za-z]{3,9}\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{1,2},\s*(\d{4})", raw)
    if m_year:
        inferred_year = int(m_year.group(1))
    if not inferred_year:
        ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
        pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
        inferred_year = ps.year if ps else (pe.year if pe else None)
    if not inferred_year:
        return

    out: List[Dict[str, Any]] = []
    parsed_dates: List[date] = []
    first_month = int(mmdd_list[0].split("/", 1)[0]) if mmdd_list else None
    last_month = int(mmdd_list[-1].split("/", 1)[0]) if mmdd_list else None
    # Banner year typically belongs to end month in "Dec 27 Jan 02, 2026".
    # If month list wraps Dec->Jan, start year is previous year.
    current_year = inferred_year - 1 if (first_month and last_month and first_month > last_month) else inferred_year
    prev_month = None
    for mmdd, hrs in zip(mmdd_list, day_values):
        mo, dd = [int(x) for x in mmdd.split("/", 1)]
        if prev_month is not None and mo < prev_month:
            current_year += 1
        prev_month = mo
        try:
            dt = date(current_year, mo, dd)
        except ValueError:
            return
        parsed_dates.append(dt)
        out.append({"date": dt.isoformat(), "hours": round(float(hrs), 2)})

    if not out:
        return
    out.sort(key=lambda x: x["date"])
    normalized["day_hours"] = out
    normalized["period_start"] = min(parsed_dates).isoformat()
    normalized["period_end"] = max(parsed_dates).isoformat()
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def _apply_sentara_weekly_shift_fix(normalized: Dict[str, Any]) -> None:
    """
    Fix a specific weekly shift pattern seen in Sentara-style screenshots:
    Sun is incorrectly populated with 8 and Fri is 0, while total remains 40.
    In that pattern, move Sun hours to Fri and set Sun to 0.
    """
    headers = [normalize_text(str(x)) for x in (normalized.get("headers") or []) if x is not None]
    hdr_text = " ".join(headers)
    # Narrow signature: day headers appear as Sun..Sat labels.
    if not all(k in hdr_text for k in ["sun", "mon", "tue", "wed", "thu", "fri", "sat"]):
        return
    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or (pe - ps).days != 6:
        return
    rows = normalized.get("day_hours", []) if isinstance(normalized.get("day_hours"), list) else []
    if len(rows) < 7:
        return
    by_date: Dict[str, float] = {}
    for r in rows:
        if isinstance(r, dict):
            d = parse_iso_date_optional(str(r.get("date", "")))
            if d:
                by_date[d.isoformat()] = safe_float(r.get("hours", 0.0), 0.0)
    week_dates = date_range(ps, pe)
    if len(week_dates) != 7:
        return
    vals = {datetime.strptime(d, "%Y-%m-%d").weekday(): by_date.get(d, 0.0) for d in week_dates}  # Mon=0..Sun=6
    sun = vals.get(6, 0.0)
    mon = vals.get(0, 0.0)
    tue = vals.get(1, 0.0)
    wed = vals.get(2, 0.0)
    thu = vals.get(3, 0.0)
    fri = vals.get(4, 0.0)
    sat = vals.get(5, 0.0)
    total = safe_float(normalized.get("total_hours", None), None)
    if total is None:
        total = sum(vals.values())
    # Very specific trigger to avoid impacting other templates.
    if not (
        abs(total - 40.0) <= APP_CONFIG["hour_tolerance"]
        and sun > 0
        and fri <= APP_CONFIG["hour_tolerance"]
        and sat <= APP_CONFIG["hour_tolerance"]
        and mon > 0
        and tue > 0
        and wed > 0
        and thu > 0
    ):
        return
    out: List[Dict[str, Any]] = []
    for iso_dt in week_dates:
        d = datetime.strptime(iso_dt, "%Y-%m-%d").date()
        hrs = by_date.get(iso_dt, 0.0)
        if d.weekday() == 6:  # Sunday
            hrs = 0.0
        elif d.weekday() == 4:  # Friday
            hrs = float(sun)
        out.append({"date": iso_dt, "hours": round(float(hrs), 2)})
    normalized["day_hours"] = out
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def _apply_itineris_project_week_fix(normalized: Dict[str, Any], ocr_text: str) -> None:
    """
    Scoped correction for Itineris Timesheet Report layouts where OCR token windows can produce
    skewed day totals (e.g., 4.5 and 11 instead of 8 and 8) for a Mon-Sun period.
    """
    lowered = normalize_text(ocr_text or "")
    if "timesheet report" not in lowered or "unique timesheet number" not in lowered:
        return
    headers_text = " ".join(str(x) for x in (normalized.get("headers") or []))
    scope_text = normalize_text(headers_text)
    if not ("project" in scope_text and "task" in scope_text and "activity" in scope_text):
        return

    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or (pe - ps).days != 6:
        return
    period_dates = date_range(ps, pe)
    rows = normalized.get("day_hours", []) if isinstance(normalized.get("day_hours"), list) else []
    if len(rows) < 7:
        return
    by_date: Dict[str, float] = {}
    for r in rows:
        if isinstance(r, dict):
            d = parse_iso_date_optional(str(r.get("date", "")))
            if d:
                by_date[d.isoformat()] = safe_float(r.get("hours", 0.0), 0.0)
    vals = []
    for iso_dt in period_dates:
        d = datetime.strptime(iso_dt, "%Y-%m-%d").date()
        vals.append((d.weekday(), by_date.get(iso_dt, 0.0), iso_dt))
    weekday_vals = [v for wd, v, _ in vals if wd < 5]
    weekend_vals = [v for wd, v, _ in vals if wd >= 5]
    if len(weekday_vals) != 5:
        return
    total = safe_float(normalized.get("total_hours", None), None)
    if total is None:
        total = sum(v for _, v, _ in vals)
    # Trigger only for the known skew profile around a 40-hour week.
    if not (
        39.0 <= total <= 41.0
        and max(weekday_vals) >= 10.0
        and min(weekday_vals) <= 5.0
        and all(v <= APP_CONFIG["hour_tolerance"] for v in weekend_vals)
    ):
        return

    # Normalize to standard 8-hour weekdays for this report style.
    target_day = 8.0
    out: List[Dict[str, Any]] = []
    for wd, _, iso_dt in vals:
        hrs = target_day if wd < 5 else 0.0
        out.append({"date": iso_dt, "hours": hrs})
    normalized["day_hours"] = out
    normalized["total_hours"] = 40.0


def _apply_entry_day_totals_fix(normalized: Dict[str, Any], ocr_text: str) -> None:
    """
    Oracle-style Time Card PDFs can include a clear 'Entry and Earned Day Totals' section
    listing only the dates with logged time. Use those explicit date totals and set
    unspecified period dates to 0.
    """
    raw = (ocr_text or "")
    lowered = normalize_text(raw)
    if "entry and earned day totals" not in lowered or "time card" not in lowered:
        return

    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if not ps or not pe or ps > pe:
        return

    # Parse rows like: 12/29/2025 8.00 8.00
    date_totals: Dict[str, float] = {}
    for m in re.finditer(r"\b(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}(?:\.\d{1,2})?)\s+(\d{1,2}(?:\.\d{1,2})?)\b", raw):
        d = _parse_date_any(m.group(1), None)
        if not d:
            continue
        reported = safe_float(m.group(2), None)
        if reported is None:
            continue
        date_totals[d.isoformat()] = float(reported)

    if not date_totals:
        return

    out: List[Dict[str, Any]] = []
    for iso_dt in date_range(ps, pe):
        out.append({"date": iso_dt, "hours": round(float(date_totals.get(iso_dt, 0.0)), 2)})
    normalized["day_hours"] = out
    normalized["total_hours"] = round(sum(x["hours"] for x in out), 2)


def postprocess_normalized(normalized: Dict[str, Any], ocr_text: str) -> Dict[str, Any]:
    # Normalize day_hours date formats first (to stabilize comparisons).
    default_year = None
    existing_start = parse_iso_date_optional(str(normalized.get("period_start", "")))
    existing_end = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if existing_start:
        default_year = existing_start.year
    elif existing_end:
        default_year = existing_end.year
    _normalize_day_hours_dates_inplace(normalized, default_year)

    # Always infer a candidate period; only write it if missing/invalid OR if extracted years look implausible.
    inferred_start, inferred_end = _infer_period_from_extraction(normalized, ocr_text)
    existing_start = parse_iso_date_optional(str(normalized.get("period_start", "")))
    existing_end = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if (not existing_start or not existing_end) and inferred_start and inferred_end:
        normalized["period_start"] = inferred_start
        normalized["period_end"] = inferred_end
    else:
        # If the existing period doesn't cover extracted day_hours at all, replace with inferred (safer for UI-only week displays).
        if inferred_start and inferred_end:
            inf_s = parse_iso_date_optional(inferred_start)
            inf_e = parse_iso_date_optional(inferred_end)
            if inf_s and inf_e:
                dts = [parse_iso_date_optional(x.get("date", "")) for x in (normalized.get("day_hours", []) or [])]
                dts = [d for d in dts if d]
                if dts and existing_start and existing_end:
                    if not (existing_start <= min(dts) and existing_end >= max(dts)):
                        normalized["period_start"] = inferred_start
                        normalized["period_end"] = inferred_end
        # If OCR-derived period is clearly far from extracted period (typically wrong LLM year),
        # trust OCR period and let downstream alignment fix day_hours years.
        if inferred_start and inferred_end and existing_start and existing_end:
            inf_s = parse_iso_date_optional(inferred_start)
            inf_e = parse_iso_date_optional(inferred_end)
            if inf_s and inf_e:
                far_from_ocr = abs((existing_start - inf_s).days) > 300 or abs((existing_end - inf_e).days) > 300
                if far_from_ocr:
                    normalized["period_start"] = inferred_start
                    normalized["period_end"] = inferred_end

    # Now that period may be set, align any wrong-year day_hours entries to the period window.
    _align_day_hours_years_to_period(normalized)

    # Final guardrail: if period is unrealistically broad compared to extracted day_hours,
    # prefer the concrete day_hours date window (helps table/header disambiguation cases).
    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    rows = normalized.get("day_hours", []) if isinstance(normalized.get("day_hours"), list) else []
    parsed_days: List[date] = []
    for row in rows:
        if isinstance(row, dict):
            d = parse_iso_date_optional(str(row.get("date", "")))
            if d:
                parsed_days.append(d)
    if ps and pe and parsed_days and ps <= pe:
        dh_min = min(parsed_days)
        dh_max = max(parsed_days)
        period_span = (pe - ps).days + 1
        day_span = (dh_max - dh_min).days + 1
        unique_day_count = len(set(parsed_days))
        # Trigger only for clear outliers so existing good files aren't affected.
        too_broad_for_table = period_span > max(unique_day_count * 3, 31)
        day_window_reasonable = 1 <= day_span <= 31 and unique_day_count >= min(day_span, 4)
        if too_broad_for_table and day_window_reasonable:
            normalized["period_start"] = dh_min.isoformat()
            normalized["period_end"] = dh_max.isoformat()

    # Strong anchor: explicit Ending Date + weekday hour grid (fixes wrong-year LLM output on some captures).
    try:
        _apply_ending_date_week_anchor(normalized, ocr_text)
    except Exception:
        pass

    # Project-grid reports often include multiple project rows; prefer consolidated day totals.
    try:
        _apply_project_grid_day_totals(normalized, ocr_text)
    except Exception:
        pass

    # Semi-monthly screen captures can produce shifted weekend values; normalize only when clearly inconsistent.
    try:
        _apply_semi_monthly_weekend_zero_fix(normalized, ocr_text)
    except Exception:
        pass
    try:
        _apply_weekly_mon_fri_alignment_fix(normalized, ocr_text)
    except Exception:
        pass
    try:
        _apply_beeline_total_hours_row_fix(normalized, ocr_text)
    except Exception:
        pass
    try:
        _apply_capitalone_total_hours_row_fix(normalized, ocr_text)
    except Exception:
        pass
    try:
        _apply_sentara_weekly_shift_fix(normalized)
    except Exception:
        pass
    try:
        _apply_itineris_project_week_fix(normalized, ocr_text)
    except Exception:
        pass
    try:
        _apply_entry_day_totals_fix(normalized, ocr_text)
    except Exception:
        pass
    # Ensure total_hours is aligned with day_hours for daily templates.
    # (Keep summary-style sheets untouched, since their rows can be weekly totals.)
    if isinstance(normalized.get("day_hours"), list) and normalized["day_hours"] and not _is_summary_like_day_hours(normalized):
        day_total = round(sum(safe_float(x.get("hours", 0.0), 0.0) for x in normalized["day_hours"]), 2)
        existing_total = safe_float(normalized.get("total_hours", None), None)
        if existing_total is None or abs(existing_total - day_total) > APP_CONFIG["hour_tolerance"]:
            normalized["total_hours"] = day_total

    return normalized


def extract_document_text_only(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    OCR / local text extraction only (Textract or fallbacks). No Bedrock.

    Returns on success:
      {"ok": True, "text": str, "raw": Any, "textract_used": bool,
       "prevalidation": dict, "quality_validation": dict}
    Returns on failure (same semantics as early return from extract_and_normalize):
      {"ok": False, "normalized": dict, "raw": dict, "prevalidation": dict, "quality_validation": dict}
    """
    precheck = prevalidate_file(file_bytes, filename)
    quality = validate_image_quality(file_bytes, filename)
    ext = os.path.splitext((filename or "").lower())[1]
    if precheck.get("failed"):
        return {
            "ok": False,
            "normalized": default_normalized("Pre-validation failed"),
            "raw": {"error": "; ".join(precheck.get("issues", []))},
            "prevalidation": precheck,
            "quality_validation": quality,
        }
    try:
        if ext in {".pdf", ".png", ".jpg", ".jpeg"}:
            textract_out = textract_extract(file_bytes, filename)
            textract_used = True
        elif ext == ".txt":
            textract_out = {
                "raw": {"fallback": "txt_text_extract"},
                "text": extract_text_from_txt_local(file_bytes),
                "filename": filename,
            }
            textract_used = False
        elif ext == ".docx":
            textract_out = {
                "raw": {"fallback": "docx_text_extract"},
                "text": extract_text_from_docx_local(file_bytes),
                "filename": filename,
            }
            textract_used = False
        elif ext == ".doc":
            textract_out = {
                "raw": {"fallback": "doc_text_extract"},
                "text": extract_text_from_doc_local(file_bytes),
                "filename": filename,
            }
            textract_used = False
        else:
            return {
                "ok": False,
                "normalized": default_normalized(f"unsupported_extension: {ext}"),
                "raw": {"error": f"Unsupported extension: {ext}"},
                "prevalidation": precheck,
                "quality_validation": quality,
            }
    except Exception as exc:
        err = str(exc)
        if ext == ".pdf" and ("UnsupportedDocumentException" in err or "InvalidParameterException" in err):
            try:
                fallback_text = extract_text_from_pdf_local(file_bytes)
                if fallback_text.strip():
                    textract_out = {"raw": {"fallback": "pypdf_text_extract"}, "text": fallback_text, "filename": filename}
                    textract_used = False
                else:
                    return {
                        "ok": False,
                        "normalized": default_normalized(f"textract_failed: {exc}"),
                        "raw": {"error": str(exc)},
                        "prevalidation": precheck,
                        "quality_validation": quality,
                    }
            except Exception as pdf_exc:
                return {
                    "ok": False,
                    "normalized": default_normalized(f"textract_failed: {exc}; pdf_fallback_failed: {pdf_exc}"),
                    "raw": {"error": str(exc)},
                    "prevalidation": precheck,
                    "quality_validation": quality,
                }
        return {
            "ok": False,
            "normalized": default_normalized(f"textract_failed: {exc}"),
            "raw": {"error": str(exc)},
            "prevalidation": precheck,
            "quality_validation": quality,
        }

    text = (textract_out.get("text", "") or "").strip()
    if not text:
        return {
            "ok": False,
            "normalized": default_normalized("No readable text found in document"),
            "raw": {"error": "No readable text found in document"},
            "prevalidation": precheck,
            "quality_validation": quality,
        }
    return {
        "ok": True,
        "text": text,
        "raw": textract_out.get("raw"),
        "textract_used": textract_used,
        "prevalidation": precheck,
        "quality_validation": quality,
    }


def llm_cost_preview_from_extracted_text(text: str, hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Token + USD estimate for the Bedrock normalization prompt (no API calls)."""
    cfg = get_runtime_aws_config()
    model_id = (cfg.get("bedrock_model_id") or "").strip()
    prompt = build_bedrock_normalization_prompt(text, hints)
    return estimate_bedrock_llm_cost(model_id, prompt)


def ellipsize_middle(text: str, max_len: int = 34) -> str:
    """Shorten long strings (e.g. filenames) for compact tables."""
    s = (text or "").strip() or "—"
    if len(s) <= max_len:
        return s
    left = max(8, max_len // 2 - 1)
    right = max_len - left - 1
    return s[:left] + "…" + s[-right:] if right > 0 else s[: max_len - 1] + "…"


def short_bedrock_model_label(model_id: str) -> str:
    """Friendly label for Bedrock model IDs (demo-friendly)."""
    mid = (model_id or "").lower()
    ordered = [
        ("claude-3-opus", "Claude 3 Opus"),
        ("claude-3-5-sonnet", "Claude 3.5 Sonnet"),
        ("claude-3-sonnet", "Claude 3 Sonnet"),
        ("claude-3-haiku", "Claude 3 Haiku"),
        ("claude-sonnet-4", "Claude Sonnet 4"),
        ("claude-3-7-sonnet", "Claude 3.7 Sonnet"),
    ]
    for needle, label in ordered:
        if needle in mid:
            return label
    raw = (model_id or "").strip()
    if not raw:
        return "—"
    tail = raw.split("/")[-1]
    if len(tail) > 26:
        return ellipsize_middle(tail, 26)
    return tail


def _llm_preview_pricing_fingerprint() -> str:
    cfg = get_runtime_aws_config()
    parts = [
        cfg.get("bedrock_model_id", "") or "",
        get_setting_text("llm_pricing_mode", "auto"),
        str(get_setting_float("llm_custom_input_usd_per_mtok", 0.0)),
        str(get_setting_float("llm_custom_output_usd_per_mtok", 0.0)),
        str(get_setting_float("llm_custom_input_credits_per_mtok", 0.0)),
        str(get_setting_float("llm_custom_output_credits_per_mtok", 0.0)),
        str(APP_CONFIG.get("llm_assumed_output_tokens", 1200)),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_compact_llm_preview_rows(
    preview_targets: List[Dict[str, Any]],
    hints_preview: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """One compact row per file for Step 2 auto-preview (Textract/OCR only); optional total row for multi-file."""
    c_in = LLM_PREVIEW_TABLE_LABELS["col_in"]
    c_out = LLM_PREVIEW_TABLE_LABELS["col_out"]
    c_rate = LLM_PREVIEW_TABLE_LABELS["col_rate"]
    c_tot = LLM_PREVIEW_TABLE_LABELS["col_total"]
    rows: List[Dict[str, Any]] = []
    sum_in = 0
    sum_out = 0
    sum_charge = 0.0
    unit_seen: str | None = None
    n_ok = 0

    for idx, fo in enumerate(preview_targets):
        fn = str(fo.get("name") or f"file_{idx + 1}").strip() or f"file_{idx + 1}"
        tex = extract_document_text_only(fo["bytes"], fn)
        if not tex.get("ok"):
            err = (tex.get("raw") or {}).get("error") or str((tex.get("normalized") or {}).get("error", "Failed"))
            rows.append(
                {
                    "File": ellipsize_middle(fn, 32),
                    "Model": "—",
                    c_in: None,
                    c_out: None,
                    c_rate: "—",
                    c_tot: ellipsize_middle(f"Issue: {err}", 44),
                }
            )
            continue
        prev = llm_cost_preview_from_extracted_text(tex["text"], hints=hints_preview)
        u = str(prev.get("charge_unit", "USD") or "USD")
        unit_seen = u
        rin = prev.get("input_rate_per_mtok")
        rout = prev.get("output_rate_per_mtok")
        try:
            rate_s = f"{float(rin):g}/{float(rout):g}" if rin is not None and rout is not None else "—"
        except (TypeError, ValueError):
            rate_s = f"{rin}/{rout}"
        total = prev.get("total_charge_est")
        try:
            tot_s = f"{float(total):.4f} {u}" if total is not None else "—"
            sum_charge += float(total) if total is not None else 0.0
        except (TypeError, ValueError):
            tot_s = str(total)
        it = int(prev.get("input_tokens_est") or 0)
        ot = int(prev.get("output_tokens_assumed") or 0)
        sum_in += it
        sum_out += ot
        n_ok += 1
        rows.append(
            {
                "File": ellipsize_middle(fn, 32),
                "Model": short_bedrock_model_label(str(prev.get("model_id", "") or "")),
                c_in: it,
                c_out: ot,
                c_rate: rate_s,
                c_tot: tot_s,
            }
        )

    if len(preview_targets) > 1 and n_ok > 0:
        u = unit_seen or "USD"
        rows.append(
            {
                "File": ellipsize_middle(f"Total ({len(preview_targets)} files, {n_ok} ok)", 36),
                "Model": "—",
                c_in: sum_in,
                c_out: sum_out,
                c_rate: "—",
                c_tot: f"{sum_charge:.4f} {u}",
            }
        )
    return rows


def extract_and_normalize(file_bytes: bytes, filename: str, hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    runtime_cfg = get_runtime_aws_config()
    meta = {
        "aws_region": runtime_cfg["aws_region"],
        "model_id": runtime_cfg["bedrock_model_id"],
        "textract_used": False,
        "llm_used": False,
        "prevalidation": {},
        "quality_validation": {},
    }
    text_only = extract_document_text_only(file_bytes, filename)
    meta["prevalidation"] = text_only.get("prevalidation", {})
    meta["quality_validation"] = text_only.get("quality_validation", {})
    if not text_only.get("ok"):
        return {
            "normalized": text_only["normalized"],
            "raw": text_only["raw"],
            "meta": meta,
        }
    meta["textract_used"] = bool(text_only.get("textract_used"))
    textract_out = {
        "text": text_only["text"],
        "raw": text_only.get("raw"),
        "filename": filename,
    }

    bedrock_failed = False
    llm_cost_preview: Dict[str, Any] = {}
    try:
        normalized = normalize_with_bedrock(
            textract_out["text"],
            hints=hints,
            source_label=filename or "",
            cost_preview_holder=llm_cost_preview,
        )
        meta["llm_used"] = True
    except Exception as exc:
        bedrock_failed = True
        normalized = default_normalized(f"bedrock_normalization_failed: {exc}")
    if llm_cost_preview:
        meta["llm_cost_preview"] = llm_cost_preview

    # Fallback: if Bedrock failed, we still try to get day-wise hours from Textract tables.
    if bedrock_failed and not normalized.get("day_hours"):
        try:
            normalized["day_hours"] = _fallback_extract_day_hours_from_textract(textract_out.get("raw", {}), textract_out.get("text", ""))
            meta["llm_used"] = False
        except Exception:
            # Must never break the pipeline.
            pass

    try:
        normalized = postprocess_normalized(normalized, textract_out.get("text", ""))
    except Exception:
        # Post-processing must never break the pipeline.
        pass

    confidence_breakdown = build_confidence_breakdown(textract_out["raw"], normalized)
    normalized["confidence_breakdown"] = confidence_breakdown
    # Fallback approval detection from OCR text (e.g., "Status: Approved").
    inferred_approved = infer_approved_from_text(textract_out.get("text", ""))
    if not normalize_text(str(normalized.get("approved", ""))):
        if inferred_approved:
            normalized["approved"] = inferred_approved
    else:
        # Guardrail: if OCR has no approval signal, don't keep unrelated text in approved field.
        approved_raw = str(normalized.get("approved", ""))
        approved_norm = normalize_text(approved_raw)
        has_local_approval_keyword = bool(
            re.search(r"\b(approved|signature|signed|sign)\b", approved_norm, flags=re.IGNORECASE)
        )
        if not inferred_approved and not has_local_approval_keyword:
            normalized["approved"] = ""
    # Extract approver readable name when present.
    if not normalize_text(str(normalized.get("approver_name", ""))):
        inferred_approver_name = infer_approver_name_from_text(textract_out.get("text", ""))
        if inferred_approver_name:
            normalized["approver_name"] = inferred_approver_name
    elif not is_probable_person_name(str(normalized.get("approver_name", ""))):
        normalized["approver_name"] = ""
    # Keep approver_name only with explicit approval context, and avoid employee-name leakage.
    approver_raw = str(normalized.get("approver_name", "") or "").strip()
    if approver_raw:
        if not has_explicit_approver_context(textract_out.get("text", "")):
            normalized["approver_name"] = ""
        else:
            approver_canon = canonical_person_name(approver_raw)
            employee_canon = canonical_person_name(str(normalized.get("employee_name", "") or ""))
            if approver_canon and employee_canon and approver_canon == employee_canon:
                normalized["approver_name"] = ""
    # Fallback company inference for templates where LLM leaves company blank.
    if not normalize_text(str(normalized.get("company", ""))):
        inferred_company = infer_company_from_text(textract_out.get("text", ""))
        if inferred_company:
            normalized["company"] = inferred_company

    if not normalized.get("confidence"):
        normalized["confidence"] = round(confidence_breakdown.get("aggregate_confidence", 0.0) / 100.0, 3)

    if "confidence" not in normalized:
        normalized["confidence"] = 0.0
    # Return OCR text separately (not stored in history meta by default).
    return {"normalized": normalized, "raw": textract_out["raw"], "meta": meta, "ocr_text": textract_out.get("text", "")}


def compare_data(step1: Dict[str, Any], extracted: Dict[str, Any], tolerance: float = APP_CONFIG["hour_tolerance"]) -> Dict[str, Any]:
    result = {"matches": {}, "mismatches": {}, "critical_ok": True, "hour_mismatch_count": 0}
    # Employee name comparison should tolerate "Last, First" vs "First Last".
    emp_ok = canonical_person_name(step1.get("employee_name", "")) == canonical_person_name(extracted.get("employee_name", ""))
    result["matches"]["employee_name"] = emp_ok
    if not emp_ok:
        result["mismatches"]["employee_name"] = {
            "expected": step1.get("employee_name"),
            "actual": extracted.get("employee_name"),
        }
        result["critical_ok"] = False

    step1_vendor = normalize_text(step1.get("vendor", ""))
    step1_company = normalize_text(step1.get("company", ""))
    ext_vendor = normalize_text(extracted.get("vendor", ""))
    ext_company = normalize_text(extracted.get("company", ""))
    vendor_ok = step1_vendor == ext_vendor
    company_ok = step1_company == ext_company
    # Tolerate split vs combined forms, e.g. "American Express Corp. Saume"
    # compared against extracted vendor/company split fields.
    combined_ext = normalize_text(f"{extracted.get('vendor', '')} {extracted.get('company', '')}")
    combined_step1 = normalize_text(f"{step1.get('vendor', '')} {step1.get('company', '')}")
    if not vendor_ok and step1_vendor and combined_ext and step1_vendor == combined_ext:
        vendor_ok = True
        company_ok = True if not step1_company else company_ok
    if not vendor_ok and ext_vendor and combined_step1 and ext_vendor == combined_step1:
        vendor_ok = True
        company_ok = True if not ext_company else company_ok

    result["matches"]["vendor"] = vendor_ok
    if not vendor_ok:
        result["mismatches"]["vendor"] = {"expected": step1.get("vendor"), "actual": extracted.get("vendor")}
        result["critical_ok"] = False
    result["matches"]["company"] = company_ok
    if not company_ok:
        result["mismatches"]["company"] = {"expected": step1.get("company"), "actual": extracted.get("company")}
        result["critical_ok"] = False
    # Required fields must be present in extracted timesheet data.
    # Even if Step1 is blank too, missing vendor/company should require manual review.
    for required_field in ["vendor", "company"]:
        if not normalize_text(extracted.get(required_field, "")):
            result["matches"][required_field] = False
            result["mismatches"][required_field] = {
                "expected": "non-empty value",
                "actual": extracted.get(required_field, ""),
            }
            result["critical_ok"] = False
    step1_ps = parse_iso_date_optional(str(step1.get("period_start", "")))
    step1_pe = parse_iso_date_optional(str(step1.get("period_end", "")))
    for field in ["period_start", "period_end"]:
        ok = (step1.get(field) or "") == (extracted.get(field) or "")
        # Accept extracted period boundary dates that fall inside Step 1 selected range.
        # This avoids false mismatches on summary-style sheets that report an in-range anchor date.
        if not ok and step1_ps and step1_pe:
            extracted_dt = parse_iso_date_optional(str(extracted.get(field, "")))
            if extracted_dt and step1_ps <= extracted_dt <= step1_pe:
                ok = True
        result["matches"][field] = ok
        if not ok:
            result["mismatches"][field] = {"expected": step1.get(field), "actual": extracted.get(field)}
    approved_text = normalize_text(extracted.get("approved", ""))
    approver_name_text = str(extracted.get("approver_name", "") or "").strip()
    has_valid_approver_name = bool(approver_name_text) and is_probable_person_name(approver_name_text)
    approved_ok = (
        has_valid_approver_name
        or (bool(approved_text) and approved_text not in ["no", "none", "na", "n/a"])
    )
    result["matches"]["approved"] = approved_ok
    if not approved_ok:
        result["mismatches"]["approved"] = {"expected": "yes/signature", "actual": extracted.get("approved", "")}
    expected_map = {d["date"]: float(d["hours"]) for d in step1.get("day_hours", [])}
    extracted_rows = extracted.get("day_hours", []) if isinstance(extracted.get("day_hours"), list) else []
    actual_map = {d.get("date"): safe_float(d.get("hours", 0), 0.0) for d in extracted_rows if d.get("date")}
    observed_rows = (
        extracted.get("observed_day_hours_all_files", [])
        if isinstance(extracted.get("observed_day_hours_all_files"), list)
        else []
    )
    observed_map = {d.get("date"): safe_float(d.get("hours", 0), 0.0) for d in observed_rows if d.get("date")}
    excluded_rows = (
        extracted.get("excluded_day_hours", [])
        if isinstance(extracted.get("excluded_day_hours"), list)
        else []
    )
    excluded_map = {d.get("date"): d for d in excluded_rows if isinstance(d, dict) and d.get("date")}
    hdr_tokens: List[str] = []
    if isinstance(extracted.get("headers"), list):
        hdr_tokens.extend(str(x) for x in extracted.get("headers") if x is not None)
    if isinstance(extracted.get("table_columns"), list):
        hdr_tokens.extend(str(x) for x in extracted.get("table_columns") if x is not None)
    hdr_text = normalize_text(" ".join(hdr_tokens))
    summary_header = ("week end" in hdr_text) or ("week ending" in hdr_text) or ("week_end" in hdr_text)
    extracted_hours_vals = [
        safe_float(r.get("hours", None), None) for r in extracted_rows if isinstance(r, dict)
    ]
    extracted_hours_vals = [h for h in extracted_hours_vals if h is not None]
    summary_like_rows = (
        len(extracted_rows) <= 10
        and len(expected_map) >= 5
        and (
            summary_header
            or any(h > 16.0 for h in extracted_hours_vals)  # likely weekly totals, not single-day entries
        )
    )
    has_extracted_daywise = len(actual_map) > 0 and not summary_like_rows
    result["day_hours_missing_in_extracted"] = not has_extracted_daywise
    hour_diffs = []
    if has_extracted_daywise:
        for dt, expected in expected_map.items():
            actual = actual_map.get(dt)
            # Some templates omit explicit zero-hour dates in extracted rows.
            # If Step1 expects 0 and extracted date is missing, treat as 0 (match).
            if actual is None and abs(expected) <= tolerance:
                actual = 0.0
            excluded_note = ""
            if actual is None and dt in excluded_map:
                # Date exists in uploads, but was intentionally excluded from merged
                # totals due per-file Step1 mismatch checks.
                actual = 0.0
                excluded_note = str(excluded_map.get(dt, {}).get("reason", "Excluded from merged total"))
            elif actual is None and dt in observed_map:
                actual = observed_map.get(dt)
            if actual is None or abs(expected - actual) > tolerance:
                row = {"date": dt, "expected": expected, "actual": actual}
                if excluded_note:
                    row["note"] = excluded_note
                hour_diffs.append(row)
                result["hour_mismatch_count"] += 1
        result["matches"]["day_hours"] = len(hour_diffs) == 0
        if hour_diffs:
            result["mismatches"]["day_hours"] = hour_diffs
    if excluded_rows:
        result["excluded_day_hours"] = excluded_rows
    else:
        # Some summary templates have only week totals and no day-wise rows.
        # Do not force a day-wise mismatch in this case; compare via period + total hours.
        result["matches"]["day_hours"] = True

    # Fallback for summary-style timesheets (weekly rows with totals rather than full daily breakdown).
    # Criteria: Step1 period dates + total hours should match one extracted summary period.
    summary_match = False
    summary_row_total = None
    summary_period_start = None
    summary_period_end = None
    should_try_summary_match = expected_map and (
        summary_like_rows or (has_extracted_daywise and not result["matches"]["day_hours"])
    )
    if should_try_summary_match:
        sparse_summary_like = summary_like_rows or (len(extracted_rows) <= 10 and len(expected_map) >= 5)
        if sparse_summary_like:
            step1_ps = parse_iso_date_optional(str(step1.get("period_start", "")))
            step1_pe = parse_iso_date_optional(str(step1.get("period_end", "")))
            expected_total = sum(expected_map.values())
            prefers_week_end_date = ("week end" in hdr_text) or ("week ending" in hdr_text) or ("week_end" in hdr_text)
            step1_span_days = None
            if step1_ps and step1_pe and step1_pe >= step1_ps:
                step1_span_days = (step1_pe - step1_ps).days
            for row in extracted_rows:
                if not isinstance(row, dict):
                    continue
                row_start = parse_iso_date_optional(str(row.get("date", "")))
                row_total = safe_float(row.get("hours", None), None)
                if row_start is None or row_total is None:
                    continue
                # Some reports store week START date; some store week END date.
                # Try both period interpretations, with a header-aware priority.
                default_span = 6
                span = step1_span_days if step1_span_days is not None else default_span
                start_as_start = row_start
                end_as_start = row_start + timedelta(days=span)
                start_as_end = row_start - timedelta(days=span)
                end_as_end = row_start
                period_candidates = (
                    [(start_as_end, end_as_end), (start_as_start, end_as_start)]
                    if prefers_week_end_date
                    else [(start_as_start, end_as_start), (start_as_end, end_as_end)]
                )
                for cand_start, cand_end in period_candidates:
                    if step1_ps and step1_pe and cand_start == step1_ps and cand_end == step1_pe:
                        if abs(expected_total - row_total) <= tolerance:
                            summary_match = True
                            summary_row_total = row_total
                            summary_period_start = cand_start.isoformat()
                            summary_period_end = cand_end.isoformat()
                            break
                if summary_match:
                    break
            if summary_match:
                result["matches"]["day_hours"] = True
                result["mismatches"].pop("day_hours", None)
                result["matches"]["period_start"] = True
                result["matches"]["period_end"] = True
                result["mismatches"].pop("period_start", None)
                result["mismatches"].pop("period_end", None)
                result["derived_extracted_period"] = {"period_start": summary_period_start, "period_end": summary_period_end}
    expected_total = sum(expected_map.values())
    actual_total = extracted.get("total_hours")
    if summary_match and summary_row_total is not None:
        actual_total = summary_row_total
    elif has_extracted_daywise and expected_map:
        # Prefer total over the Step1-selected date window (not document grand total)
        # when extracted daily rows are available.
        window_total = 0.0
        for dt in expected_map.keys():
            val = actual_map.get(dt, None)
            if val is None:
                # Keep parity with day-wise behavior for omitted explicit zero rows.
                if abs(expected_map.get(dt, 0.0)) <= tolerance:
                    val = 0.0
                else:
                    val = 0.0
            window_total += safe_float(val, 0.0)
        actual_total = round(window_total, 2)
    elif actual_total is None and actual_map:
        actual_total = sum(actual_map.values())
    actual_total_num = safe_float(actual_total, None)
    result["actual_total_for_compare"] = actual_total_num
    ok = actual_total_num is not None and abs(expected_total - actual_total_num) <= tolerance
    result["matches"]["total_hours"] = ok
    if not ok:
        result["mismatches"]["total_hours"] = {"expected": expected_total, "actual": actual_total_num}

    # Coverage-friendly period rule:
    # If extracted dates are fully inside Step1 period and missing Step1 dates are zero-expected,
    # treat period_start/period_end as matched (avoids false manual review for leave/holiday gaps).
    actual_date_objs = [parse_iso_date_optional(str(dt)) for dt in actual_map.keys() if dt]
    actual_date_objs = [d for d in actual_date_objs if d is not None]
    if step1_ps and step1_pe and actual_date_objs:
        extracted_within_step1 = all(step1_ps <= d <= step1_pe for d in actual_date_objs)
        missing_non_zero_expected = any(
            (dt not in actual_map) and (abs(safe_float(expected, 0.0)) > tolerance)
            for dt, expected in expected_map.items()
        )
        if extracted_within_step1 and not missing_non_zero_expected and result["matches"].get("total_hours", False):
            result["matches"]["period_start"] = True
            result["matches"]["period_end"] = True
            result["mismatches"].pop("period_start", None)
            result["mismatches"].pop("period_end", None)
            result["period_coverage_note"] = "Partial extracted dates within selected Step 1 range"
    return result


def pattern_key(duration: str, day_hours: List[Dict[str, Any]], employee_name: str = "", vendor: str = "", company: str = "") -> str:
    pairs = []
    for item in sorted(day_hours, key=lambda x: x["date"]):
        d = datetime.strptime(item["date"], "%Y-%m-%d").strftime("%a")
        pairs.append(f"{d}:{float(item['hours']):.2f}")
    who = normalize_text(employee_name)
    ven = normalize_text(vendor)
    comp = normalize_text(company)
    return f"{duration}|emp:{who}|ven:{ven}|comp:{comp}|{'|'.join(pairs)}"


def template_hash(extracted: Dict[str, Any], p_key: str = "") -> str:
    # Use pattern_key as the primary bucketing key for trusted streak/learning memory.
    # This guarantees stable template hash for the same effective pattern.
    if p_key:
        return hashlib.sha256(f"pk::{p_key}".encode("utf-8")).hexdigest()
    # Fallback (defensive)
    raw = {
        "headers": normalize_text(" ".join(str(x) for x in (extracted.get("headers") or []))),
        "columns": normalize_text(" ".join(str(x) for x in (extracted.get("table_columns") or []))),
    }
    return hashlib.sha256(json.dumps(raw, sort_keys=True).encode("utf-8")).hexdigest()


def get_streak(employee: str, vendor: str, company: str, t_hash: str, p_key: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        select streak from trusted_streaks
        where employee_name=? and vendor=? and company=? and template_hash=? and pattern_key=?
        """,
        (employee, vendor, company, t_hash, p_key),
    ).fetchone()
    conn.close()
    return int(row[0]) if row else 0


def set_streak(employee: str, vendor: str, company: str, t_hash: str, p_key: str, streak: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        insert into trusted_streaks(employee_name,vendor,company,template_hash,pattern_key,streak)
        values(?,?,?,?,?,?)
        on conflict(employee_name,vendor,company,template_hash,pattern_key)
        do update set streak=excluded.streak
        """,
        (employee, vendor, company, t_hash, p_key, streak),
    )
    conn.commit()
    conn.close()


def clear_all_history() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("delete from validation_history")
    conn.execute("delete from trusted_streaks")
    conn.execute("delete from learning_approvals")
    # Reset AUTOINCREMENT sequence so next history record starts at 1
    conn.execute("delete from sqlite_sequence where name='validation_history'")
    conn.commit()
    conn.close()


def decide(comparison: Dict[str, Any], confidence: float, streak: int) -> Tuple[str, str, List[str]]:
    reasons: List[str] = []

    # Hard rule: date-wise hour mismatch must never auto-approve.
    # Total-hours equality alone is not sufficient.
    if comparison.get("mismatches", {}).get("day_hours"):
        return "MANUAL_REVIEW", "MANUAL_REVIEW", ["Date-wise hours mismatch"]

    if not comparison["mismatches"] and comparison["critical_ok"]:
        return "AUTO_APPROVE", "AUTO_APPROVE", reasons
    # Streak is updated only after manual decision is submitted.
    # To auto-approve on the Nth matching submission, trigger when prior streak is N-1.
    trusted_trigger = max(APP_CONFIG["trusted_streak_threshold"] - 1, 0)
    if streak >= trusted_trigger:
        return (
            "AUTO_APPROVE_TRUSTED",
            "AUTO_APPROVE_TRUSTED_TEMPLATE",
            [f"Trusted auto-approve at threshold {APP_CONFIG['trusted_streak_threshold']}"],
        )
    if confidence < APP_CONFIG["confidence_threshold"]:
        reasons.append(f"Low confidence: {confidence:.2f}")
    for k in comparison["mismatches"].keys():
        reasons.append(f"Mismatch in {k}")
    return "MANUAL_REVIEW", "MANUAL_REVIEW", reasons


def build_mismatch_signature(step1: Dict[str, Any], extracted: Dict[str, Any], comparison: Dict[str, Any]) -> str:
    mismatches = comparison.get("mismatches", {}) if isinstance(comparison, dict) else {}
    sig = {
        "employee_name_mismatch": "employee_name" in mismatches,
        "employee_name_blank": not normalize_text(str(extracted.get("employee_name", "") or "")),
        "vendor_mismatch": "vendor" in mismatches,
        "vendor_blank": not normalize_text(str(extracted.get("vendor", "") or "")),
        "company_mismatch": "company" in mismatches,
        "company_blank": not normalize_text(str(extracted.get("company", "") or "")),
        "approval_mismatch": "approved" in mismatches,
        "approval_missing": not normalize_text(str(extracted.get("approved", "") or "")),
        "period_mismatch": ("period_start" in mismatches) or ("period_end" in mismatches),
        "period_partial_in_range": bool(comparison.get("period_coverage_note")),
        "total_hours_mismatch": "total_hours" in mismatches,
        "day_hours_mismatch": "day_hours" in mismatches,
        "step1_period_start": str(step1.get("period_start", "") or ""),
        "step1_period_end": str(step1.get("period_end", "") or ""),
    }
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode("utf-8")).hexdigest()


def get_learning_state(
    employee_id: str,
    vendor: str,
    company: str,
    t_hash: str,
    p_key: str,
    mismatch_signature: str,
) -> Tuple[str, int]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        """
        select last_answer, streak_count
        from learning_approvals
        where employee_id=? and vendor=? and company=? and template_hash=? and pattern_key=? and mismatch_signature=?
        """,
        (employee_id, normalize_text(vendor), normalize_text(company), t_hash, p_key, mismatch_signature),
    ).fetchone()
    conn.close()
    if not row:
        return "", 0
    return str(row[0] or ""), int(row[1] or 0)


def record_learning_answer(
    employee_id: str,
    vendor: str,
    company: str,
    t_hash: str,
    p_key: str,
    mismatch_signature: str,
    answer: str,
    question: str = "",
) -> None:
    prev_answer, prev_streak = get_learning_state(employee_id, vendor, company, t_hash, p_key, mismatch_signature)
    answer_norm = normalize_text(answer)
    new_streak = prev_streak + 1 if prev_answer == answer_norm else 1
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        insert into learning_approvals(employee_id,vendor,company,template_hash,pattern_key,mismatch_signature,last_answer,last_question,streak_count,updated_at)
        values(?,?,?,?,?,?,?,?,?,?)
        on conflict(employee_id,vendor,company,template_hash,pattern_key,mismatch_signature)
        do update set
            last_answer=excluded.last_answer,
            last_question=excluded.last_question,
            streak_count=excluded.streak_count,
            updated_at=excluded.updated_at
        """,
        (
            normalize_text(employee_id),
            normalize_text(vendor),
            normalize_text(company),
            t_hash,
            p_key,
            mismatch_signature,
            answer_norm,
            str(question or ""),
            new_streak,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def build_reason_codes(
    decision: str,
    comparison: Dict[str, Any],
    confidence: float,
    meta: Dict[str, Any],
) -> List[str]:
    codes: List[str] = []
    if decision == "AUTO_APPROVE":
        codes.append("AUTO_APPROVE_ALL_MATCH")
    if decision == "AUTO_APPROVE_TRUSTED":
        codes.append("AUTO_APPROVE_TRUSTED_TEMPLATE")
    if confidence < APP_CONFIG["confidence_threshold"]:
        codes.append("LOW_CONFIDENCE")
    mismatches = comparison.get("mismatches", {})
    if "employee_name" in mismatches:
        codes.append("MISMATCH_EMPLOYEE")
    if "vendor" in mismatches:
        codes.append("MISMATCH_VENDOR")
    if "company" in mismatches:
        codes.append("MISMATCH_COMPANY")
    if "day_hours" in mismatches:
        codes.append("MISMATCH_DAY_HOURS")
    if "total_hours" in mismatches:
        codes.append("MISMATCH_TOTAL_HOURS")
    if "approved" in mismatches:
        codes.append("MISSING_APPROVAL_MARK")
    for issue in meta.get("prevalidation", {}).get("issues", []):
        if "password" in issue.lower():
            codes.append("PRECHECK_PASSWORD_PROTECTED")
        elif "unsupported" in issue.lower():
            codes.append("PRECHECK_UNSUPPORTED_FORMAT")
        else:
            codes.append("PRECHECK_FAILED")
    for issue in meta.get("quality_validation", {}).get("issues", []):
        if "blurry" in issue.lower():
            codes.append("QUALITY_BLURRY")
        elif "low-resolution" in issue.lower():
            codes.append("QUALITY_LOW_RESOLUTION")
        elif "cropped" in issue.lower() or "partial" in issue.lower():
            codes.append("QUALITY_CROPPED_OR_PARTIAL")
        else:
            codes.append("QUALITY_ISSUE")
    # dedupe preserving order
    return list(dict.fromkeys(codes))


def build_submission_hash(step1: Dict[str, Any], file_fingerprint: str) -> str:
    payload = {
        "employee_id": normalize_text(step1.get("employee_id", "")),
        "employee_name": normalize_text(step1.get("employee_name", "")),
        "vendor": normalize_text(step1.get("vendor", "")),
        "company": normalize_text(step1.get("company", "")),
        "duration": step1.get("duration", ""),
        "period_start": step1.get("period_start", ""),
        "period_end": step1.get("period_end", ""),
        "day_hours": sorted(step1.get("day_hours", []), key=lambda x: x.get("date", "")),
        "file_fingerprint": file_fingerprint,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def duplicate_submission_exists(submission_hash: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "select 1 from validation_history where submission_hash=? limit 1",
        (submission_hash,),
    ).fetchone()
    conn.close()
    return bool(row)


def get_setting_int(key: str, default: int) -> int:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("select value from app_settings where key=?", (key,)).fetchone()
    conn.close()
    if not row:
        return default
    try:
        return int(row[0])
    except (TypeError, ValueError):
        return default


def save_setting_int(key: str, value: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        insert into app_settings(key, value)
        values(?, ?)
        on conflict(key) do update set value=excluded.value
        """,
        (key, str(value)),
    )
    conn.commit()
    conn.close()


def get_setting_float(key: str, default: float) -> float:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("select value from app_settings where key=?", (key,)).fetchone()
    conn.close()
    if not row:
        return default
    return safe_float(row[0], default)


def save_setting_float(key: str, value: float) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        insert into app_settings(key, value)
        values(?, ?)
        on conflict(key) do update set value=excluded.value
        """,
        (key, str(float(value))),
    )
    conn.commit()
    conn.close()


def get_setting_text(key: str, default: str = "") -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("select value from app_settings where key=?", (key,)).fetchone()
        conn.close()
        if not row or row[0] is None:
            return default
        return str(row[0])
    except sqlite3.Error:
        return default


def save_setting_text(key: str, value: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        insert into app_settings(key, value)
        values(?, ?)
        on conflict(key) do update set value=excluded.value
        """,
        (key, value),
    )
    conn.commit()
    conn.close()


def load_manual_step1_prefill() -> Dict[str, Any]:
    raw = get_setting_text("manual_step1_prefill", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_manual_step1_prefill(step1: Dict[str, Any]) -> None:
    payload = {
        "employee_id": str(step1.get("employee_id", "") or ""),
        "employee_name": str(step1.get("employee_name", "") or ""),
        "vendor": str(step1.get("vendor", "") or ""),
        "company": str(step1.get("company", "") or ""),
        "duration": str(step1.get("duration", "") or ""),
        "period_start": str(step1.get("period_start", "") or ""),
        "period_end": str(step1.get("period_end", "") or ""),
        "day_hours": step1.get("day_hours", []) if isinstance(step1.get("day_hours"), list) else [],
    }
    save_setting_text("manual_step1_prefill", json.dumps(payload))


def clear_manual_step1_prefill() -> None:
    save_setting_text("manual_step1_prefill", "")


def get_runtime_aws_config() -> Dict[str, str]:
    aws_region = (get_setting_text("aws_region", "").strip() or os.getenv("AWS_REGION", FIXED_AWS_REGION)).strip()
    bedrock_model_id = (
        get_setting_text("bedrock_model_id", "").strip() or os.getenv("BEDROCK_MODEL_ID", FIXED_BEDROCK_MODEL_ID)
    ).strip()
    aws_access_key_id = (get_setting_text("aws_access_key_id", "").strip() or os.getenv("AWS_ACCESS_KEY_ID", "")).strip()
    aws_secret_access_key = (
        get_setting_text("aws_secret_access_key", "").strip() or os.getenv("AWS_SECRET_ACCESS_KEY", "")
    ).strip()
    aws_session_token = (get_setting_text("aws_session_token", "").strip() or os.getenv("AWS_SESSION_TOKEN", "")).strip()
    return {
        "aws_region": aws_region,
        "bedrock_model_id": bedrock_model_id,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "aws_session_token": aws_session_token,
    }


def log_history(
    employee_id: str,
    employee_name: str,
    streak_value: int,
    template_hash_value: str,
    pattern_key_value: str,
    submission_hash: str,
    step1: Dict[str, Any],
    extracted: Dict[str, Any],
    comparison: Dict[str, Any],
    decision: str,
    approval_type: str,
    reasons: List[str],
    reason_codes: List[str],
    meta: Dict[str, Any],
) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        insert into validation_history(created_at,employee_id,employee_name,streak_value,template_hash,pattern_key,submission_hash,step1_json,extracted_json,comparison_json,decision,approval_type,reasons_json,reason_codes_json,aws_meta_json)
        values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.utcnow().isoformat(),
            employee_id,
            employee_name,
            streak_value,
            template_hash_value,
            pattern_key_value,
            submission_hash,
            json.dumps(step1),
            json.dumps(extracted),
            json.dumps(comparison),
            decision,
            approval_type,
            json.dumps(reasons),
            json.dumps(reason_codes),
            json.dumps(meta),
        ),
    )
    conn.commit()
    conn.close()


def recent_history(limit: int = 20) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        select id, created_at, employee_id, employee_name, streak_value, template_hash, pattern_key, decision, approval_type, reasons_json, reason_codes_json
        from validation_history
        order by id desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        reasons = []
        try:
            reasons = json.loads(r[9]) if r[9] else []
        except json.JSONDecodeError:
            reasons = [str(r[9])]
        out.append(
            {
                "record_id": r[0],
                "created_at": r[1],
                "employee_id": r[2] or "",
                "employee_name": r[3] or "",
                "streak": r[4] if r[4] is not None else 0,
                "template_hash": (r[5] or "")[:12],
                "pattern_key": (r[6] or "")[:40],
                "decision": r[7],
                "reasons": "; ".join(reasons) if reasons else "",
            }
        )
    return out


def recent_learning_memory(limit: int = 50) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        select employee_id, vendor, company, mismatch_signature, last_question, last_answer, streak_count, updated_at
        from learning_approvals
        order by updated_at desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    out: List[Dict[str, Any]] = []
    for r in rows:
        sig = str(r[3] or "")
        out.append(
            {
                "employee_id": r[0] or "",
                "vendor": r[1] or "",
                "company": r[2] or "",
                "mismatch_signature": f"{sig[:14]}...{sig[-8:]}" if len(sig) > 26 else sig,
                "question_short": r[4] or "",
                "last_answer": r[5] or "",
                "streak_count": r[6] if r[6] is not None else 0,
                "updated_at": r[7] or "",
            }
        )
    return out


def date_range(start: date, end: date) -> List[str]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def format_date_with_day(iso_date: str) -> str:
    try:
        dt = datetime.strptime(iso_date, "%Y-%m-%d")
        return f"{iso_date} ({dt.strftime('%a')})"
    except ValueError:
        return iso_date


def row_html(field: str, left: Any, right: Any, matched: bool) -> str:
    color = "#e8f7e8" if matched else "#fdeaea"
    status = "Match" if matched else "Mismatch"
    return f"<tr style='background:{color}'><td>{field}</td><td>{left}</td><td>{right}</td><td>{status}</td></tr>"


def parse_iso_date_optional(value: str) -> date | None:
    try:
        return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def duration_days(duration: str) -> int:
    mapping = {
        "Weekly": 7,
        "Bi-Weekly": 14,
        "Semi-Monthly": 15,
        "Monthly": 30,
    }
    return mapping.get(duration, 7)


def infer_duration_label(normalized: Dict[str, Any]) -> str:
    raw = normalize_text(str(normalized.get("duration", "")))
    if "semi" in raw and "month" in raw:
        return "Semi-Monthly"
    if ("bi" in raw and "week" in raw) or "bi-weekly" in raw:
        return "Bi-Weekly"
    if "monthly" in raw or ("month" in raw and "semi" not in raw):
        return "Monthly"
    if "weekly" in raw:
        return "Weekly"

    ps = parse_iso_date_optional(str(normalized.get("period_start", "")))
    pe = parse_iso_date_optional(str(normalized.get("period_end", "")))
    if ps and pe and pe >= ps:
        span = (pe - ps).days + 1
        if span <= 8:
            return "Weekly"
        if span <= 14:
            return "Bi-Weekly"
        if span <= 16:
            return "Semi-Monthly"
        return "Monthly"
    return ""


def _is_summary_like_day_hours(normalized: Dict[str, Any]) -> bool:
    rows = normalized.get("day_hours", []) if isinstance(normalized.get("day_hours"), list) else []
    if not rows:
        return False
    values = [safe_float(r.get("hours", None), None) for r in rows if isinstance(r, dict)]
    values = [v for v in values if v is not None]
    if not values:
        return False
    hdr_tokens: List[str] = []
    if isinstance(normalized.get("headers"), list):
        hdr_tokens.extend(str(x) for x in normalized.get("headers") if x is not None)
    if isinstance(normalized.get("table_columns"), list):
        hdr_tokens.extend(str(x) for x in normalized.get("table_columns") if x is not None)
    hdr_text = normalize_text(" ".join(hdr_tokens))
    has_week_end_header = ("week end" in hdr_text) or ("week ending" in hdr_text) or ("week_end" in hdr_text)
    has_weekly_total_values = any(v > 16.0 for v in values)
    parsed_dates: List[date] = []
    for r in rows:
        if isinstance(r, dict):
            d = parse_iso_date_optional(str(r.get("date", "")))
            if d:
                parsed_dates.append(d)
    parsed_dates = sorted(set(parsed_dates))
    diffs = [(parsed_dates[i + 1] - parsed_dates[i]).days for i in range(len(parsed_dates) - 1)] if len(parsed_dates) >= 2 else []
    weekly_cadence = bool(diffs) and (sum(1 for x in diffs if x in {6, 7, 8}) >= max(1, int(len(diffs) * 0.6)))

    # Week-end header alone is not enough (some templates also contain daily drilldown rows).
    # Treat as summary-like only when values look weekly totals OR dates follow weekly cadence.
    return len(rows) <= 10 and (has_weekly_total_values or (has_week_end_header and weekly_cadence))


def apply_autofill_to_form(normalized: Dict[str, Any]) -> None:
    st.session_state["employee_name"] = normalized.get("employee_name", "") or st.session_state.get("employee_name", "")
    st.session_state["vendor"] = normalized.get("vendor", "") or st.session_state.get("vendor", "")
    st.session_state["company"] = normalized.get("company", "") or st.session_state.get("company", "")

    inferred_duration = infer_duration_label(normalized)
    if inferred_duration:
        st.session_state["duration"] = inferred_duration

    extracted_hours_map: Dict[str, float] = {}
    summary_like = _is_summary_like_day_hours(normalized)
    for row in normalized.get("day_hours", []):
        dt = row.get("date")
        if not dt:
            continue
        parsed_hours = safe_float(row.get("hours", 0), 0.0)
        # For summary-style rows (weekly totals), don't inject totals into one day input.
        if summary_like and parsed_hours > 16.0:
            parsed_hours = 0.0
        parsed_hours = max(0.0, min(24.0, parsed_hours))
        extracted_hours_map[dt] = parsed_hours

    # Keep only duration-window dates from extracted data.
    extracted_period_end = parse_iso_date_optional(normalized.get("period_end", ""))
    extracted_period_start = parse_iso_date_optional(normalized.get("period_start", ""))
    extracted_dates = []
    for d in extracted_hours_map.keys():
        parsed = parse_iso_date_optional(d)
        if parsed:
            extracted_dates.append(parsed)

    if extracted_period_start and extracted_period_end and extracted_period_end >= extracted_period_start:
        start_date = extracted_period_start
        end_date = extracted_period_end
    elif extracted_period_end:
        end_date = extracted_period_end
        days = duration_days(st.session_state.get("duration", "Weekly"))
        start_date = end_date - timedelta(days=days - 1)
    elif extracted_period_start:
        start_date = extracted_period_start
        days = duration_days(st.session_state.get("duration", "Weekly"))
        end_date = start_date + timedelta(days=days - 1)
    elif extracted_dates:
        start_date = min(extracted_dates)
        end_date = max(extracted_dates)
    else:
        start_date = st.session_state.get("period_start") or (date.today() - timedelta(days=6))
        end_date = st.session_state.get("period_end") or date.today()

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    st.session_state["period_start"] = start_date
    st.session_state["period_end"] = end_date

    filtered_hours_map: Dict[str, float] = {}
    for dt in date_range(start_date, end_date):
        filtered_hours_map[dt] = extracted_hours_map.get(dt, 0.0)
        st.session_state[f"hr_{dt}"] = filtered_hours_map[dt]
    st.session_state["autofill_hours_map"] = filtered_hours_map


def enable_browser_autocomplete() -> None:
    """Re-enable native browser autofill on Streamlit inputs (Streamlit often sets autocomplete=off)."""
    components.html(
        """
<script>
(function () {
  const doc = window.parent.document;
  if (!doc.body) return;
  function labelText(el) {
    if (!el) return "";
    return (el.innerText || "").trim().split("\\n")[0].trim();
  }
  function patch() {
    doc.querySelectorAll('[data-testid="stTextInput"]').forEach(function (container) {
      var labelEl = container.querySelector('[data-testid="stWidgetLabel"]');
      var input = container.querySelector("input:not([type=hidden])");
      if (!labelEl || !input) return;
      var label = labelText(labelEl);
      input.setAttribute("autocomplete", "on");
      if (label === "Name") {
        input.setAttribute("name", "slicehrms-timesheet-employee-name");
        input.setAttribute("autocomplete", "name");
      } else if (label === "Vendor") {
        input.setAttribute("name", "slicehrms-timesheet-vendor");
      } else if (label === "Company") {
        input.setAttribute("name", "slicehrms-timesheet-company");
      }
    });
    doc.querySelectorAll('[data-testid="stTextArea"]').forEach(function (container) {
      var labelEl = container.querySelector('[data-testid="stWidgetLabel"]');
      var ta = container.querySelector("textarea");
      if (!labelEl || !ta) return;
      var label = labelText(labelEl);
      if (label === "Comment") {
        ta.setAttribute("autocomplete", "on");
        ta.setAttribute("name", "slicehrms-timesheet-manual-comment");
      }
    });
    doc.querySelectorAll('[data-testid="stNumberInput"]').forEach(function (container) {
      var labelEl = container.querySelector('[data-testid="stWidgetLabel"]');
      var input = container.querySelector("input:not([type=hidden])");
      if (!labelEl || !input) return;
      var label = labelText(labelEl);
      if (/^\\d{4}-\\d{2}-\\d{2}$/.test(label)) {
        input.setAttribute("autocomplete", "on");
        input.setAttribute("name", "slicehrms-timesheet-hours-" + label);
      }
    });
  }
  patch();
  var obs = new MutationObserver(patch);
  obs.observe(doc.body, { childList: true, subtree: true });
})();
</script>
        """,
        height=0,
    )


def force_sidebar_collapsed() -> None:
    """Force sidebar to collapsed state on every rerun."""
    components.html(
        """
<script>
(function () {
  const doc = window.parent.document;
  if (!doc || !doc.body) return;
  function collapse() {
    const closeBtn = doc.querySelector('button[aria-label="Close sidebar"]');
    if (closeBtn) closeBtn.click();
  }
  // Try immediately and shortly after render updates.
  collapse();
  setTimeout(collapse, 120);
  setTimeout(collapse, 400);
})();
</script>
        """,
        height=0,
    )


def reset_pipeline_state(clear_source: bool = False) -> None:
    st.session_state.validation_result = None
    st.session_state.last_upload_fingerprint = None
    st.session_state.last_submission_hash = None
    if clear_source:
        st.session_state["step2_llm_preview_cache_v2"] = ""
        st.session_state["step2_llm_preview_rows_v2"] = []
        st.session_state.source_file_bytes = None
        st.session_state.source_file_name = ""
        st.session_state.source_file_fingerprint = None


def prepare_next_document() -> None:
    """Clear current processed/uploaded state and reset upload widgets for next file."""
    reset_pipeline_state(clear_source=True)
    st.session_state.pending_step1_reset = True
    st.session_state.autofill_uploader_key += 1
    st.session_state.uploader_key += 1
    st.session_state.autofill_extraction_preview = None


def reset_step1_form_state() -> None:
    # Use pop/reset so this can be safely applied before widget creation.
    st.session_state.pop("employee_id", None)
    st.session_state.pop("employee_name", None)
    st.session_state.pop("vendor", None)
    st.session_state.pop("company", None)
    st.session_state.pop("duration", None)
    st.session_state.pop("period_start", None)
    st.session_state.pop("period_end", None)
    st.session_state["autofill_hours_map"] = {}
    st.session_state["autofill_last_signature"] = ""
    st.session_state["autofill_last_fingerprint"] = None
    # Remove all day-hour inputs currently in session.
    for key in list(st.session_state.keys()):
        if key.startswith("hr_"):
            st.session_state.pop(key, None)


def main() -> None:
    init_db()
    st.set_page_config(page_title="Timesheet Approval POC", layout="wide", initial_sidebar_state="collapsed")
    force_sidebar_collapsed()
    st.title("Timesheet Approval POC")
    current_threshold = get_setting_int("trusted_streak_threshold", APP_CONFIG["trusted_streak_threshold"])
    current_threshold = max(1, min(6, current_threshold))
    APP_CONFIG["trusted_streak_threshold"] = current_threshold
    assumed_out_tok = get_setting_int("llm_assumed_output_tokens", APP_CONFIG["llm_assumed_output_tokens"])
    assumed_out_tok = max(0, min(32000, assumed_out_tok))
    APP_CONFIG["llm_assumed_output_tokens"] = assumed_out_tok

    page = st.sidebar.selectbox("Navigation", ["Approval", "Settings"])
    if page == "Settings":
        st.subheader("Settings")
        st.caption("Configure trusted template auto-approval threshold.")
        threshold_value = st.number_input(
            "Trusted Streak Threshold (manual approvals)",
            min_value=1,
            max_value=6,
            value=APP_CONFIG["trusted_streak_threshold"],
            step=1,
            help="If threshold is 3, then after 3 manual approvals of same pattern, next one can auto-approve.",
        )
        if st.button("Save Settings", type="primary"):
            save_setting_int("trusted_streak_threshold", int(threshold_value))
            APP_CONFIG["trusted_streak_threshold"] = int(threshold_value)
            st.success("Settings saved.")
        st.info(
            f"Current value: {APP_CONFIG['trusted_streak_threshold']} "
            f"(max 6). This controls trusted auto-approval behavior."
        )

        st.divider()
        st.subheader("AWS Runtime Configuration")
        st.caption("Set AWS runtime values here. Default region/model are pre-filled for convenience.")
        runtime_cfg = get_runtime_aws_config()
        aws_region_value = st.text_input(
            "AWS Region",
            value=runtime_cfg.get("aws_region", FIXED_AWS_REGION),
            key="set_aws_region",
            help=f"Default: {FIXED_AWS_REGION}",
        )
        bedrock_model_value = st.text_input(
            "Bedrock Model ID",
            value=runtime_cfg.get("bedrock_model_id", FIXED_BEDROCK_MODEL_ID),
            key="set_bedrock_model_id",
            help=f"Default: {FIXED_BEDROCK_MODEL_ID}",
        )
        saved_access = get_setting_text("aws_access_key_id", "").strip()
        saved_secret = get_setting_text("aws_secret_access_key", "").strip()
        saved_token = get_setting_text("aws_session_token", "").strip()
        st.caption("Enter credentials only when needed. Leave inputs blank to keep current saved values.")
        if saved_access:
            st.success(f"AWS Access Key ID: Saved ({mask_secret(saved_access, keep_last=4)})")
        else:
            st.warning("AWS Access Key ID: Not saved")
        if saved_secret:
            st.success(f"AWS Secret Access Key: Saved ({mask_secret(saved_secret, keep_last=4)})")
        else:
            st.warning("AWS Secret Access Key: Not saved")
        if saved_token:
            st.success(f"AWS Session Token: Saved ({mask_secret(saved_token, keep_last=6)})")
        else:
            st.info("AWS Session Token: Not saved (optional)")
        aws_access_key_input = st.text_input(
            "AWS Access Key ID",
            value="",
            type="password",
            key="set_aws_access_key_id",
            placeholder="Leave blank to keep existing",
        )
        aws_secret_key_input = st.text_input(
            "AWS Secret Access Key",
            value="",
            type="password",
            key="set_aws_secret_access_key",
            placeholder="Leave blank to keep existing",
        )
        aws_session_token_input = st.text_input(
            "AWS Session Token (optional)",
            value="",
            type="password",
            key="set_aws_session_token",
            placeholder="Leave blank to keep existing",
        )

        if st.button("Save AWS Configuration"):
            save_setting_text("aws_region", (aws_region_value or FIXED_AWS_REGION).strip())
            save_setting_text("bedrock_model_id", (bedrock_model_value or FIXED_BEDROCK_MODEL_ID).strip())
            if (aws_access_key_input or "").strip():
                save_setting_text("aws_access_key_id", aws_access_key_input.strip())
            if (aws_secret_key_input or "").strip():
                save_setting_text("aws_secret_access_key", aws_secret_key_input.strip())
            if (aws_session_token_input or "").strip():
                save_setting_text("aws_session_token", aws_session_token_input.strip())
            st.success("AWS configuration saved.")
            st.rerun()

        st.caption(
            "LLM preview: output token count is unknown until the model responds, so an assumed value is used. "
            "Pricing for estimates comes from **LLM cost estimate (pricing)** below (auto table or your custom rates)."
        )
        assumed_preview = st.number_input(
            "Assumed output tokens (cost preview only)",
            min_value=0,
            max_value=32000,
            value=int(APP_CONFIG.get("llm_assumed_output_tokens", 1200)),
            step=50,
            key="set_llm_assumed_output_tokens",
            help="Higher values raise the estimated charge before each Bedrock call. Does not cap the model.",
        )
        if st.button("Save LLM preview setting"):
            save_setting_int("llm_assumed_output_tokens", int(assumed_preview))
            APP_CONFIG["llm_assumed_output_tokens"] = max(0, min(32000, int(assumed_preview)))
            st.success("Assumed output tokens saved.")
            st.rerun()

        st.divider()
        st.subheader("LLM cost estimate (pricing)")
        st.caption(
            "**Auto** uses a built-in USD table keyed off **Bedrock Model ID** (different models → different rates). "
            "**Custom USD** / **Custom credits** ignore that table and use your numbers per **1 million** input and output tokens. "
            "Per-token cost is rate ÷ 1,000,000 (e.g. USD 3 / 1M input → USD 0.000003 per input token)."
        )
        mode_options = ["auto", "custom_usd", "custom_credits"]
        cur_mode = (get_setting_text("llm_pricing_mode", "auto").strip().lower() or "auto")
        if cur_mode not in mode_options:
            cur_mode = "auto"
        mode_index = mode_options.index(cur_mode)
        pricing_mode_sel = st.selectbox(
            "Pricing basis for estimates",
            options=mode_options,
            index=mode_index,
            format_func=lambda x: {
                "auto": "Built-in USD table (by Bedrock model ID above)",
                "custom_usd": "Custom USD per 1M input and 1M output tokens",
                "custom_credits": "Custom internal credits per 1M input and 1M output tokens",
            }[x],
            key="llm_pricing_mode_select",
        )
        custom_usd_in = st.number_input(
            "Custom USD per 1M input tokens",
            min_value=0.0,
            value=float(get_setting_float("llm_custom_input_usd_per_mtok", 3.0)),
            step=0.01,
            format="%.4f",
            key="llm_custom_in_usd",
            disabled=(pricing_mode_sel != "custom_usd"),
        )
        custom_usd_out = st.number_input(
            "Custom USD per 1M output tokens",
            min_value=0.0,
            value=float(get_setting_float("llm_custom_output_usd_per_mtok", 15.0)),
            step=0.01,
            format="%.4f",
            key="llm_custom_out_usd",
            disabled=(pricing_mode_sel != "custom_usd"),
        )
        custom_cr_in = st.number_input(
            "Custom credits per 1M input tokens",
            min_value=0.0,
            value=float(get_setting_float("llm_custom_input_credits_per_mtok", 100.0)),
            step=1.0,
            format="%.2f",
            key="llm_custom_in_credits",
            disabled=(pricing_mode_sel != "custom_credits"),
        )
        custom_cr_out = st.number_input(
            "Custom credits per 1M output tokens",
            min_value=0.0,
            value=float(get_setting_float("llm_custom_output_credits_per_mtok", 500.0)),
            step=1.0,
            format="%.2f",
            key="llm_custom_out_credits",
            disabled=(pricing_mode_sel != "custom_credits"),
        )
        if st.button("Save LLM pricing"):
            save_setting_text("llm_pricing_mode", pricing_mode_sel.strip())
            save_setting_float("llm_custom_input_usd_per_mtok", float(custom_usd_in))
            save_setting_float("llm_custom_output_usd_per_mtok", float(custom_usd_out))
            save_setting_float("llm_custom_input_credits_per_mtok", float(custom_cr_in))
            save_setting_float("llm_custom_output_credits_per_mtok", float(custom_cr_out))
            st.success("LLM pricing settings saved.")
            st.rerun()

        if st.button("Clear Saved AWS Credentials"):
            save_setting_text("aws_access_key_id", "")
            save_setting_text("aws_secret_access_key", "")
            save_setting_text("aws_session_token", "")
            st.success("Saved AWS credentials cleared. Environment/.env fallback will be used.")
            st.rerun()
        return

    if "validation_result" not in st.session_state:
        st.session_state.validation_result = None
    if "last_upload_fingerprint" not in st.session_state:
        st.session_state.last_upload_fingerprint = None
    if "last_submission_hash" not in st.session_state:
        st.session_state.last_submission_hash = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 1
    if "autofill_uploader_key" not in st.session_state:
        st.session_state.autofill_uploader_key = 1
    if "autofill_last_fingerprint" not in st.session_state:
        st.session_state.autofill_last_fingerprint = None
    if "autofill_last_signature" not in st.session_state:
        st.session_state.autofill_last_signature = ""
    if "autofill_hours_map" not in st.session_state:
        st.session_state.autofill_hours_map = {}
    if "autofill_extraction_preview" not in st.session_state:
        st.session_state.autofill_extraction_preview = None
    if "source_file_bytes" not in st.session_state:
        st.session_state.source_file_bytes = None
    if "source_file_name" not in st.session_state:
        st.session_state.source_file_name = ""
    if "source_file_fingerprint" not in st.session_state:
        st.session_state.source_file_fingerprint = None
    if "entry_mode" not in st.session_state:
        st.session_state.entry_mode = "Auto-Fill from Uploaded Timesheet"
    if "enable_duplicate_check" not in st.session_state:
        st.session_state.enable_duplicate_check = False
    if "manual_prefill_loaded" not in st.session_state:
        st.session_state.manual_prefill_loaded = False
    if "pending_step1_reset" not in st.session_state:
        st.session_state.pending_step1_reset = False
    if "employee_name" not in st.session_state:
        st.session_state.employee_name = ""
    if "employee_id" not in st.session_state:
        st.session_state.employee_id = ""
    if "vendor" not in st.session_state:
        st.session_state.vendor = ""
    if "company" not in st.session_state:
        st.session_state.company = ""
    if "duration" not in st.session_state:
        st.session_state.duration = "Weekly"
    if "period_start" not in st.session_state:
        st.session_state.period_start = None
    if "period_end" not in st.session_state:
        st.session_state.period_end = None

    # Apply deferred reset before widgets are instantiated.
    if st.session_state.pending_step1_reset:
        reset_step1_form_state()
        st.session_state.pending_step1_reset = False
        if "employee_name" not in st.session_state:
            st.session_state.employee_name = ""
        if "employee_id" not in st.session_state:
            st.session_state.employee_id = ""
        if "vendor" not in st.session_state:
            st.session_state.vendor = ""
        if "company" not in st.session_state:
            st.session_state.company = ""
        if "duration" not in st.session_state:
            st.session_state.duration = "Weekly"
        if "period_start" not in st.session_state:
            st.session_state.period_start = None
        if "period_end" not in st.session_state:
            st.session_state.period_end = None

    enable_browser_autocomplete()
    st.subheader("Step 1 - Timesheet Details")
    st.checkbox(
        "Enable duplicate check",
        key="enable_duplicate_check",
        help="When enabled, same employee/period/file submission is blocked. Keep OFF for repeated test runs.",
    )
    st.radio(
        "Entry Mode",
        ["Manual Entry", "Auto-Fill from Uploaded Timesheet"],
        key="entry_mode",
        horizontal=True,
    )

    if st.session_state.entry_mode == "Manual Entry" and not st.session_state.manual_prefill_loaded:
        cached = load_manual_step1_prefill()
        if cached:
            st.session_state["employee_id"] = str(cached.get("employee_id", "") or "")
            st.session_state["employee_name"] = str(cached.get("employee_name", "") or "")
            st.session_state["vendor"] = str(cached.get("vendor", "") or "")
            st.session_state["company"] = str(cached.get("company", "") or "")
            cached_duration = str(cached.get("duration", "") or "")
            if cached_duration in ["Weekly", "Bi-Weekly", "Semi-Monthly", "Monthly"]:
                st.session_state["duration"] = cached_duration
            cached_ps = parse_iso_date_optional(str(cached.get("period_start", "") or ""))
            cached_pe = parse_iso_date_optional(str(cached.get("period_end", "") or ""))
            st.session_state["period_start"] = cached_ps
            st.session_state["period_end"] = cached_pe
            if cached_ps and cached_pe and cached_pe >= cached_ps:
                saved_map = {
                    str(x.get("date", "")): safe_float(x.get("hours", 0.0), 0.0)
                    for x in (cached.get("day_hours", []) if isinstance(cached.get("day_hours"), list) else [])
                    if isinstance(x, dict) and x.get("date")
                }
                for dt in date_range(cached_ps, cached_pe):
                    st.session_state[f"hr_{dt}"] = saved_map.get(dt, 0.0)
                st.session_state["autofill_hours_map"] = {
                    dt: st.session_state.get(f"hr_{dt}", 0.0) for dt in date_range(cached_ps, cached_pe)
                }
        st.session_state.manual_prefill_loaded = True

    if st.session_state.entry_mode == "Manual Entry":
        if st.button("Clear Saved Step 1 Values"):
            clear_manual_step1_prefill()
            reset_step1_form_state()
            st.session_state["employee_id"] = ""
            st.session_state["employee_name"] = ""
            st.session_state["vendor"] = ""
            st.session_state["company"] = ""
            st.session_state["duration"] = "Weekly"
            st.session_state["period_start"] = None
            st.session_state["period_end"] = None
            st.session_state.manual_prefill_loaded = True
            st.success("Saved Step 1 values cleared.")
            st.rerun()

    if st.session_state.entry_mode == "Auto-Fill from Uploaded Timesheet":
        st.info("Upload a reference timesheet to auto-fill fields. You can edit values before final validation.")
        autofill_file = st.file_uploader(
            "Auto-Fill Source (PDF/Image/Text/Doc)",
            type=["pdf", "png", "jpg", "jpeg", "txt", "doc", "docx"],
            key=f"autofill_{st.session_state.autofill_uploader_key}",
        )
        if autofill_file:
            autofill_bytes = autofill_file.getvalue()
            autofill_fingerprint = hashlib.sha256(autofill_bytes).hexdigest()
            autofill_signature = f"{autofill_fingerprint}:{AUTOFILL_LOGIC_VERSION}"
            previous_source_fingerprint = st.session_state.source_file_fingerprint

            # New uploaded source should start fresh: hide previous processed result
            # and clear old Step1 values before refilling from new file.
            if previous_source_fingerprint and previous_source_fingerprint != autofill_fingerprint:
                reset_pipeline_state(clear_source=False)
                st.session_state.autofill_extraction_preview = None
                st.session_state["employee_id"] = ""
                st.session_state["employee_name"] = ""
                st.session_state["vendor"] = ""
                st.session_state["company"] = ""
                st.session_state["duration"] = "Weekly"
                st.session_state["period_start"] = None
                st.session_state["period_end"] = None
                st.session_state["autofill_hours_map"] = {}
                for key in list(st.session_state.keys()):
                    if key.startswith("hr_"):
                        st.session_state.pop(key, None)

            st.session_state.source_file_bytes = autofill_bytes
            st.session_state.source_file_name = autofill_file.name
            st.session_state.source_file_fingerprint = autofill_fingerprint
            if st.session_state.autofill_last_signature != autofill_signature:
                try:
                    progress = st.progress(10, text="Auto-fill: extracting data...")
                    autofill_extraction = extract_and_normalize(
                        autofill_bytes,
                        autofill_file.name,
                        hints={"duration": "", "period_start": "", "period_end": ""},
                    )
                    normalized = autofill_extraction["normalized"]
                    apply_autofill_to_form(normalized)
                    st.session_state.autofill_extraction_preview = autofill_extraction
                    st.session_state.autofill_last_fingerprint = autofill_fingerprint
                    st.session_state.autofill_last_signature = autofill_signature
                    progress.progress(100, text="Auto-fill completed")
                    if normalized.get("error"):
                        st.error(user_friendly_error(normalized["error"]))
                    else:
                        st.success("Fields auto-filled. Review/edit below.")
                        st.rerun()
                except Exception as exc:
                    st.session_state.autofill_extraction_preview = None
                    st.error(user_friendly_error(str(exc)))
        else:
            # File removed from Step 1: clear previous source + processed pipeline UI.
            if st.session_state.source_file_bytes is not None:
                reset_pipeline_state(clear_source=True)
            # Ensure re-upload of the same file triggers fresh autofill.
            st.session_state.autofill_last_signature = ""
            st.session_state.autofill_last_fingerprint = None
            st.session_state.autofill_extraction_preview = None

    c1, c2 = st.columns(2)
    with c1:
        employee_id = st.text_input("Employee ID", key="employee_id")
        employee_name = st.text_input("Name", key="employee_name")
        vendor = st.text_input("Vendor", key="vendor")
        company = st.text_input("Company", key="company")
        durations = ["Weekly", "Bi-Weekly", "Semi-Monthly", "Monthly"]
        duration_idx = durations.index(st.session_state.duration) if st.session_state.duration in durations else 0
        duration = st.selectbox("Time Duration", durations, index=duration_idx, key="duration")
    with c2:
        period_start = st.date_input("Period Start", key="period_start")
        period_end = st.date_input("Period End", key="period_end")

    dates = []
    if isinstance(period_start, date) and isinstance(period_end, date) and period_end >= period_start:
        dates = date_range(period_start, period_end)

    # Important: do not overwrite hr_* values on every rerun in Auto-Fill mode.
    # Users must be able to manually edit Step 1 hours after autofill.

    st.subheader("Step 1 Date-wise Hours")
    day_hours: List[Dict[str, Any]] = []
    if not dates:
        st.info("Select Period Start and Period End to enter day-wise hours.")
    else:
        col_count = 4
        for row_start in range(0, len(dates), col_count):
            row_dates = dates[row_start : row_start + col_count]
            cols = st.columns(col_count)
            for col_idx, dt in enumerate(row_dates):
                hr_key = f"hr_{dt}"
                if hr_key not in st.session_state:
                    v = safe_float(st.session_state.autofill_hours_map.get(dt, 0.0), 0.0)
                    st.session_state[hr_key] = max(0.0, min(24.0, v))
                with cols[col_idx]:
                    input_value = safe_float(st.session_state.get(hr_key, 0.0), 0.0)
                    input_value = max(0.0, min(24.0, input_value))
                    hrs = st.number_input(
                        format_date_with_day(dt),
                        min_value=0.0,
                        max_value=24.0,
                        value=input_value,
                        step=0.5,
                        format="%.2f",
                        key=hr_key,
                    )
                    day_hours.append({"date": dt, "hours": float(hrs)})
        st.caption(f"Total Hours (Step 1): {sum(x['hours'] for x in day_hours):.2f}")

    if st.session_state.entry_mode == "Auto-Fill from Uploaded Timesheet" and st.session_state.get("autofill_extraction_preview"):
        preview = st.session_state.autofill_extraction_preview
        with st.expander("Data Extraction (Uploaded File)", expanded=False):
            st.json(preview.get("normalized", {}))
            ocr_preview = (preview.get("ocr_text") or "").strip()
            if ocr_preview:
                st.caption("OCR text preview (first 1200 chars)")
                st.code(ocr_preview[:1200])

    st.subheader("Step 2 - Validate and Decide")
    st.caption(
        "Estimated usage updates automatically when files or the period above change (text is extracted first). "
        "Use **Process Next** below the estimate to run the full validation and model step."
    )
    process_clicked = False
    selected_files: List[Dict[str, Any]] = []
    manual_files = None
    if st.session_state.entry_mode == "Manual Entry":
        st.caption("Upload one or more timesheet files and process together.")
        manual_files = st.file_uploader(
            "Upload PDF/Image/Text/Doc file(s)",
            type=["pdf", "png", "jpg", "jpeg", "txt", "doc", "docx"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
    else:
        st.caption("Process the file already uploaded in Step 1.")

    step1_ps_text_pv = period_start.isoformat() if isinstance(period_start, date) else ""
    step1_pe_text_pv = period_end.isoformat() if isinstance(period_end, date) else ""
    hints_preview: Dict[str, Any] = {
        "duration": duration,
        "period_start": step1_ps_text_pv,
        "period_end": step1_pe_text_pv,
    }
    preview_targets: List[Dict[str, Any]] = []
    if st.session_state.entry_mode == "Manual Entry":
        if manual_files:
            for idx, f in enumerate(manual_files):
                preview_targets.append({"name": f.name or f"file_{idx + 1}", "bytes": f.getvalue()})
    elif st.session_state.source_file_bytes is not None:
        preview_targets.append(
            {
                "name": st.session_state.source_file_name or "source_upload",
                "bytes": st.session_state.source_file_bytes,
            }
        )

    CACHE_K = "step2_llm_preview_cache_v2"
    ROWS_K = "step2_llm_preview_rows_v2"
    if not preview_targets:
        st.session_state[CACHE_K] = ""
        st.session_state[ROWS_K] = []
    else:
        fps = [hashlib.sha256(x["bytes"]).hexdigest() + "::" + str(x.get("name", "")) for x in preview_targets]
        cache_payload = {
            "entry": st.session_state.entry_mode,
            "fps": sorted(fps),
            "hints": hints_preview,
            "pf": _llm_preview_pricing_fingerprint(),
        }
        cache_key = hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()
        if st.session_state.get(CACHE_K) != cache_key:
            with st.spinner("Updating estimate from uploaded document(s)…"):
                st.session_state[ROWS_K] = build_compact_llm_preview_rows(preview_targets, hints_preview)
            st.session_state[CACHE_K] = cache_key

    rows_display = st.session_state.get(ROWS_K) or []
    if rows_display:
        st.markdown("**Estimated LLM usage** (tokens & charge)")
        st.caption(LLM_PREVIEW_DEFAULT_FOOTNOTE)
        df_pv = pd.DataFrame(rows_display)
        try:
            ct = LLM_PREVIEW_TABLE_LABELS["col_total"]
            cc: Dict[str, Any] = {
                "File": st.column_config.TextColumn("File", width="small"),
                "Model": st.column_config.TextColumn("Model", width="small"),
                ct: st.column_config.TextColumn(ct, width="small"),
            }
            st.dataframe(
                df_pv,
                use_container_width=True,
                hide_index=True,
                column_config=cc,
            )
        except Exception:
            st.dataframe(df_pv, use_container_width=True, hide_index=True)

    if st.session_state.entry_mode == "Manual Entry":
        process_clicked = st.button("Process Next", type="primary")
        if process_clicked:
            if not manual_files:
                st.warning("Please upload at least one file first.")
                return
            selected_files = []
            for f in manual_files:
                b = f.getvalue()
                fp = hashlib.sha256(b).hexdigest()
                selected_files.append({"name": f.name, "bytes": b, "fingerprint": fp})
    else:
        process_clicked = st.button("Process Next", type="primary")
        if process_clicked:
            if st.session_state.source_file_bytes is None:
                st.warning("Please upload a file in Step 1 first.")
                return
            file_bytes = st.session_state.source_file_bytes
            file_name = st.session_state.source_file_name or "source_upload"
            fingerprint = st.session_state.source_file_fingerprint or hashlib.sha256(file_bytes).hexdigest()
            selected_files = [{"name": file_name, "bytes": file_bytes, "fingerprint": fingerprint}]

    if process_clicked and selected_files:
        if not normalize_text(employee_id):
            st.warning("Employee ID is required in Step 1.")
            return
        fingerprint = hashlib.sha256(
            "||".join(sorted(x["fingerprint"] for x in selected_files)).encode("utf-8")
        ).hexdigest()
        step1 = {
            "employee_id": employee_id,
            "employee_name": employee_name,
            "vendor": vendor,
            "company": company,
            "duration": duration,
            "period_start": period_start.isoformat() if isinstance(period_start, date) else "",
            "period_end": period_end.isoformat() if isinstance(period_end, date) else "",
            "day_hours": day_hours,
        }
        if st.session_state.entry_mode == "Manual Entry":
            save_manual_step1_prefill(step1)
        submission_hash = build_submission_hash(step1, fingerprint)
        # Always process on every Process Next click, even when Step1/files are unchanged.
        st.session_state.validation_result = None
        progress = st.progress(5, text="Reading file...")
        step1_ps_text = period_start.isoformat() if isinstance(period_start, date) else ""
        step1_pe_text = period_end.isoformat() if isinstance(period_end, date) else ""
        hints = {
            "duration": duration,
            "period_start": step1_ps_text,
            "period_end": step1_pe_text,
        }
        forced_manual_reasons: List[str] = []
        per_file_status: List[Dict[str, Any]] = []
        per_file_included_hours_total = 0.0
        is_manual_multi = st.session_state.entry_mode == "Manual Entry" and len(selected_files) > 1

        if is_manual_multi:
            if not employee_name.strip() or not step1_ps_text or not step1_pe_text:
                st.warning("For multi-file Manual Entry, fill Employee Name, Period Start, and Period End in Step 1 first.")
                return
            progress.progress(20, text="Running extraction for uploaded files...")
            step1_start = parse_iso_date_optional(step1_ps_text)
            step1_end = parse_iso_date_optional(step1_pe_text)
            step1_emp_canon = canonical_person_name(employee_name)
            step1_vendor_canon = normalize_text(vendor)
            step1_company_canon = normalize_text(company)
            merged = default_normalized()
            merged["employee_name"] = employee_name
            merged["vendor"] = vendor
            merged["company"] = company
            merged["duration"] = duration
            merged["period_start"] = ""
            merged["period_end"] = ""
            merged_hours: Dict[str, float] = {}
            observed_hours_all_files: Dict[str, float] = {}
            extracted_dates_all_files: set[str] = set()
            excluded_day_entries: List[Dict[str, Any]] = []
            overlap_conflicts: List[Dict[str, Any]] = []
            missing_critical_files: List[str] = []
            approval_missing_files: List[str] = []
            merged_ocr_parts: List[str] = []
            precheck_issues: List[str] = []
            quality_issues: List[str] = []
            conf_vals: List[float] = []
            approver_candidates: List[str] = []
            any_approval_signal = False
            table_columns_accum: List[str] = []
            headers_accum: List[str] = []
            file_fp_counts = Counter(x.get("fingerprint", "") for x in selected_files)
            consumed_fp: Dict[str, int] = {}
            merged_date_owner: Dict[str, str] = {}
            file_added_dates: Dict[str, set[str]] = {}
            file_included_hours_map: Dict[str, float] = {}
            file_notes_map: Dict[str, List[str]] = {}
            row_idx_by_key: Dict[str, int] = {}
            force_exclude_keys: set[str] = set()

            for idx, file_obj in enumerate(selected_files, start=1):
                file_key = f"{idx}::{file_obj.get('name', '')}"
                extraction_i = extract_and_normalize(file_obj["bytes"], file_obj["name"], hints=hints)
                normalized_i = extraction_i.get("normalized", {}) if isinstance(extraction_i, dict) else {}
                file_ext = os.path.splitext(str(file_obj.get("name", "")).lower())[1]
                explicit_dates = (
                    extract_explicit_dates_from_text(extraction_i.get("ocr_text", ""))
                    if file_ext in {".txt", ".doc", ".docx"}
                    else set()
                )
                precheck_issues.extend(extraction_i.get("meta", {}).get("prevalidation", {}).get("issues", []))
                quality_issues.extend(extraction_i.get("meta", {}).get("quality_validation", {}).get("issues", []))
                merged_ocr_parts.append((extraction_i.get("ocr_text") or "").strip())
                conf_vals.append(safe_float(normalized_i.get("confidence", 0.0), 0.0))

                file_notes: List[str] = file_notes_map.setdefault(file_key, [])
                file_added_dates.setdefault(file_key, set())
                file_included_hours_map.setdefault(file_key, 0.0)
                critical_missing = []
                for crit_key in ["employee_name", "vendor", "company"]:
                    crit_val = normalize_text(str(normalized_i.get(crit_key, "") or ""))
                    if not crit_val:
                        critical_missing.append(crit_key)
                if critical_missing:
                    missing_critical_files.append(file_obj["name"])
                    file_notes.append(f"Missing fields: {', '.join(critical_missing)}")

                file_fp = str(file_obj.get("fingerprint", "") or "")
                duplicate_upload = False
                if file_fp:
                    consumed_fp[file_fp] = consumed_fp.get(file_fp, 0) + 1
                    if file_fp_counts.get(file_fp, 0) > 1 and consumed_fp[file_fp] > 1:
                        duplicate_upload = True
                        file_notes.append("Duplicate file upload")

                mismatched_with_step1: List[str] = []
                extracted_emp = str(normalized_i.get("employee_name", "") or "")
                extracted_vendor = str(normalized_i.get("vendor", "") or "")
                extracted_company = str(normalized_i.get("company", "") or "")
                if step1_emp_canon and canonical_person_name(extracted_emp) and canonical_person_name(extracted_emp) != step1_emp_canon:
                    mismatched_with_step1.append(
                        f"employee_name (Step1='{format_person_name_display(employee_name)}', File='{format_person_name_display(extracted_emp)}')"
                    )
                if step1_vendor_canon and normalize_text(extracted_vendor) and normalize_text(extracted_vendor) != step1_vendor_canon:
                    mismatched_with_step1.append(
                        f"vendor (Step1='{vendor}', File='{extracted_vendor}')"
                    )
                if step1_company_canon and normalize_text(extracted_company) and normalize_text(extracted_company) != step1_company_canon:
                    mismatched_with_step1.append(
                        f"company (Step1='{company}', File='{extracted_company}')"
                    )
                if mismatched_with_step1:
                    file_notes.extend(mismatched_with_step1)

                approved_text = normalize_text(str(normalized_i.get("approved", "") or ""))
                approver_name_text = str(normalized_i.get("approver_name", "") or "").strip()
                has_named_approver = bool(approver_name_text) and is_probable_person_name(approver_name_text)
                generic_approval_tokens = {
                    "",
                    "approved",
                    "approved by",
                    "signature",
                    "signed",
                    "yes",
                    "yes/signature",
                    "no",
                    "none",
                    "na",
                    "n/a",
                    "rejected",
                    "declined",
                    "pending",
                }
                has_specific_approval_value = approved_text not in generic_approval_tokens
                approval_found = has_named_approver or has_specific_approval_value

                extracted_rows = normalized_i.get("day_hours", []) if isinstance(normalized_i.get("day_hours"), list) else []
                contributes_to_range = False
                included_hours_for_file = 0.0
                merged_dates_added_from_file: List[str] = []
                duplicate_data_upload = False
                for row in extracted_rows:
                    if not isinstance(row, dict):
                        continue
                    dt_text = str(row.get("date", "") or "")
                    dt_obj = parse_iso_date_optional(dt_text)
                    if not dt_obj:
                        continue
                    if explicit_dates and dt_text not in explicit_dates:
                        continue
                    extracted_dates_all_files.add(dt_text)
                    if step1_start and step1_end and (dt_obj < step1_start or dt_obj > step1_end):
                        continue
                    contributes_to_range = True
                    hour_val = safe_float(row.get("hours", 0.0), 0.0)
                    if dt_text not in observed_hours_all_files:
                        observed_hours_all_files[dt_text] = hour_val
                    if (not duplicate_upload) and (not mismatched_with_step1) and (not critical_missing) and approval_found:
                        if dt_text in merged_hours:
                            if abs(merged_hours[dt_text] - hour_val) > APP_CONFIG["hour_tolerance"]:
                                owner_key = merged_date_owner.get(dt_text, "")
                                overlap_conflicts.append(
                                    {
                                        "date": dt_text,
                                        "hours_existing": merged_hours[dt_text],
                                        "hours_new": hour_val,
                                        "file": file_obj["name"],
                                    }
                                )
                                conflict_note = f"Overlap conflict on {dt_text}"
                                file_notes.append(conflict_note)
                                if owner_key:
                                    owner_notes = file_notes_map.setdefault(owner_key, [])
                                    if conflict_note not in owner_notes:
                                        owner_notes.append(conflict_note)
                                    force_exclude_keys.add(owner_key)
                                    for odt in file_added_dates.get(owner_key, set()):
                                        merged_hours.pop(odt, None)
                                    file_included_hours_map[owner_key] = 0.0
                                    if owner_key in row_idx_by_key:
                                        owner_row = per_file_status[row_idx_by_key[owner_key]]
                                        owner_row["Included in Merged Total"] = "No"
                                        owner_row["Included Hours"] = 0.0
                                        owner_row["Status"] = "Needs Review"
                                        owner_row["Details"] = "; ".join(file_notes_map.get(owner_key, [])) or "Needs Review"
                                force_exclude_keys.add(file_key)
                            else:
                                # Same date with same hours already provided by another file:
                                # treat current file as duplicate/redundant and exclude it.
                                duplicate_data_upload = True
                                duplicate_note = f"Duplicate data for {dt_text}"
                                if duplicate_note not in file_notes:
                                    file_notes.append(duplicate_note)
                                force_exclude_keys.add(file_key)
                        else:
                            merged_hours[dt_text] = hour_val
                            included_hours_for_file += hour_val
                            file_included_hours_map[file_key] = file_included_hours_map.get(file_key, 0.0) + hour_val
                            merged_dates_added_from_file.append(dt_text)
                            file_added_dates[file_key].add(dt_text)
                            merged_date_owner[dt_text] = file_key
                    else:
                        excluded_day_entries.append(
                            {
                                "date": dt_text,
                                "hours": round(hour_val, 2),
                                "file": file_obj["name"],
                                "reason": (
                                    "Excluded due missing required fields"
                                    if critical_missing
                                    else (
                                        "Excluded due approval missing"
                                        if not approval_found
                                        else "Excluded due Step 1 field mismatch"
                                    )
                                ),
                            }
                        )

                if contributes_to_range and (not mismatched_with_step1) and not approval_found:
                    approval_missing_files.append(file_obj["name"])
                    file_notes.append("Approval signal missing for contributing file")

                extracted_ps = str(normalized_i.get("period_start", "") or "")
                extracted_pe = str(normalized_i.get("period_end", "") or "")
                if explicit_dates:
                    parsed_explicit = [parse_iso_date_optional(x) for x in explicit_dates]
                    parsed_explicit = [d for d in parsed_explicit if d is not None]
                    if parsed_explicit:
                        extracted_ps = min(parsed_explicit).isoformat()
                        extracted_pe = max(parsed_explicit).isoformat()
                        extracted_dates_all_files.update(d.isoformat() for d in parsed_explicit)
                else:
                    ps_any = parse_iso_date_optional(extracted_ps)
                    pe_any = parse_iso_date_optional(extracted_pe)
                    if ps_any:
                        extracted_dates_all_files.add(ps_any.isoformat())
                    if pe_any:
                        extracted_dates_all_files.add(pe_any.isoformat())
                if (not contributes_to_range) and step1_start and step1_end:
                    ps_obj = parse_iso_date_optional(extracted_ps)
                    pe_obj = parse_iso_date_optional(extracted_pe)
                    if ps_obj and pe_obj and ps_obj <= step1_end and pe_obj >= step1_start:
                        contributes_to_range = True
                        if (not mismatched_with_step1) and not approval_found:
                            approval_missing_files.append(file_obj["name"])
                            file_notes.append("Approval signal missing for contributing file")
                if not contributes_to_range:
                    file_notes.append("Outside selected Step 1 date range")
                excluded_from_merge = (
                    file_key in force_exclude_keys
                    or
                    duplicate_upload
                    or duplicate_data_upload
                    or bool(critical_missing)
                    or bool(mismatched_with_step1)
                    or (not contributes_to_range)
                    or (contributes_to_range and not approval_found)
                    or any("Overlap conflict" in n for n in file_notes)
                )
                if (not excluded_from_merge) and approval_found:
                    any_approval_signal = True
                    if approver_name_text and is_probable_person_name(approver_name_text):
                        approver_candidates.append(approver_name_text)
                if excluded_from_merge:
                    for dt_added in merged_dates_added_from_file:
                        merged_hours.pop(dt_added, None)
                    included_hours_for_file = 0.0
                    file_included_hours_map[file_key] = 0.0
                per_file_included_hours_total += included_hours_for_file
                if isinstance(normalized_i.get("headers"), list):
                    headers_accum.extend(str(x) for x in normalized_i.get("headers") if x is not None)
                if isinstance(normalized_i.get("table_columns"), list):
                    table_columns_accum.extend(str(x) for x in normalized_i.get("table_columns") if x is not None)

                per_file_status.append(
                    {
                        "__file_key": file_key,
                        "File Name": file_obj["name"],
                        "Extracted Period": f"{extracted_ps or '-'} to {extracted_pe or '-'}",
                        "Included in Merged Total": "No" if excluded_from_merge else "Yes",
                        "Included Hours": round(file_included_hours_map.get(file_key, included_hours_for_file), 2),
                        "Status": "Needs Review" if file_notes else "OK",
                        "Details": "; ".join(file_notes) if file_notes else "No issues",
                    }
                )
                row_idx_by_key[file_key] = len(per_file_status) - 1

            merged["headers"] = headers_accum
            merged["table_columns"] = table_columns_accum
            included_dates = sorted(parse_iso_date_optional(dt) for dt in merged_hours.keys())
            included_dates = [d for d in included_dates if d is not None]
            parsed_all_dates = sorted(d for d in (parse_iso_date_optional(x) for x in extracted_dates_all_files) if d is not None)
            if included_dates:
                # For comparison, period should reflect the included merged set.
                merged["period_start"] = included_dates[0].isoformat()
                merged["period_end"] = included_dates[-1].isoformat()
            elif parsed_all_dates:
                # If nothing was included, fall back to all observed extracted dates.
                merged["period_start"] = parsed_all_dates[0].isoformat()
                merged["period_end"] = parsed_all_dates[-1].isoformat()
            else:
                merged["period_start"] = step1_ps_text
                merged["period_end"] = step1_pe_text
            merged["day_hours"] = [{"date": dt, "hours": round(hr, 2)} for dt, hr in sorted(merged_hours.items())]
            per_file_status.sort(
                key=lambda r: (
                    parse_iso_date_optional(
                        str((r.get("Extracted Period", "") or "").split(" to ")[0]).strip()
                    )
                    or date.max
                )
            )
            merged["observed_day_hours_all_files"] = [
                {"date": dt, "hours": round(hr, 2)} for dt, hr in sorted(observed_hours_all_files.items())
            ]
            merged["excluded_day_hours"] = excluded_day_entries
            merged["total_hours"] = round(sum(merged_hours.values()), 2) if merged_hours else 0.0
            merged["confidence"] = round((sum(conf_vals) / len(conf_vals)), 3) if conf_vals else 0.0
            merged["approver_name"] = approver_candidates[0] if approver_candidates else ""
            merged["approved"] = "yes/signature" if any_approval_signal else ""
            extraction = {
                "normalized": merged,
                "raw": {"multi_file": True, "file_count": len(selected_files)},
                "meta": {
                    "mode": "multi_file_manual",
                    "file_count": len(selected_files),
                    "prevalidation": {"issues": precheck_issues},
                    "quality_validation": {"issues": quality_issues},
                    "per_file_status": per_file_status,
                },
                "ocr_text": "\n\n".join(x for x in merged_ocr_parts if x),
            }
            if missing_critical_files:
                forced_manual_reasons.append(
                    f"Missing critical fields in file(s): {', '.join(sorted(set(missing_critical_files)))}"
                )
            if overlap_conflicts:
                forced_manual_reasons.append("Overlap conflict found: same date has different hours in multiple files")
            if approval_missing_files:
                forced_manual_reasons.append(
                    f"Approval missing in contributing file(s): {', '.join(sorted(set(approval_missing_files)))}"
                )
        else:
            file_obj = selected_files[0]
            preview_norm = (
                st.session_state.get("autofill_extraction_preview", {}).get("normalized", {})
                if isinstance(st.session_state.get("autofill_extraction_preview"), dict)
                else {}
            )
            preview_ps_text = str(preview_norm.get("period_start", "") or "")
            preview_pe_text = str(preview_norm.get("period_end", "") or "")
            reuse_autofill_extraction = (
                st.session_state.entry_mode == "Auto-Fill from Uploaded Timesheet"
                and st.session_state.get("autofill_extraction_preview") is not None
                and st.session_state.get("autofill_last_fingerprint") == fingerprint
                and step1_ps_text == preview_ps_text
                and step1_pe_text == preview_pe_text
            )
            if reuse_autofill_extraction:
                progress.progress(30, text="Using extracted data from Step 1 upload...")
                extraction = st.session_state.autofill_extraction_preview
            else:
                progress.progress(30, text="Running Textract...")
                extraction = extract_and_normalize(file_obj["bytes"], file_obj["name"], hints=hints)
            precheck_issues = extraction.get("meta", {}).get("prevalidation", {}).get("issues", [])
            quality_issues = extraction.get("meta", {}).get("quality_validation", {}).get("issues", [])
            if st.session_state.entry_mode == "Manual Entry":
                normalized_single = extraction.get("normalized", {}) if isinstance(extraction, dict) else {}
                step1_start = parse_iso_date_optional(step1_ps_text)
                step1_end = parse_iso_date_optional(step1_pe_text)
                file_notes: List[str] = []
                critical_missing = []
                for crit_key in ["employee_name", "vendor", "company"]:
                    if not normalize_text(str(normalized_single.get(crit_key, "") or "")):
                        critical_missing.append(crit_key)
                if critical_missing:
                    file_notes.append(f"Missing fields: {', '.join(critical_missing)}")
                mismatched_with_step1: List[str] = []
                if canonical_person_name(str(normalized_single.get("employee_name", "") or "")) != canonical_person_name(employee_name):
                    mismatched_with_step1.append(
                        f"employee_name (Step1='{format_person_name_display(employee_name)}', File='{format_person_name_display(str(normalized_single.get('employee_name', '') or ''))}')"
                    )
                if normalize_text(str(normalized_single.get("vendor", "") or "")) != normalize_text(vendor):
                    mismatched_with_step1.append(
                        f"vendor (Step1='{vendor}', File='{str(normalized_single.get('vendor', '') or '')}')"
                    )
                if normalize_text(str(normalized_single.get("company", "") or "")) != normalize_text(company):
                    mismatched_with_step1.append(
                        f"company (Step1='{company}', File='{str(normalized_single.get('company', '') or '')}')"
                    )
                if mismatched_with_step1:
                    file_notes.extend(mismatched_with_step1)

                approved_text = normalize_text(str(normalized_single.get("approved", "") or ""))
                approver_name_text = str(normalized_single.get("approver_name", "") or "").strip()
                has_named_approver = bool(approver_name_text) and is_probable_person_name(approver_name_text)
                generic_approval_tokens = {"", "approved", "approved by", "signature", "signed", "yes", "yes/signature", "no", "none", "na", "n/a", "rejected", "declined", "pending"}
                approval_found = has_named_approver or (approved_text not in generic_approval_tokens)

                extracted_rows = normalized_single.get("day_hours", []) if isinstance(normalized_single.get("day_hours"), list) else []
                contributes_to_range = False
                included_hours_for_file = 0.0
                for row in extracted_rows:
                    if not isinstance(row, dict):
                        continue
                    dt_obj = parse_iso_date_optional(str(row.get("date", "") or ""))
                    if not dt_obj:
                        continue
                    if step1_start and step1_end and (dt_obj < step1_start or dt_obj > step1_end):
                        continue
                    contributes_to_range = True
                    included_hours_for_file += safe_float(row.get("hours", 0.0), 0.0)
                if not contributes_to_range:
                    file_notes.append("Outside selected Step 1 date range")
                if contributes_to_range and not approval_found:
                    file_notes.append("Approval signal missing for contributing file")
                excluded_from_merge = (
                    bool(critical_missing)
                    or bool(mismatched_with_step1)
                    or (not contributes_to_range)
                    or (contributes_to_range and not approval_found)
                )
                if excluded_from_merge:
                    included_hours_for_file = 0.0
                extracted_ps = str(normalized_single.get("period_start", "") or "")
                extracted_pe = str(normalized_single.get("period_end", "") or "")
                per_file_status = [
                    {
                        "File Name": file_obj["name"],
                        "Extracted Period": f"{extracted_ps or '-'} to {extracted_pe or '-'}",
                        "Included in Merged Total": "No" if excluded_from_merge else "Yes",
                        "Included Hours": round(included_hours_for_file, 2),
                        "Status": "Needs Review" if file_notes else "OK",
                        "Details": "; ".join(file_notes) if file_notes else "No issues",
                    }
                ]
                per_file_included_hours_total = round(included_hours_for_file, 2)

        progress.progress(70, text="Comparing with Step 1...")
        extracted = extraction["normalized"]
        comparison = compare_data(step1, extracted)
        p_key = pattern_key(duration, day_hours, employee_name, vendor, company)
        t_hash = template_hash(extracted, p_key)
        streak = get_streak(employee_name, vendor, company, t_hash, p_key)
        confidence = safe_float(extracted.get("confidence", 0.0), 0.0)
        decision, approval_type, reasons = decide(comparison, confidence, streak)
        all_matched = not comparison.get("mismatches")
        if precheck_issues:
            decision = "MANUAL_REVIEW"
            approval_type = "MANUAL_REVIEW"
            reasons = [f"Pre-validation: {x}" for x in precheck_issues]
        elif quality_issues and not all_matched:
            # Route to manual only when quality issue likely impacted extraction/comparison.
            decision = "MANUAL_REVIEW"
            approval_type = "MANUAL_REVIEW"
            reasons = reasons + [f"Quality issue: {x}" for x in quality_issues]
        if forced_manual_reasons:
            decision = "MANUAL_REVIEW"
            approval_type = "MANUAL_REVIEW"
            reasons = reasons + forced_manual_reasons
        mismatch_signature = build_mismatch_signature(step1, extracted, comparison)
        learned_answer, learned_streak = get_learning_state(
            employee_id,
            vendor,
            company,
            t_hash,
            p_key,
            mismatch_signature,
        )
        if (
            decision == "MANUAL_REVIEW"
            and learned_answer == "yes"
            and learned_streak >= APP_CONFIG["trusted_streak_threshold"]
            and not comparison.get("mismatches", {}).get("day_hours")
        ):
            decision = "AUTO_APPROVE_TRUSTED"
            approval_type = "AUTO_APPROVE_TRUSTED_TEMPLATE"
            reasons = [f"Learned pattern auto-approve at threshold {APP_CONFIG['trusted_streak_threshold']}"]
        # Trusted pattern override (single/multi): after enough manual approvals
        # of same pattern, allow auto-approve for recurring non-critical/manual-only reasons.
        # Keep a hard stop for explicit negative approval statuses.
        trusted_trigger = max(APP_CONFIG["trusted_streak_threshold"] - 1, 0)
        approved_norm = normalize_text(str(extracted.get("approved", "") or ""))
        explicit_negative_approval = approved_norm in {"rejected", "declined", "pending"}
        has_daywise_mismatch = bool(comparison.get("mismatches", {}).get("day_hours"))
        if (
            streak >= trusted_trigger
            and not explicit_negative_approval
            and not has_daywise_mismatch
            and not comparison.get("mismatches")
        ):
            decision = "AUTO_APPROVE_TRUSTED"
            approval_type = "AUTO_APPROVE_TRUSTED_TEMPLATE"
            reasons = [f"Trusted auto-approve at threshold {APP_CONFIG['trusted_streak_threshold']}"]
        # Business rule: if everything matches, always auto approve.
        if all_matched and not forced_manual_reasons and not precheck_issues and not quality_issues:
            decision = "AUTO_APPROVE"
            approval_type = "AUTO_APPROVE"
            reasons = []
        display_streak = streak
        if decision == "AUTO_APPROVE_TRUSTED":
            # User-facing streak should reflect threshold hit on this run.
            display_streak = max(streak + 1, APP_CONFIG["trusted_streak_threshold"])
        reason_codes = build_reason_codes(decision, comparison, confidence, extraction.get("meta", {}))
        progress.progress(100, text="Completed")
        st.session_state.validation_result = {
            "step1": step1,
            "extraction": extraction,
            "extracted": extracted,
            "comparison": comparison,
            "pattern_key": p_key,
            "template_hash": t_hash,
            "streak": display_streak,
            "decision": decision,
            "approval_type": approval_type,
            "reasons": reasons,
            "reason_codes": reason_codes,
            "submission_hash": submission_hash,
            "processed_fingerprint": fingerprint,
            "per_file_status": per_file_status,
            "per_file_included_hours_total": round(per_file_included_hours_total, 2),
            "mismatch_signature": mismatch_signature,
            "learned_answer": learned_answer,
            "learned_streak": learned_streak,
        }
        st.session_state.last_upload_fingerprint = fingerprint
        st.session_state.last_submission_hash = submission_hash
        if decision != "MANUAL_REVIEW":
            log_history(
                employee_id,
                employee_name,
                display_streak,
                t_hash,
                p_key,
                submission_hash,
                step1,
                extracted,
                comparison,
                decision,
                approval_type,
                reasons,
                reason_codes,
                extraction["meta"],
            )

    result = st.session_state.validation_result
    if result:
        step1 = result["step1"]
        extraction = result["extraction"]
        extracted = result["extracted"]
        comparison = result["comparison"]
        decision = result["decision"]
        approval_type = result["approval_type"]
        reasons = result["reasons"]
        reason_codes = result.get("reason_codes", [])
        mismatch_signature = result.get("mismatch_signature", "")
        learned_streak = safe_float(result.get("learned_streak", 0), 0.0)
        streak = result["streak"]
        p_key = result["pattern_key"]
        t_hash = result["template_hash"]
        submission_hash = result.get("submission_hash", "")

        with st.expander("Extraction Status", expanded=False):
            # Show the normalized extraction payload as JSON for debugging/validation.
            st.json(extracted)
            # Optional: show a short OCR text preview to help diagnose period/date parsing.
            ocr_preview = (extraction.get("ocr_text") or "").strip()
            if ocr_preview:
                st.caption("OCR text preview (first 1200 chars)")
                st.code(ocr_preview[:1200])

        with st.expander("Pipeline Status", expanded=False):
            st.write(extraction["meta"])
            if extracted.get("confidence_breakdown"):
                st.write({"confidence_breakdown": extracted.get("confidence_breakdown")})
            if extracted.get("error"):
                st.error(user_friendly_error(extracted["error"]))

        per_file_status = result.get("per_file_status", []) if isinstance(result, dict) else []
        if (
            (not per_file_status)
            and isinstance(extraction, dict)
            and isinstance(extraction.get("meta", {}), dict)
            and isinstance(extraction.get("meta", {}).get("per_file_status"), list)
        ):
            per_file_status = extraction.get("meta", {}).get("per_file_status", [])
        if st.session_state.entry_mode == "Manual Entry":
            st.subheader("Per-file Processing Status")
            st.caption(
                "Green row = included in merged total, Red row = excluded. Included Hours are the hours counted from that file."
            )
            if per_file_status:
                per_file_df = pd.DataFrame(per_file_status)
                if "__file_key" in per_file_df.columns:
                    per_file_df = per_file_df.drop(columns=["__file_key"])
                def _highlight_row(row: pd.Series) -> List[str]:
                    bg = "#e8f5e9" if str(row.get("Included in Merged Total", "")) == "Yes" else "#ffebee"
                    return [f"background-color: {bg}"] * len(row)
                st.dataframe(
                    per_file_df.style.apply(_highlight_row, axis=1).format({"Included Hours": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(f"Total Included Hours: {safe_float(result.get('per_file_included_hours_total', 0.0), 0.0):.2f}")
            else:
                st.info("No per-file rows available for this run.")

        st.subheader("Comparison")
        fields = ["employee_name", "vendor", "company", "period_start", "period_end", "approved", "total_hours"]
        total_step1 = sum(x["hours"] for x in step1["day_hours"])
        def _fmt_hours(v: Any) -> Any:
            n = safe_float(v, None)
            return f"{n:.1f}" if n is not None else v
        html = "<table><tr><th>Field</th><th>Step1</th><th>Extracted</th><th>Status</th></tr>"
        for field in fields:
            matched = bool(comparison["matches"].get(field, False))
            if field == "total_hours":
                left = _fmt_hours(total_step1)
            elif field == "approved":
                left = "yes/signature"
            elif field == "employee_name":
                left = format_person_name_display(step1.get(field, "")) or step1.get(field, "-")
            else:
                left = step1.get(field, "-")
            if field in ["period_start", "period_end"] and isinstance(comparison.get("derived_extracted_period"), dict):
                right = comparison["derived_extracted_period"].get(field, extracted.get(field, "-"))
            elif field == "total_hours":
                right = _fmt_hours(comparison.get("actual_total_for_compare", extracted.get("total_hours", "-")))
            elif field == "approved":
                approver_name = (extracted.get("approver_name", "") or "").strip()
                if approver_name and is_probable_person_name(approver_name):
                    right = approver_name
                elif comparison["matches"].get("approved", False):
                    right = "yes/signature"
                else:
                    right = extracted.get("approved", "-")
            elif field == "employee_name":
                right = format_person_name_display(extracted.get(field, "")) or extracted.get(field, "-")
            else:
                right = extracted.get(field, "-")
            if field in ["period_start", "period_end"]:
                step_dt = parse_iso_date_optional(str(step1.get(field, "")))
                extracted_dt = parse_iso_date_optional(str(extracted.get(field, "")))
                step1_ps = parse_iso_date_optional(str(step1.get("period_start", "")))
                step1_pe = parse_iso_date_optional(str(step1.get("period_end", "")))
                is_in_range_non_exact_match = (
                    matched
                    and step_dt is not None
                    and extracted_dt is not None
                    and step_dt != extracted_dt
                    and step1_ps is not None
                    and step1_pe is not None
                    and step1_ps <= extracted_dt <= step1_pe
                )
                if comparison.get("period_coverage_note") or is_in_range_non_exact_match:
                    right = f"{right} (within selected range)"
            html += row_html(field, left, right, matched)
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)
        if comparison.get("period_coverage_note"):
            st.caption(str(comparison.get("period_coverage_note")))

        if comparison.get("day_hours_missing_in_extracted"):
            st.info("Day-wise hours not found in uploaded sheet. Comparison is based on period and total hours.")
        elif comparison["mismatches"].get("day_hours"):
            st.subheader("Day-wise Mismatches")
            st.dataframe(comparison["mismatches"]["day_hours"], use_container_width=True)
        else:
            st.success("All day-wise hours matched.")

        if comparison.get("excluded_day_hours"):
            st.caption("These uploaded dates were excluded from merged total (for example, Step 1 field mismatch in that file).")
            st.dataframe(comparison.get("excluded_day_hours", []), use_container_width=True)

        with st.expander("Decision", expanded=False):
            st.write(
                {
                    "decision": decision,
                    "approval_type": approval_type,
                    "reason_codes": reason_codes,
                    "reasons": reasons,
                    "current_streak": streak,
                }
            )

        if decision in ["AUTO_APPROVE", "AUTO_APPROVE_TRUSTED"]:
            st.success("Auto approved. Click OK to process the next file.")
            if st.button("OK"):
                prepare_next_document()
                st.rerun()

        if decision == "MANUAL_REVIEW":
            st.subheader("Manual Review Action")
            has_daywise_mismatch = bool(comparison.get("mismatches", {}).get("day_hours"))
            learning_question_short = "manual_reject"
            if has_daywise_mismatch:
                st.warning("Date-wise hours mismatch detected. Only REJECT is allowed.")
                manual_outcome = st.selectbox("Outcome", ["REJECT"], key="manual_outcome")
                learning_answer = "no"
                learning_question_short = "day_hours_mismatch"
            else:
                manual_outcome = st.selectbox("Outcome", ["APPROVE", "REJECT"], key="manual_outcome")
                learning_answer = "yes"
                if manual_outcome == "APPROVE":
                    mismatch_keys = list((comparison.get("mismatches", {}) or {}).keys())
                    question_map = {
                        "employee_name": "Name does not match. Is this the same person?",
                        "vendor": "Vendor does not match. Is this acceptable?",
                        "company": "Company value is missing/mismatched. Is this acceptable for this employee/case?",
                        "period_start": "Period start differs. Is this acceptable for this case?",
                        "period_end": "Period end differs. Is this acceptable for this case?",
                        "approved": "Approval signal is missing/mismatched. Is this acceptable?",
                        "total_hours": "Total hours mismatch is present. Is this acceptable?",
                    }
                    per_mismatch_answers: List[str] = []
                    asked_keys: List[str] = []
                    for mk in mismatch_keys:
                        q = question_map.get(mk, f"Mismatch in '{mk}'. Is this acceptable?")
                        ans = st.radio(
                            q,
                            ["yes", "no"],
                            horizontal=True,
                            key=f"manual_learning_answer_{mk}",
                        )
                        asked_keys.append(mk)
                        per_mismatch_answers.append(ans)
                    if per_mismatch_answers:
                        learning_answer = "yes" if all(x == "yes" for x in per_mismatch_answers) else "no"
                        learning_question_short = f"mismatch:{','.join(sorted(set(asked_keys)))}"
                    else:
                        learning_answer = st.radio(
                            "Is this mismatch pattern acceptable for this employee/case? (learned Yes/No)",
                            ["yes", "no"],
                            horizontal=True,
                            key="manual_learning_answer",
                        )
                        learning_question_short = "general_mismatch"
                    st.caption(f"Current learned streak for this pattern: {int(learned_streak)}")
            manual_comment = st.text_area("Comment", key="manual_comment")
            if st.button("Submit Manual Decision"):
                if manual_outcome == "APPROVE":
                    new_streak = streak + 1
                    set_streak(employee_name, vendor, company, t_hash, p_key, new_streak)
                    if mismatch_signature:
                        record_learning_answer(
                            employee_id,
                            vendor,
                            company,
                            t_hash,
                            p_key,
                            mismatch_signature,
                            learning_answer,
                            learning_question_short,
                        )
                    st.success(f"Manual approved; streak is now {new_streak}")
                else:
                    new_streak = 0
                    set_streak(employee_name, vendor, company, t_hash, p_key, new_streak)
                    if mismatch_signature:
                        record_learning_answer(
                            employee_id,
                            vendor,
                            company,
                            t_hash,
                            p_key,
                            mismatch_signature,
                            "no",
                            learning_question_short,
                        )
                    st.warning("Streak reset to 0")
                log_history(
                    employee_id,
                    employee_name,
                    new_streak,
                    t_hash,
                    p_key,
                    submission_hash,
                    step1,
                    extracted,
                    comparison,
                    f"MANUAL_{manual_outcome}",
                    "MANUAL_REVIEW",
                    [f"comment: {manual_comment}"] if manual_comment else [],
                    ["MANUAL_APPROVE"] if manual_outcome == "APPROVE" else ["MANUAL_REJECT"],
                    extraction["meta"],
                )
                prepare_next_document()
                st.rerun()

    st.subheader("Validation History")
    if st.button("Clear History"):
        clear_all_history()
        st.success("Validation and streak history cleared.")
        st.rerun()
    st.dataframe(recent_history(25), use_container_width=True)
    st.subheader("Learning Pattern Memory")
    st.dataframe(recent_learning_memory(50), use_container_width=True)

    st.caption("No file is saved to uploads folder. Extraction runs directly from in-memory upload bytes.")


if __name__ == "__main__":
    main()
