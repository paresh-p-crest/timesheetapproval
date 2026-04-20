import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import boto3
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from PIL import Image, UnidentifiedImageError
from pypdf import PdfReader

load_dotenv(".env")

DB_PATH = "streamlit_poc.db"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
APP_CONFIG = {
    "hour_tolerance": 0.01,
    "confidence_threshold": 0.8,
    "trusted_streak_threshold": 3,
    "critical_fields": ["employee_name", "vendor", "company"],
}
# Bump this when extraction/autofill mapping logic changes so cached autofill is recalculated.
AUTOFILL_LOGIC_VERSION = "2026-04-17-v3"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
    existing_cols = {row[1] for row in conn.execute("pragma table_info(validation_history)").fetchall()}
    if "employee_name" not in existing_cols:
        conn.execute("alter table validation_history add column employee_name text")
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
        return "Uploaded file is not a supported timesheet document. Please upload a valid PDF/JPG/PNG timesheet."
    if "password-protected pdf" in lowered:
        return "This PDF is password protected. Please upload an unlocked timesheet file."
    if "corrupted" in lowered or "unreadable" in lowered:
        return "The uploaded file looks unreadable or corrupted. Please upload a clear timesheet file."
    if "textract_failed" in lowered:
        return "Could not read text from this file. Please upload a clear timesheet PDF/image."
    if "bedrock_normalization_failed" in lowered:
        return "The file was read, but timesheet fields could not be interpreted. Please upload a clearer timesheet."
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
        # Keep only human-like names (at least 2 tokens, unless comma format).
        token_count = len([t for t in re.split(r"[,\s]+", v) if t.strip()])
        if token_count < 2:
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
    # Column-style fallback: pick first strong "Last, First" candidate from OCR lines.
    for ln in lines:
        m = re.search(r"\b([A-Za-z]{2,}\s*,\s*[A-Za-z][A-Za-z .'-]{1,40})\b", ln)
        if m:
            nm = clean_name(m.group(1))
            if nm:
                return nm
    return ""


def textract_extract(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    client = boto3.client("textract", region_name=os.getenv("AWS_REGION", "us-east-1"))
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


def prevalidate_file(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    ext = os.path.splitext((filename or "").lower())[1]
    issues: List[str] = []
    details: Dict[str, Any] = {"supported_format": ext in SUPPORTED_EXTENSIONS}
    if ext not in SUPPORTED_EXTENSIONS:
        issues.append(f"Unsupported format: {ext or 'unknown'}")
        return {"failed": True, "issues": issues, "details": details}

    if ext == ".pdf":
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


def normalize_with_bedrock(text: str, hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    model_id = os.getenv("BEDROCK_MODEL_ID")
    if not model_id:
        raise ValueError("BEDROCK_MODEL_ID missing. Set it in .env")
    llm = ChatBedrock(
        model_id=model_id,
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_kwargs={"temperature": 0},
    )
    hints = hints or {}
    hint_duration = hints.get("duration", "")
    hint_period_start = hints.get("period_start", "")
    hint_period_end = hints.get("period_end", "")
    prompt = f"""
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
    out = llm.invoke(prompt).content
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
    candidates: List[List[float]] = []

    for ln in lines:
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
        token_vals = [safe_float(m.group(1), None) for m in re.finditer(r"\b(\d{1,2}\.\d{2})\b", ocr_text or "")]
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

    # Ensure total_hours is aligned with day_hours for daily templates.
    # (Keep summary-style sheets untouched, since their rows can be weekly totals.)
    if isinstance(normalized.get("day_hours"), list) and normalized["day_hours"] and not _is_summary_like_day_hours(normalized):
        day_total = round(sum(safe_float(x.get("hours", 0.0), 0.0) for x in normalized["day_hours"]), 2)
        existing_total = safe_float(normalized.get("total_hours", None), None)
        if existing_total is None or abs(existing_total - day_total) > APP_CONFIG["hour_tolerance"]:
            normalized["total_hours"] = day_total

    return normalized


def extract_and_normalize(file_bytes: bytes, filename: str, hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    meta = {
        "aws_region": os.getenv("AWS_REGION", "us-east-1"),
        "model_id": os.getenv("BEDROCK_MODEL_ID", ""),
        "textract_used": False,
        "llm_used": False,
        "prevalidation": {},
        "quality_validation": {},
    }
    precheck = prevalidate_file(file_bytes, filename)
    meta["prevalidation"] = precheck
    if precheck.get("failed"):
        return {
            "normalized": default_normalized("Pre-validation failed"),
            "raw": {"error": "; ".join(precheck.get("issues", []))},
            "meta": meta,
        }

    quality = validate_image_quality(file_bytes, filename)
    meta["quality_validation"] = quality
    ext = os.path.splitext((filename or "").lower())[1]
    try:
        textract_out = textract_extract(file_bytes, filename)
        meta["textract_used"] = True
    except Exception as exc:
        err = str(exc)
        # Some PDFs are rejected by AnalyzeDocument depending on internal format.
        # Fallback to local PDF text extraction so Bedrock can still normalize fields.
        if ext == ".pdf" and ("UnsupportedDocumentException" in err or "InvalidParameterException" in err):
            try:
                fallback_text = extract_text_from_pdf_local(file_bytes)
                if fallback_text.strip():
                    textract_out = {"raw": {"fallback": "pypdf_text_extract"}, "text": fallback_text, "filename": filename}
                    meta["textract_used"] = False
                else:
                    return {"normalized": default_normalized(f"textract_failed: {exc}"), "raw": {"error": str(exc)}, "meta": meta}
            except Exception as pdf_exc:
                return {
                    "normalized": default_normalized(f"textract_failed: {exc}; pdf_fallback_failed: {pdf_exc}"),
                    "raw": {"error": str(exc)},
                    "meta": meta,
                }
        else:
            return {"normalized": default_normalized(f"textract_failed: {exc}"), "raw": {"error": str(exc)}, "meta": meta}

    bedrock_failed = False
    try:
        normalized = normalize_with_bedrock(textract_out["text"], hints=hints)
        meta["llm_used"] = True
    except Exception as exc:
        bedrock_failed = True
        normalized = default_normalized(f"bedrock_normalization_failed: {exc}")

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
    if not normalize_text(str(normalized.get("approved", ""))):
        inferred_approved = infer_approved_from_text(textract_out.get("text", ""))
        if inferred_approved:
            normalized["approved"] = inferred_approved
    # Extract approver readable name when present.
    if not normalize_text(str(normalized.get("approver_name", ""))):
        inferred_approver_name = infer_approver_name_from_text(textract_out.get("text", ""))
        if inferred_approver_name:
            normalized["approver_name"] = inferred_approver_name

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

    for field in ["vendor", "company"]:
        ok = normalize_text(step1.get(field, "")) == normalize_text(extracted.get(field, ""))
        result["matches"][field] = ok
        if not ok:
            result["mismatches"][field] = {"expected": step1.get(field), "actual": extracted.get(field)}
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
    for field in ["period_start", "period_end"]:
        ok = (step1.get(field) or "") == (extracted.get(field) or "")
        result["matches"][field] = ok
        if not ok:
            result["mismatches"][field] = {"expected": step1.get(field), "actual": extracted.get(field)}
    approved_text = normalize_text(extracted.get("approved", ""))
    approved_ok = bool(approved_text) and approved_text not in ["no", "none", "na", "n/a"]
    result["matches"]["approved"] = approved_ok
    if not approved_ok:
        result["mismatches"]["approved"] = {"expected": "yes/signature", "actual": extracted.get("approved", "")}
    expected_map = {d["date"]: float(d["hours"]) for d in step1.get("day_hours", [])}
    extracted_rows = extracted.get("day_hours", []) if isinstance(extracted.get("day_hours"), list) else []
    actual_map = {d.get("date"): safe_float(d.get("hours", 0), 0.0) for d in extracted_rows if d.get("date")}
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
            if actual is None or abs(expected - actual) > tolerance:
                hour_diffs.append({"date": dt, "expected": expected, "actual": actual})
                result["hour_mismatch_count"] += 1
        result["matches"]["day_hours"] = len(hour_diffs) == 0
        if hour_diffs:
            result["mismatches"]["day_hours"] = hour_diffs
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
    if actual_total is None and actual_map:
        actual_total = sum(actual_map.values())
    actual_total_num = safe_float(actual_total, None)
    result["actual_total_for_compare"] = actual_total_num
    ok = actual_total_num is not None and abs(expected_total - actual_total_num) <= tolerance
    result["matches"]["total_hours"] = ok
    if not ok:
        result["mismatches"]["total_hours"] = {"expected": expected_total, "actual": actual_total_num}
    return result


def pattern_key(duration: str, day_hours: List[Dict[str, Any]]) -> str:
    pairs = []
    for item in sorted(day_hours, key=lambda x: x["date"]):
        d = datetime.strptime(item["date"], "%Y-%m-%d").strftime("%a")
        pairs.append(f"{d}:{float(item['hours']):.2f}")
    return f"{duration}|{'|'.join(pairs)}"


def template_hash(extracted: Dict[str, Any]) -> str:
    raw = {
        "headers": extracted.get("headers", []),
        "columns": extracted.get("table_columns", []),
        "row_count": len(extracted.get("day_hours", [])),
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
    # Reset AUTOINCREMENT sequence so next history record starts at 1
    conn.execute("delete from sqlite_sequence where name='validation_history'")
    conn.commit()
    conn.close()


def decide(comparison: Dict[str, Any], confidence: float, streak: int) -> Tuple[str, str, List[str]]:
    reasons: List[str] = []

    # If explicit approval status indicates rejection/pending, never trusted-auto-approve.
    if comparison.get("mismatches", {}).get("approved"):
        return "MANUAL_REVIEW", "MANUAL_REVIEW", ["Approval status indicates not approved"]

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


def log_history(
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
        insert into validation_history(created_at,employee_name,streak_value,template_hash,pattern_key,submission_hash,step1_json,extracted_json,comparison_json,decision,approval_type,reasons_json,reason_codes_json,aws_meta_json)
        values(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.utcnow().isoformat(),
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
        select id, created_at, employee_name, streak_value, template_hash, pattern_key, decision, approval_type, reasons_json, reason_codes_json
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
            reasons = json.loads(r[8]) if r[8] else []
        except json.JSONDecodeError:
            reasons = [str(r[8])]
        out.append(
            {
                "record_id": r[0],
                "created_at": r[1],
                "employee_name": r[2] or "",
                "streak": r[3] if r[3] is not None else 0,
                "template_hash": (r[4] or "")[:12],
                "pattern_key": (r[5] or "")[:40],
                "decision": r[6],
                "reasons": "; ".join(reasons) if reasons else "",
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
    return len(rows) <= 10 and (has_week_end_header or has_weekly_total_values)


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
        start_date = st.session_state.get("period_start", date.today() - timedelta(days=6))
        end_date = st.session_state.get("period_end", date.today())

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


def reset_pipeline_state(clear_source: bool = False) -> None:
    st.session_state.validation_result = None
    st.session_state.last_upload_fingerprint = None
    if clear_source:
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
    st.set_page_config(page_title="Timesheet Approval POC", layout="wide")
    st.title("Timesheet Approval POC")
    current_threshold = get_setting_int("trusted_streak_threshold", APP_CONFIG["trusted_streak_threshold"])
    current_threshold = max(1, min(6, current_threshold))
    APP_CONFIG["trusted_streak_threshold"] = current_threshold

    page = st.sidebar.selectbox("Page", ["Approval", "Settings"])
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
        return

    if "validation_result" not in st.session_state:
        st.session_state.validation_result = None
    if "last_upload_fingerprint" not in st.session_state:
        st.session_state.last_upload_fingerprint = None
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
        st.session_state.entry_mode = "Manual Entry"
    if "enable_duplicate_check" not in st.session_state:
        st.session_state.enable_duplicate_check = False
    if "pending_step1_reset" not in st.session_state:
        st.session_state.pending_step1_reset = False
    if "employee_name" not in st.session_state:
        st.session_state.employee_name = ""
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

    if st.session_state.entry_mode == "Auto-Fill from Uploaded Timesheet":
        st.info("Upload a reference timesheet to auto-fill fields. You can edit values before final validation.")
        autofill_file = st.file_uploader(
            "Auto-Fill Source (PDF/Image)",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"autofill_{st.session_state.autofill_uploader_key}",
        )
        if autofill_file:
            autofill_bytes = autofill_file.getvalue()
            autofill_fingerprint = hashlib.sha256(autofill_bytes).hexdigest()
            autofill_signature = f"{autofill_fingerprint}:{AUTOFILL_LOGIC_VERSION}"
            st.session_state.source_file_bytes = autofill_bytes
            st.session_state.source_file_name = autofill_file.name
            st.session_state.source_file_fingerprint = autofill_fingerprint
            if st.session_state.autofill_last_signature != autofill_signature:
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
    if st.session_state.entry_mode == "Manual Entry":
        st.caption("Upload a timesheet file and process it.")
        manual_file = st.file_uploader(
            "Upload PDF/Image",
            type=["pdf", "png", "jpg", "jpeg"],
            key=f"uploader_{st.session_state.uploader_key}",
        )
        process_clicked = st.button("Process Next", type="primary")
        if process_clicked:
            if not manual_file:
                st.warning("Please upload a file first.")
                return
            file_bytes = manual_file.getvalue()
            file_name = manual_file.name
            fingerprint = hashlib.sha256(file_bytes).hexdigest()
        else:
            file_bytes = None
            file_name = ""
            fingerprint = None
    else:
        st.caption("Process the file already uploaded in Step 1.")
        process_clicked = st.button("Process Next", type="primary")
        if process_clicked:
            if st.session_state.source_file_bytes is None:
                st.warning("Please upload a file in Step 1 first.")
                return
            file_bytes = st.session_state.source_file_bytes
            file_name = st.session_state.source_file_name or "source_upload"
            fingerprint = st.session_state.source_file_fingerprint or hashlib.sha256(file_bytes).hexdigest()
        else:
            file_bytes = None
            file_name = ""
            fingerprint = None

    if process_clicked and file_bytes is not None and fingerprint is not None:

        if st.session_state.last_upload_fingerprint != fingerprint:
            progress = st.progress(5, text="Reading file...")
            step1 = {
                "employee_name": employee_name,
                "vendor": vendor,
                "company": company,
                "duration": duration,
                "period_start": period_start.isoformat() if isinstance(period_start, date) else "",
                "period_end": period_end.isoformat() if isinstance(period_end, date) else "",
                "day_hours": day_hours,
            }
            submission_hash = build_submission_hash(step1, fingerprint)
            if st.session_state.enable_duplicate_check and duplicate_submission_exists(submission_hash):
                st.warning("Duplicate submission detected for the same employee/period/file. Skipping re-processing.")
                return
            progress.progress(30, text="Running Textract...")
            extraction = extract_and_normalize(
                file_bytes,
                file_name,
                hints={
                    "duration": duration,
                    "period_start": period_start.isoformat() if isinstance(period_start, date) else "",
                    "period_end": period_end.isoformat() if isinstance(period_end, date) else "",
                },
            )
            progress.progress(70, text="Comparing with Step 1...")
            extracted = extraction["normalized"]
            comparison = compare_data(step1, extracted)
            p_key = pattern_key(duration, day_hours)
            t_hash = template_hash(extracted)
            streak = get_streak(employee_name, vendor, company, t_hash, p_key)
            confidence = safe_float(extracted.get("confidence", 0.0), 0.0)
            decision, approval_type, reasons = decide(comparison, confidence, streak)
            precheck_issues = extraction.get("meta", {}).get("prevalidation", {}).get("issues", [])
            quality_issues = extraction.get("meta", {}).get("quality_validation", {}).get("issues", [])
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
            # Business rule: if everything matches, always auto approve.
            if all_matched:
                decision = "AUTO_APPROVE"
                approval_type = "AUTO_APPROVE"
                reasons = []
            reason_codes = build_reason_codes(decision, comparison, confidence, extraction.get("meta", {}))
            progress.progress(100, text="Completed")
            st.session_state.validation_result = {
                "step1": step1,
                "extraction": extraction,
                "extracted": extracted,
                "comparison": comparison,
                "pattern_key": p_key,
                "template_hash": t_hash,
                "streak": streak,
                "decision": decision,
                "approval_type": approval_type,
                "reasons": reasons,
                "reason_codes": reason_codes,
                "submission_hash": submission_hash,
            }
            st.session_state.last_upload_fingerprint = fingerprint
            if decision != "MANUAL_REVIEW":
                log_history(
                    employee_name,
                    streak,
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
        else:
            st.info("This file was already processed. Click OK/Submit decision first, or upload a different file.")

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
                right = extracted.get("approver_name", "") or extracted.get("approved", "-")
            elif field == "employee_name":
                right = format_person_name_display(extracted.get(field, "")) or extracted.get(field, "-")
            else:
                right = extracted.get(field, "-")
            html += row_html(field, left, right, matched)
        html += "</table>"
        st.markdown(html, unsafe_allow_html=True)

        if comparison.get("day_hours_missing_in_extracted"):
            st.info("Day-wise hours not found in uploaded sheet. Comparison is based on period and total hours.")
        elif comparison["mismatches"].get("day_hours"):
            st.subheader("Day-wise Mismatches")
            st.dataframe(comparison["mismatches"]["day_hours"], use_container_width=True)
        else:
            st.success("All day-wise hours matched.")

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
            manual_outcome = st.selectbox("Outcome", ["APPROVE", "REJECT"], key="manual_outcome")
            manual_comment = st.text_area("Comment", key="manual_comment")
            if st.button("Submit Manual Decision"):
                if manual_outcome == "APPROVE":
                    new_streak = streak + 1
                    set_streak(employee_name, vendor, company, t_hash, p_key, new_streak)
                    st.success(f"Manual approved; streak is now {new_streak}")
                else:
                    new_streak = 0
                    set_streak(employee_name, vendor, company, t_hash, p_key, new_streak)
                    st.warning("Streak reset to 0")
                log_history(
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

    st.caption("No file is saved to uploads folder. Extraction runs directly from in-memory upload bytes.")


if __name__ == "__main__":
    main()
