import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent / ".env"
    load_dotenv(_ENV_PATH, override=True)
except ImportError:
    pass

# Prevents unsupported error caused by type mismatch by GPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import httpx   # Used to raise timeout errors, since large tables stalled the run 
import ollama
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from openai import OpenAI
    _OPENAI_INSTALLED = True
except ImportError:
    _OPENAI_INSTALLED = False

MODEL_LOCAL = "qwen2.5vl:3b"
MODEL_OPENAI = "gpt-4o-mini"

# Hard cap on a single local-model call. If qwen hangs on a table, kill the call and move on.
LOCAL_TIMEOUT_SECONDS = 120
_OLLAMA = ollama.Client(timeout=LOCAL_TIMEOUT_SECONDS)

PROMPT = """You are a data-extraction tool. You will receive ONE TABLE from a PDF, exported as HTML with its structure preserved: multi-level column headers via nested <thead> rows and <th colspan="N">, merged cells via colspan/rowspan, caption via the <caption> tag if present.

Transform it into a single JSON object:

{
  "title": "<the table's caption or heading if provided, otherwise null>",
  "data": <the table's contents, structured to mirror its layout>
}

Choose the data structure that best fits the table:

1. Flat table with one header row + data rows -> array of row objects:
   [{"col_a": value, "col_b": value}, ...]

2. Two-column key/value table -> flat object:
   {"label_a": value, "label_b": value}

3. Multi-level column headers (a <th> with colspan groups inner columns) -> for each row, nest the inner columns under the outer column key. Example:

   <thead>
     <tr><th rowspan=2>Activity</th><th colspan=2>Cycle 1</th><th>Cycle 2</th></tr>
     <tr><th>D1</th><th>D8</th><th>D1</th></tr>
   </thead>
   <tbody>
     <tr><td>PK draw</td><td>X</td><td>X</td><td>X</td></tr>
   </tbody>

   ->

   [{"activity": "PK draw", "cycle_1": {"d1": "X", "d8": "X"}, "cycle_2": {"d1": "X"}}]

4. Nested sub-table (a <td> contains its own <table>) -> nested object/array mirroring the sub-table's shape.
5. Preserve column-header text verbatim when converting to keys (lowercase + replace spaces with underscores; don't reorder, paraphrase, or drop words). Drop only non-ASCII glyphs like ± and °.
6. Header-cell counting: every <th> cell in the header band MUST produce a corresponding key in the output. If a <th colspan="N"> groups inner columns,
that key must contain exactly N sub-keys.

Field rules:
- ALL keys MUST be snake_case ("Protocol Number" -> "protocol_number"). Never emit a key with spaces, capitals, or punctuation.
- Pluralize the key for a list (sites, patients, events).
- Strip thousands-separator commas; emit numbers as int or float ("125,000" -> 125000).
- Keep percentages and units as strings ("99.9%", "5 mg/kg").
- Empty cells, "-", "--", "N/A" become null.
- Single "X", "✓", or footnote-marked "X^a" in a schedule grid stays as the string ("X" or "X^a").
- If a row spans the entire width with descriptive text instead of cell values, capture the text as: {"<row_label_snake_case>": "<the full text>"}.

Output ONLY the JSON object described above with exactly the two top-level keys "title" and "data". Do NOT emit a "cells" array, Row_Start/Col_End coordinates, or any other schema. No markdown, no commentary."""


def is_complex_table(html):
    """Should this table be routed to the stronger cloud model?
    Any one of:
      - 3+ <tr> rows containing <th> cells  (nested header band)
      - Any <th colspan="N"> with N >= 2    (grouped columns)
      - Any row with 10+ <td> cells         (wide schedule grid)
    """
    tr_blocks = re.findall(r"<tr.*?>.*?</tr>", html, flags=re.DOTALL | re.IGNORECASE)
    header_rows = sum(1 for tr in tr_blocks if re.search(r"<th[ >]", tr))
    if header_rows >= 3:
        return True
    if re.search(r'<th[^>]*\bcolspan="([2-9]|\d{2,})"', html):
        return True
    for tr in tr_blocks:
        if len(re.findall(r"<td[ >]", tr)) >= 10:
            return True
    return False


def _openai_client():
    if not _OPENAI_INSTALLED:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY env var not set. Get a key at https://platform.openai.com/api-keys"
        )
    return OpenAI(api_key=api_key)


def _user_block(table_html, caption):
    return f"CAPTION: {caption or '(none)'}\n\nTABLE HTML:\n{table_html}"


def normalize_table_local(table_html, caption):
    """Send one table's HTML to local Ollama; retry once on JSON parse failure. Uses the module-level _OLLAMA client which has a
    timeout. If a single generate call hangs past LOCAL_TIMEOUT_SECONDS, httpx raises a TimeoutException that propagates up to
    extract_pdf and the table gets recorded as failed instead of stalling the run."""
    user_block = _user_block(table_html, caption)
    last_err = None
    for attempt in range(2):
        resp = _OLLAMA.generate(
            model=MODEL_LOCAL,
            prompt=f"{PROMPT}\n\n{user_block}",
            format="json",
            options={
                "temperature": 0 if attempt == 0 else 0.2,
                "num_ctx": 16384,
                "num_predict": -1,
            },
        )
        try:
            return json.loads(resp["response"])
        except json.JSONDecodeError as e:
            last_err = e
    raise last_err


def normalize_table_openai(table_html, caption, retry_hint=None, request_logprobs=False):
    """Send one table's HTML to OpenAI for structural reasoning. If retry_hint is provided, it
    is appended to the system prompt as explicit feedback about previous mistakes.

    When request_logprobs is True, the API call includes `logprobs=True` so the caller can
    compute a token-level confidence score. In that mode we return (parsed_dict, raw_response)
    so the caller has access to the per-token logprobs metadata. Otherwise just the dict.
    """
    client = _openai_client()
    system_content = PROMPT
    if retry_hint:
        system_content += (
            "\n\nIMPORTANT - a previous attempt produced output with the following "
            f"errors. Re-extract carefully, correcting these issues:\n{retry_hint}"
        )
    kwargs = {
        "model": MODEL_OPENAI,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": _user_block(table_html, caption)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    if request_logprobs:
        kwargs["logprobs"] = True
    resp = client.chat.completions.create(**kwargs)
    parsed = json.loads(resp.choices[0].message.content)
    if request_logprobs:
        return parsed, resp
    return parsed


def _compute_logprob_confidence(resp, threshold=None):
    """Token-level confidence from an OpenAI chat-completion response.
    Score = mean per-token probability (mean of exp(logprob) across every token in the generated JSON). A score close to 1.0 means 
    the model was uniformly confident; a low score means at least some tokens were near-coin-flips.
    We also surface diagnostic info:
      * `min_token_prob`  — the single least-confident token
      * `low_conf_count`  — how many tokens were below 0.5
      * `worst_tokens`    — the eight least-confident tokens (label + prob), for human review
    Returns None if the response doesn't carry logprobs (caller forgot to request them)."""
    try:
        tokens = resp.choices[0].logprobs.content
    except (AttributeError, IndexError):
        return None
    if not tokens:
        return None
    probs = [(t.token, math.exp(t.logprob))
             for t in tokens if t.logprob is not None]
    if not probs:
        return None
    plist = [p for _, p in probs]
    mean_p = sum(plist) / len(plist)
    min_p = min(plist)
    low_conf = sum(1 for p in plist if p < 0.5)
    worst = sorted(probs, key=lambda x: x[1])[:8]
    out = {
        "method":          "logprobs_mean",
        "score":           round(mean_p, 4),
        "n_tokens":        len(plist),
        "min_token_prob":  round(min_p, 4),
        "low_conf_count":  low_conf,
        "worst_tokens":    [{"token": t, "prob": round(p, 4)} for t, p in worst],
    }
    if threshold is not None:
        out["threshold"]    = threshold
        out["needs_review"] = mean_p < threshold
    return out

def _count_leaves(obj):
    """Count every leaf value in a nested dict/list structure."""
    if isinstance(obj, dict):
        return sum(_count_leaves(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(_count_leaves(v) for v in obj)
    return 1   # str, int, float, None, bool


def _row_data_leaves(row):
    """Count leaves in one row object, ignoring whichever key carries the row label. Skip the 'activity' key if present; otherwise skip the first key."""
    if not isinstance(row, dict) or not row:
        return _count_leaves(row)
    skip = "activity" if "activity" in row else next(iter(row.keys()))
    return sum(_count_leaves(v) for k, v in row.items() if k != skip)


def _expected_row_width(html):
    """Most common leaf-column count across data rows in the HTML, with colspan on data cells expanded. This is the canonical "row width" for a wide
    grid."""
    tr_blocks = re.findall(r"<tr.*?>.*?</tr>", html, flags=re.DOTALL | re.IGNORECASE)
    widths = []
    for tr in tr_blocks:
        # Find every <td ...> and its attributes (if any). We use a single regex that captures the attribute block.
        cells = re.findall(r"<td(\s[^>]*)?>", tr)
        if not cells:
            continue
        w = 0
        for attrs in cells:
            m = re.search(r'\bcolspan\s*=\s*"(\d+)"', attrs or "")
            w += int(m.group(1)) if m else 1
        widths.append(w)
    if not widths:
        return 0
    from collections import Counter
    return Counter(widths).most_common(1)[0][0]


def _grid_mismatches(result, html):
    """For an array-of-rows result, find rows whose leaf count doesn't match either the expected row width (regular row) or 1 (full-width text-span row).
    Returns (expected_width, list_of_mismatches). Each mismatch is (row_index, label, actual_leaves). Returns (0, []) if the table doesn't look like a
    wide grid worth checking.
    """
    expected = _expected_row_width(html)
    if expected < 5:
        return expected, []   # too narrow to be a grid; skip validation
    data = result.get("data")
    if not isinstance(data, list):
        return expected, []
    out = []
    for i, row in enumerate(data):
        leaves = _row_data_leaves(row)
        if leaves != expected and leaves != 1:
            # Try to find a label for the diagnostic hint
            label = "?"
            if isinstance(row, dict):
                label = str(row.get("activity") or next(iter(row.values()), "?"))[:50]
            out.append((i, label, leaves))
    return expected, out


def _build_retry_hint(expected, mismatches):
    """Format a concise feedback string for the model's retry attempt."""
    lines = [
        f"This table has EXACTLY {expected} leaf data columns per regular row.",
        "When you nest cycles/days/timepoints, the total number of leaf values "
        f"in each row (excluding the row label) MUST be {expected}, OR exactly 1 "
        "if the row is a full-width descriptive text row.",
        "Your previous output had these incorrect rows:",
    ]
    for i, label, actual in mismatches[:6]:   # cap hint to 6 examples
        lines.append(f"  - row {i} ({label!r}): produced {actual} leaf values, expected {expected}")
    if len(mismatches) > 6:
        lines.append(f"  - ...and {len(mismatches)-6} more")
    return "\n".join(lines)


_DISPATCH_PROBE_PRINTED = False
_NO_OPENAI = os.environ.get("NO_OPENAI") == "1"

# Human-in-the-loop confidence scoring (opt-in via --confidence-check / CONFIDENCE_CHECK=1).
# When on, every OpenAI call requests per-token logprobs; we compute the mean token probability and tag any result below
# _CONFIDENCE_THRESHOLD with `_confidence.needs_review=true` so the UI can flag the table for a human reviewer. Local-fallback 
# results don't carry confidence info because Ollama doesn't surface logprobs cleanly through this client.
_CONFIDENCE_CHECK = os.environ.get("CONFIDENCE_CHECK") == "1"
try:
    _CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
except ValueError:
    _CONFIDENCE_THRESHOLD = 0.7


def normalize_table(table_html, caption):
    """Route complex tables to OpenAI when available. For complex grids, validate the output's row widths and retry once with
    explicit feedback if there's a miscount. Returns (normalized_dict, backend_label). Falls back to local on API errors."""
    global _DISPATCH_PROBE_PRINTED
    key_present = bool(os.environ.get("OPENAI_API_KEY"))
    openai_ready = _OPENAI_INSTALLED and key_present and not _NO_OPENAI
    if not _DISPATCH_PROBE_PRINTED:
        print(
            f"  [dispatcher] _OPENAI_INSTALLED={_OPENAI_INSTALLED} "
            f"key_present={key_present} "
            f"no_openai={_NO_OPENAI} openai_ready={openai_ready}",
            file=sys.stderr, flush=True,
        )
        _DISPATCH_PROBE_PRINTED = True
    complex_ = is_complex_table(table_html)
    if complex_:
        cached = _load_openai_cache(table_html, caption)
        if cached is not None and (not _CONFIDENCE_CHECK or "_confidence" in cached):
            print(f"  [cache] openai hit -> {_openai_cache_path(table_html, caption).name}",
                  file=sys.stderr, flush=True)
            return cached, f"cloud:openai/{MODEL_OPENAI} (cached)"

    if complex_ and openai_ready:
        try:
            # Initial attempt. Request logprobs only when confidence-check is on, otherwise the API response payload stays minimal.
            if _CONFIDENCE_CHECK:
                result, raw = normalize_table_openai(table_html, caption, request_logprobs=True)
                confidence = _compute_logprob_confidence(raw, threshold=_CONFIDENCE_THRESHOLD)
            else:
                result = normalize_table_openai(table_html, caption)
                confidence = None
            expected, mismatches = _grid_mismatches(result, table_html)
            if mismatches:
                print(
                    f"  [validate] {len(mismatches)} row(s) miscounted "
                    f"(expected {expected} leaves); retrying with hint",
                    file=sys.stderr, flush=True,
                )
                hint = _build_retry_hint(expected, mismatches)
                # Retry. Re-request logprobs so we can replace the confidence score if the retry wins (the original score belongs
                #  to the discarded attempt).
                if _CONFIDENCE_CHECK:
                    retried, retried_raw = normalize_table_openai(
                        table_html, caption, retry_hint=hint, request_logprobs=True)
                else:
                    retried = normalize_table_openai(table_html, caption, retry_hint=hint)
                    retried_raw = None
                _, retried_mm = _grid_mismatches(retried, table_html)
                # Keep whichever attempt has fewer mismatches; original wins on tie
                if len(retried_mm) < len(mismatches):
                    print(
                        f"  [validate] retry improved: {len(mismatches)} -> {len(retried_mm)} mismatches",
                        file=sys.stderr, flush=True,
                    )
                    result = retried
                    if retried_raw is not None:
                        confidence = _compute_logprob_confidence(
                            retried_raw, threshold=_CONFIDENCE_THRESHOLD)
                else:
                    print(
                        f"  [validate] retry did not improve ({len(retried_mm)} mismatches); keeping original",
                        file=sys.stderr, flush=True,
                    )
            # Attach confidence info AFTER the retry decision so it reflects the actually-kept attempt. Log when human review is 
            # recommended.
            if confidence is not None:
                result["_confidence"] = confidence
                if confidence.get("needs_review"):
                    print(
                        f"  [confidence] score={confidence['score']:.3f} < "
                        f"threshold={_CONFIDENCE_THRESHOLD} -> needs human review "
                        f"({confidence['low_conf_count']} low-conf token(s), "
                        f"min={confidence['min_token_prob']:.3f})",
                        file=sys.stderr, flush=True,
                    )
            # Only cache on OpenAI success. Local-fallback results stay un-cached.
            _save_openai_cache(table_html, caption, result)
            return result, f"cloud:openai/{MODEL_OPENAI}"
        except Exception as e:
            print(f"  (openai failed, falling back to local: {e})",
                  file=sys.stderr, flush=True)
    return normalize_table_local(table_html, caption), f"local:{MODEL_LOCAL}"


def _table_html(table, doc):
    """HTML export."""
    try:
        return table.export_to_html(doc=doc)
    except TypeError:
        return table.export_to_html()


def _table_caption(table, doc):
    """Best-effort caption text."""
    try:
        text = table.caption_text(doc)
        return text.strip() if text else None
    except (AttributeError, TypeError):
        return None


def _table_page(table):
    """Page number (1-indexed) from a table's provenance, or None."""
    try:
        return table.prov[0].page_no
    except (AttributeError, IndexError):
        return None


#  cross-page table merging
# Long tables (schedule of events, lab panels) are typically  detected by Docling as separate tables when they break across pages.
#  These functions stitch them back together based on caption + adjacency + schema.

_CONTINUED_RE = re.compile(r"\bcont(?:inued|'d|\.)\b|\(continued\)", re.IGNORECASE)
_PAREN_CONT_RE = re.compile(r"\s*\(?\s*cont(?:inued|'d|\.)?\s*\)?\s*$", re.IGNORECASE)


def _normalize_caption(cap):
    """Strip trailing 'continued' markers and lowercase so the SAME caption on a continuation page matches its predecessor."""
    if not cap:
        return ""
    return _PAREN_CONT_RE.sub("", cap).strip().lower()


def _schemas_compatible(prev_data, curr_data, threshold=0.7):
    """True if both tables are array-of-dict and their row schemas overlap by at least `threshold` of the larger key set."""
    if not (isinstance(prev_data, list) and isinstance(curr_data, list)):
        return False
    if not prev_data or not curr_data:
        return False
    if not (isinstance(prev_data[0], dict) and isinstance(curr_data[0], dict)):
        return False
    p_keys = set(prev_data[0].keys())
    c_keys = set(curr_data[0].keys())
    if not p_keys or not c_keys:
        return False
    return len(p_keys & c_keys) / max(len(p_keys), len(c_keys)) >= threshold


def _is_continuation(prev, curr):
    """Should `curr` be merged into `prev` as a continuation of the same table?"""
    p_page = prev.get("page")
    c_page = curr.get("page")
    if p_page is None or c_page is None:
        return False
    # Adjacent pages only
    if c_page - p_page not in (0, 1):
        return False
    p_cap = _normalize_caption(prev.get("title"))
    c_cap = _normalize_caption(curr.get("title"))
    curr_title_raw = curr.get("title") or ""
    has_continued_marker = bool(_CONTINUED_RE.search(curr_title_raw))
    caption_signal = (
        (p_cap and c_cap and p_cap == c_cap) or   # same caption (normalized)
        has_continued_marker or                    # explicit "(continued)"
        (p_cap and not c_cap)                      # prev has caption, curr doesn't
    )
    if not caption_signal:
        return False
    return _schemas_compatible(prev.get("data"), curr.get("data"))


def _merge_into(prev, curr):
    """Mutate `prev` in place: append curr's rows, union footnotes/discussions,
    record the additional page in prev['pages']."""
    prev["data"] = (prev.get("data") or []) + (curr.get("data") or [])
    # Page list - keep prev["page"] as the primary anchor
    pages = prev.get("pages") or [prev.get("page")]
    if curr.get("page") not in pages:
        pages.append(curr.get("page"))
    prev["pages"] = pages
    # Union footnotes (prev wins on conflict)
    if curr.get("footnotes"):
        prev.setdefault("footnotes", {})
        for k, v in curr["footnotes"].items():
            prev["footnotes"].setdefault(k, v)
    # Concatenate discussions, de-duplicated by text
    if curr.get("discussions"):
        existing_texts = {d.get("text") for d in (prev.get("discussions") or [])}
        for d in curr["discussions"]:
            if d.get("text") not in existing_texts:
                prev.setdefault("discussions", []).append(d)
                existing_texts.add(d.get("text"))
    # Record the continuation bbox so a frontend can highlight all pages.
    if curr.get("_bbox"):
        prev.setdefault("_bboxes_continued", []).append({
            "page": curr.get("page"),
            "bbox": curr["_bbox"],
        })
    # Count of how many physical tables were folded into this one.
    prev["_merged_from"] = (prev.get("_merged_from") or 1) + 1
    return prev


def merge_continued_tables(tables):
    """Walk the table list once; whenever consecutive tables look like continuations of the same logical table, fold them together."""
    if not tables:
        return tables
    merged = [tables[0]]
    for curr in tables[1:]:
        if _is_continuation(merged[-1], curr):
            _merge_into(merged[-1], curr)
        else:
            merged.append(curr)
    return merged


def _table_bbox(table):
    try:
        b = table.prov[0].bbox
    except (AttributeError, IndexError):
        return None
    for attrs in (
        ("l", "t", "r", "b"),
        ("left", "top", "right", "bottom"),
        ("x0", "y0", "x1", "y1"),
    ):
        try:
            return {
                "left":   float(getattr(b, attrs[0])),
                "top":    float(getattr(b, attrs[1])),
                "right":  float(getattr(b, attrs[2])),
                "bottom": float(getattr(b, attrs[3])),
            }
        except AttributeError:
            continue
    return None


# Main-text anchoring
# Find body paragraphs that reference the table by number (e.g. "Table 7  shows...").

_TABLE_NUM_FROM_CAPTION = re.compile(r"\s*Table\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


def _table_number(caption):
    """Pull the author-assigned table number from its caption, e.g. 'Table 7. Schedule of Pharmacodynamic Assessments' -> '7'.
    Returns None if no number found (we can't anchor an unnumbered table)."""
    if not caption:
        return None
    m = _TABLE_NUM_FROM_CAPTION.match(caption)
    return m.group(1) if m else None


def find_table_discussions(doc, table_number, table_caption, max_refs=5, max_chars=1500):
    """Walk the document's text items and find paragraphs that reference 'Table N'. Returns a list of {page, text} dicts.
    Filters out:
      - The table's own caption.
      - Lines that ARE captions (start with 'Table N').
      - Items beyond `max_refs` discussions or `max_chars` per snippet.
    """
    if not table_number:
        return []
    ref_pattern = re.compile(
        rf"\bTable\s+{re.escape(table_number)}\b", re.IGNORECASE
    )
    caption_prefix = (table_caption or "").strip()[:30]
    starts_like_caption = re.compile(
        rf"\s*Table\s+{re.escape(table_number)}\b", re.IGNORECASE
    )
    out = []
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text or not ref_pattern.search(text):
            continue
        clean = text.strip()
        if caption_prefix and clean.startswith(caption_prefix):
            continue
        if starts_like_caption.match(clean):
            continue
        page = None
        try:
            page = item.prov[0].page_no
        except (AttributeError, IndexError):
            pass
        out.append({
            "page": page,
            "text": clean[:max_chars] + ("..." if len(clean) > max_chars else ""),
        })
        if len(out) >= max_refs:
            break
    return out


# Matches the dot-leader + page-number pattern unique to TOC entries:
# "Section title......................42"
_TOC_VALUE_PATTERN = re.compile(r"\.{5,}\s*\d+\s*$")


def _flatten_values(obj):
    """Yield every leaf value out of an arbitrarily nested dict/list structure."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten_values(v)
    else:
        yield obj


def is_toc_like_table(table):
    """A Table of Contents / List of Tables / List of Figures, not real data.
    Two independent signals; either alone qualifies:
      1. Title explicitly mentions it.
      2. >=60% of leaf string values match the dot-leader + page-number pattern.
    "List of Abbreviations" is intentionally NOT flagged - it's a useful reference dictionary even though it shares a title family with the TOC group.
    """
    title = (table.get("title") or "").lower()
    if any(kw in title for kw in (
        "table of contents", "list of tables", "list of figures"
    )):
        return True

    data = table.get("data")
    if not isinstance(data, dict) or len(data) < 5:
        return False    # only flat dicts qualify; short tables are probably real data

    str_values = [v for v in _flatten_values(data) if isinstance(v, str)]
    if not str_values:
        return False
    toc_hits = sum(1 for v in str_values if _TOC_VALUE_PATTERN.search(v))
    return toc_hits / len(str_values) >= 0.6


# footnote synthesis
# Tables in clinical-trial PDFs often carry single-letter footnote markers ("X^a", "Fresh tumor biopsy c") whose definitions appear in plain text right
# after the table. These functions:
# (1) find which markers a table uses,
# (2) collect text from the same page after the table,
# (3) parse footnote definitions, and
# (4) attach them as a top-level `footnotes` dict so a downstream consumer can resolve any "X^a" or label-trailing letter back to
# its meaning.
# Marker patterns in HTML cells:
#   "X^a", "X a", "X<sup>a</sup>"  (cell with X + footnote letter)
#   "Fresh tumor biopsy c"         (row label ending with a single letter)
_CELL_MARKER_PATTERN = re.compile(r"\bX\s*\^?\s*([a-z])\b")


def _markers_in_html(html):
    """Find single-letter footnote markers actually used in a table's HTML. Returns a set of lowercase letters (e.g. {'a', 'b'})."""
    markers = set()
    # Pattern 1: "X^a" or "X a" inside a cell with X
    for m in _CELL_MARKER_PATTERN.finditer(html):
        markers.add(m.group(1))
    # Pattern 2: row labels like "<th>Fresh tumor biopsy c</th>"
    for cell in re.findall(r"<t[hd][^>]*>([^<]*)</t[hd]>", html, flags=re.IGNORECASE):
        words = cell.strip().split()
        if words and len(words[-1]) == 1 and words[-1].isalpha() and words[-1].islower():
            markers.add(words[-1])
    return markers


def _page_text_after_table(doc, table):
    """Concatenate text items that appear AFTER the given table, on the same page, in document order. Capped at ~2000 chars."""
    table_page = _table_page(table)
    if table_page is None:
        return ""
    parts = []
    total = 0
    found = False
    for item, _level in doc.iterate_items():
        if item is table:
            found = True
            continue
        if not found:
            continue
        # Same page?
        item_page = None
        try:
            item_page = item.prov[0].page_no
        except (AttributeError, IndexError):
            pass
        if item_page != table_page:
            continue
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
            total += len(text)
            if total >= 2000:
                break
    # Join with blank line: each Docling text item is a paragraph, so this gives the footnote regex's "stop at blank line" rule a
    # boundary.
    return "\n\n".join(parts)


def _extract_footnote_definitions(text, markers):
    """Parse footnote definitions out of text. Matches lines that begin with one of the marker letters, optionally followed by ".", ":", or whitespace, then a capitalized word.
    Capture terminates at any of:
      - next footnote line (single letter + . : + capital)
      - a blank line (paragraph break)
      - 500 characters reached (length cap against runaway captures)
    """
    if not markers or not text:
        return {}
    found = {}
    for marker in markers:
        pattern = (
            rf"(?:^|\n)\s*{re.escape(marker)}\s*[\.\:]?\s+"
            r"([A-Z][^\n]+"
            r"(?:\n"
            r"   (?!\s*$)"                                # stop at blank line
            r"   (?!\s*[a-z]\s*[\.\:]?\s+[A-Z])"          # stop at next footnote
            r"   [^\n]+)*)"
        )
        m = re.search(pattern, text, flags=re.VERBOSE)
        if not m:
            continue
        cleaned = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(cleaned) > 500:
            cleaned = cleaned[:497].rstrip() + "..."
        found[marker] = cleaned
    return found


def attach_footnotes(table_dict, html, doc, table_obj):
    """If the table uses footnote markers, find their definitions in the surrounding page text and attach them to table_dict as a
    'footnotes' key."""
    markers = _markers_in_html(html)
    if not markers:
        return
    surrounding = _page_text_after_table(doc, table_obj)
    defs = _extract_footnote_definitions(surrounding, markers)
    if defs:
        table_dict["footnotes"] = defs


# Vector-geometry check (opt-in via --vector-check / VECTOR_CHECK=1)
# Reads the table directly from the PDF text layer using pdfplumber (ruling lines + char positions — no model). For each row, 
# counts how many cells hold an X / ✓ / X-with-footnote-letter marker, then compares against the LLM's output. Any row that 
# disagrees gets attached to the result as a `_geometric_warning'.
# This is annotation-only: it never modifies the LLM's `data`, never triggers a retry, never calls the API. 

_X_CELL_RE = re.compile(r"^\s*[X✓]\s*\^?\s*[a-z]?\s*$", re.IGNORECASE)


def _is_x_cell(value):
    if value is None:
        return False
    return bool(_X_CELL_RE.match(str(value)))


def _row_label_from_json(row):
    """Pull the row's display label from a JSON row dict."""
    if not isinstance(row, dict) or not row:
        return None
    if "activity" in row:
        v = row["activity"]
    else:
        v = next(iter(row.values()))
    return str(v).strip() if v is not None else None


def _count_x_leaves(row):
    """How many leaf values in this JSON row look like X cells, excluding the row-label key."""
    if not isinstance(row, dict) or not row:
        return 0
    skip = "activity" if "activity" in row else next(iter(row.keys()))
    n = 0
    for k, v in row.items():
        if k == skip:
            continue
        for leaf in _flatten_values(v):
            if _is_x_cell(leaf):
                n += 1
    return n


def _docling_bbox_to_pdfplumber(bbox, page_height):
    """Convert a docling bbox dict (PDF points, y-up if top > bottom) into a pdfplumber tuple (x0, top, x1, bottom) in y-down 
    coordinates. Returns None if bbox is missing required keys."""
    if not bbox:
        return None
    try:
        left   = float(bbox["left"])
        right  = float(bbox["right"])
        top    = float(bbox["top"])
        bottom = float(bbox["bottom"])
    except (KeyError, TypeError, ValueError):
        return None
    # If top > bottom in docling space, the bbox is y-up; flip to y-down.
    if top > bottom:
        pp_top, pp_bottom = page_height - top, page_height - bottom
    else:
        pp_top, pp_bottom = top, bottom
    return (left, pp_top, right, pp_bottom)


def _vector_check_table(pdf_path, page_no, table_bbox):
    """Read the table at (page_no, table_bbox) from the PDF directly. Returns
        {row_label: {"x_count": N, "cells": [data_cell_strings...]}}
    where `cells` is the row's data cells (label cell stripped, one entry per visible
    column, values preserved verbatim). Returns None if pdfplumber isn't installed or
    extraction yields nothing."""
    if not pdf_path or page_no is None:
        return None
    try:
        import pdfplumber
    except ImportError:
        print("  [vector-check] pdfplumber not installed; skipping. "
              "pip install pdfplumber", file=sys.stderr, flush=True)
        return None

    try:
        with pdfplumber.open(pdf_path) as pdf:
            idx = page_no - 1   # docling is 1-indexed
            if idx < 0 or idx >= len(pdf.pages):
                return None
            page = pdf.pages[idx]
            crop = _docling_bbox_to_pdfplumber(table_bbox, page.height)
            if crop:
                pad = 4   # pad so we don't lose edge cells to rounding
                crop = (crop[0] - pad, crop[1] - pad, crop[2] + pad, crop[3] + pad)
                try:
                    page = page.crop(crop)
                except (ValueError, Exception):
                    pass    # bbox past page; fall back to full page
            tables = page.extract_tables()
    except Exception as e:
        print(f"  [vector-check] pdfplumber error ({type(e).__name__}: {e}); skipping",
              file=sys.stderr, flush=True)
        return None

    if not tables:
        return None
    grid = max(tables, key=len)

    out = {}
    for row in grid:
        # Locate the label cell: first non-empty, non-X cell. Data cells = everything to the right of it. 
        label = None
        label_idx = None
        for i, c in enumerate(row):
            t = (c or "").strip()
            if not t:
                continue
            if _is_x_cell(t):
                break
            label = re.sub(r"\s+", " ", t)
            label_idx = i
            break
        if not label:
            continue
        data_cells = list(row[label_idx + 1:]) if label_idx is not None else list(row)
        x_count = sum(1 for c in data_cells if _is_x_cell(c))
        prev = out.get(label)
        # If the same label appears twice (continuation rows), keep the one with more X's.
        if prev is None or x_count > prev["x_count"]:
            out[label] = {"x_count": x_count, "cells": data_cells}

    # Compress columns that are empty in every data row. Dropping the all-empty columns collapses pdfplumber's column count back 
    # down to what the LLM saw.
    return _drop_structural_empty_columns(out)


def _drop_structural_empty_columns(geo_data):
    """For columns that are empty/None in every DATA row, drop them across all rows. pdfplumber over-segments multi-level headers, leaving phantom
    spacer columns that have header TEXT ("D1") but no data in any row that has X glyphs. If we let header rows participate in the column-emptiness vote, those phantom columns
    survive and the cell count won't align with the LLM's leaves. By only considering rows with at least one X marker, we keep 
    columns that carry real data and drop the spacers.

    `x_count` is unchanged because we never drop a column that has an X anywhere."""
    if not geo_data:
        return geo_data
    n_cols = max(len(info["cells"]) for info in geo_data.values())
    padded = {}
    for label, info in geo_data.items():
        cells = list(info["cells"])
        while len(cells) < n_cols:
            cells.append(None)
        padded[label] = (info["x_count"], cells)

    def is_blank(c):
        return c is None or (isinstance(c, str) and c.strip() == "")

    data_labels = [lbl for lbl, (xc, _) in padded.items() if xc > 0]
    voters = data_labels if data_labels else list(padded.keys())

    keep = [j for j in range(n_cols)
            if any(not is_blank(padded[lbl][1][j]) for lbl in voters)]
    if len(keep) == n_cols:
        return geo_data
    return {
        label: {"x_count": xc, "cells": [cells[j] for j in keep]}
        for label, (xc, cells) in padded.items()
    }


def _geometric_mismatches(normalized, geo_data):
    """Compare per-row X counts between the LLM's JSON and pdfplumber's geometric grid.
    `geo_data` is what _vector_check_table returns: {label: {"x_count": N, "cells": [...]}}.
    Returns mismatch dicts of two kinds:
      kind="count_diff" -- a row in both LLM and PDF, but X counts disagree.
      kind="missing_row" -- a row exists in the PDF (with at least one X glyph) but no LLM
        row matches its label (LLM dropped the row or mis-interpreted the table)."""
    if not geo_data:
        return []
    data = normalized.get("data")
    if not isinstance(data, list):
        return []
    geo_norm = {re.sub(r"\s+", " ", k).strip().lower(): (k, v)
                for k, v in geo_data.items()}

    mismatches = []
    matched_geo_keys = set()

    for row in data:
        label = _row_label_from_json(row)
        if not label:
            continue
        key = re.sub(r"\s+", " ", label).strip().lower()
        geo_label, geo_info, matched_key = None, None, None
        if key in geo_norm:
            geo_label, geo_info = geo_norm[key]
            matched_key = key
        else:
            for gk, (orig, info) in geo_norm.items():
                if gk in key or key in gk:
                    geo_label, geo_info, matched_key = orig, info, gk
                    break
        if geo_label is None:
            continue
        matched_geo_keys.add(matched_key)
        json_count = _count_x_leaves(row)
        geo_count = geo_info["x_count"]
        if json_count != geo_count:
            mismatches.append({
                "kind":          "count_diff",
                "row":           geo_label,
                "llm_x_count":   json_count,
                "pdf_x_count":   geo_count,
                "delta":         json_count - geo_count,
            })
    for gk, (orig, info) in geo_norm.items():
        if gk in matched_geo_keys or info["x_count"] == 0:
            continue
        mismatches.append({
            "kind":          "missing_row",
            "row":           orig,
            "llm_x_count":   None,
            "pdf_x_count":   info["x_count"],
        })
    return mismatches


def attach_geometric_warning(table_dict, pdf_path, page_no, table_bbox, geo_data=None):
    """Run the pdfplumber check (unless `geo_data` is passed in) and attach `_geometric_warning` to `table_dict` if any row's X 
    count differs. No-op when pdfplumber isn't installed, the bbox is missing, or every row agrees."""
    if geo_data is None:
        geo_data = _vector_check_table(pdf_path, page_no, table_bbox)
    if not geo_data:
        return
    mismatches = _geometric_mismatches(table_dict, geo_data)
    if mismatches:
        table_dict["_geometric_warning"] = {
            "source":     "pdfplumber",
            "mismatches": mismatches,
        }
        counts_by_kind = {}
        for m in mismatches:
            counts_by_kind[m["kind"]] = counts_by_kind.get(m["kind"], 0) + 1
        summary = ", ".join(f"{n} {k}" for k, n in counts_by_kind.items())
        first3 = "; ".join(
            f"{m['row']!r} ({m['kind']}: llm={m['llm_x_count']} pdf={m['pdf_x_count']})"
            for m in mismatches[:3]
        )
        more = f" (+{len(mismatches)-3} more)" if len(mismatches) > 3 else ""
        print(f"  [vector-check] {len(mismatches)} row(s) flagged ({summary}): {first3}{more}",
              file=sys.stderr, flush=True)

# Cell-level correction (opt-in via --vector-correct / VECTOR_CORRECT=1)
# Uses pdfplumber's per-cell readings and patches the LLM's leaves in place. Aligns by position: the LLM's row leaves are zipped 
# 1:1 against pdfplumber's data-cell list for that same row. Each cell where the two disagree is rewritten with the PDF's value, 
# and the change is logged to `_geometric_corrections`.
# Two-way: a hallucinated X becomes null, and a *missed* X becomes "X". The LLM's structural choice is preserved -- only leaf 
# values change.

def _leaves_with_paths(obj, path):
    """Yield (path, leaf_value) pairs for every leaf in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves_with_paths(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves_with_paths(v, path + (i,))
    else:
        yield path, obj


def _row_data_leaf_paths(row):
    """All leaf (path, value) pairs in a JSON row, with the label key excluded."""
    if not isinstance(row, dict) or not row:
        return
    skip = "activity" if "activity" in row else next(iter(row.keys()))
    for k, v in row.items():
        if k == skip:
            continue
        yield from _leaves_with_paths(v, (k,))


def _set_at_path(row, path, value):
    """Walk path keys and replace the final leaf with `value`."""
    cur = row
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


def _pdf_cell_to_jsonval(cell):
    """Convert a pdfplumber cell string to the JSON value the LLM SHOULD have produced. Empty / None / '-' / 'N/A' -> None.  
    X-with-optional-footnote -> 'X' or 'X^a'."""
    if cell is None:
        return None
    s = str(cell).strip()
    if not s or s in ("-", "--", "N/A", "n/a"):
        return None
    m = re.match(r"^([X✓])\s*\^?\s*([a-z])?\s*$", s, re.IGNORECASE)
    if m:
        return f"X^{m.group(2).lower()}" if m.group(2) else "X"
    return s


def _equivalent(a, b):
    """True if two JSON values represent the same cell content. Treats None/''/N-A as one bucket and is case-insensitive for short markers like X/Xa."""
    na = _pdf_cell_to_jsonval(a) if isinstance(a, str) else a
    nb = _pdf_cell_to_jsonval(b) if isinstance(b, str) else b
    if na is None and nb is None:
        return True
    if na is None or nb is None:
        return False
    return str(na).strip().lower() == str(nb).strip().lower()


def _correct_row_in_place(row, pdf_cells):
    """Position-align LLM leaves with PDF cells and rewrite mismatches in place. Returns [{"path": [...], "old": X, "new": Y}, ...]. Returns [] if leaf count
    doesn't match cell count."""
    leaves = list(_row_data_leaf_paths(row))
    if len(leaves) != len(pdf_cells):
        return []
    corrections = []
    for (path, old_val), pdf_val in zip(leaves, pdf_cells):
        new_val = _pdf_cell_to_jsonval(pdf_val)
        if _equivalent(old_val, new_val):
            continue
        _set_at_path(row, path, new_val)
        corrections.append({"path": list(path), "old": old_val, "new": new_val})
    return corrections


def apply_geometric_corrections(table_dict, pdf_path, page_no, table_bbox):
    """Run the geometric check and patch mismatched cells in the LLM data in place. Adds `_geometric_corrections` listing every 
    (row, path, old, new) tuple. Also calls attach_geometric_warning so the original mismatch info is preserved."""
    geo_data = _vector_check_table(pdf_path, page_no, table_bbox)
    if not geo_data:
        return
    # Attach the warning first so users can see what was off, even after patch.
    attach_geometric_warning(table_dict, pdf_path, page_no, table_bbox, geo_data=geo_data)

    data = table_dict.get("data")
    if not isinstance(data, list):
        return
    geo_norm = {re.sub(r"\s+", " ", k).strip().lower(): (k, v)
                for k, v in geo_data.items()}
    all_changes = []
    for row in data:
        label = _row_label_from_json(row)
        if not label:
            continue
        key = re.sub(r"\s+", " ", label).strip().lower()
        geo_label, geo_info = None, None
        if key in geo_norm:
            geo_label, geo_info = geo_norm[key]
        else:
            for gk, (orig, info) in geo_norm.items():
                if gk in key or key in gk:
                    geo_label, geo_info = orig, info
                    break
        if geo_info is None:
            continue
        row_changes = _correct_row_in_place(row, geo_info["cells"])
        if row_changes:
            all_changes.append({"row": geo_label, "corrections": row_changes})
    if all_changes:
        table_dict["_geometric_corrections"] = all_changes
        n_cells = sum(len(c["corrections"]) for c in all_changes)
        print(f"  [vector-correct] patched {n_cells} cell(s) across "
              f"{len(all_changes)} row(s) using pdfplumber",
              file=sys.stderr, flush=True)


# Off by default. The CLI parser may flip these to True when --vector-check / --vector-correct is passed. --vector-correct implies --vector-check.
_VECTOR_CHECK = os.environ.get("VECTOR_CHECK") == "1"
_VECTOR_CORRECT = os.environ.get("VECTOR_CORRECT") == "1"


# OpenAI result cache
# Successful OpenAI normalizations are cached.
# A complex table first checks the cache. On hit, returns the cached result with backend label 'cloud:openai/<model> (cached)' —
# no API call. On miss, tries OpenAI; on success the FINAL result (post-retry-and-validate) is written back. Local-fallback results are NEVER cached.
# This means: if the quota dies mid-run, a re-run picks up exactly where the last one left off — all OpenAI-succeeded tables come from cache (free), and the still-local ones get another OpenAI attempt.
# After enough re-runs every complex table converges to an OpenAI result. Cache hits still work when --no-openai is set (the flag
# blocks API calls, not free cache reads).
# Set --no-cache / NO_CACHE=1 to disable both reads and writes for one run. Wipe with `rm -rf .cache/openai`.

CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "openai"


def _openai_cache_key(html, caption):
    """ Same HTML+caption+prompt+model -> same key, regardless of which PDF it came from. Prompt + model are folded into the key 
    so prompt or model changes auto-invalidate stale entries."""
    payload = (
        PROMPT + "\n\n" +
        MODEL_OPENAI + "\n\n" +
        (html or "") + "\n\n" +
        (caption or "")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _openai_cache_path(html, caption):
    return CACHE_DIR / f"{_openai_cache_key(html, caption)}.json"


def _load_openai_cache(html, caption):
    """Return cached OpenAI result dict, or None on miss / corrupt / disabled."""
    if os.environ.get("NO_CACHE") == "1":
        return None
    p = _openai_cache_path(html, caption)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  cache read failed for {p.name} ({type(e).__name__}); ignoring",
              file=sys.stderr, flush=True)
        return None


def _save_openai_cache(html, caption, result):
    """Persist a successful OpenAI result so future runs don't re-pay quota."""
    if os.environ.get("NO_CACHE") == "1":
        return
    p = _openai_cache_path(html, caption)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        print(f"  cache write failed for {p.name} ({type(e).__name__}): {e}",
              file=sys.stderr, flush=True)


def _build_converter():
    """Docling configured to run on CPU. Apple Silicon MPS can't allocate float64 tensors which the RT-DETR layout model needs, so
      we pin to CPU."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CPU,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def extract_pdf(pdf_path):
    print(f"Converting {pdf_path} with Docling (CPU)...",
          file=sys.stderr, flush=True)
    converter = _build_converter()
    doc = converter.convert(pdf_path).document
    n = len(doc.tables)
    print(f"  Docling found {n} tables", file=sys.stderr, flush=True)

    out = {"tables": []}
    failed = []
    for i, table in enumerate(doc.tables, start=1):
        page = _table_page(table)
        caption = _table_caption(table, doc)
        html = _table_html(table, doc)
        complexity = "complex" if is_complex_table(html) else "simple"
        print(f"Table {i}/{n} (page {page}, {complexity})...",
              file=sys.stderr, flush=True)
        try:
            normalized, backend = normalize_table(html, caption)
        except (json.JSONDecodeError, ollama.ResponseError,
                httpx.TimeoutException) as e:
            # Distinguish timeouts in the log so a hung-table situation is obvious vs a normal parse/model error.
            kind = "TIMEOUT" if isinstance(e, httpx.TimeoutException) else type(e).__name__
            print(f"  ! table {i} failed ({kind}): {e}",
                  file=sys.stderr, flush=True)
            failed.append({
                "index": i, "page": page, "caption": caption,
                "reason": kind,
            })
            # Don't silently drop the table: emit a stub entry so the viewer still shows a card + bbox, flagged for human review.
            stub = {
                "title": caption,
                "data": None,
                "page": page,
                "_failed": {
                    "reason": kind,
                    "detail": str(e)[:200],
                    "needs_review": True,
                },
            }
            stub_bbox = _table_bbox(table)
            if stub_bbox is not None:
                stub["_bbox"] = stub_bbox
            out["tables"].append(stub)
            continue
        normalized["page"] = page
        normalized["_backend"] = backend
        # Frontend-anchoring data: bbox lets a viewer draw an overlay highlight when a cell is clicked.
        bbox = _table_bbox(table)
        if bbox is not None:
            normalized["_bbox"] = bbox
        # Find footnote markers in this table and try to resolve them to definitions from the surrounding page text.
        attach_footnotes(normalized, html, doc, table)
        # Main-text anchoring: paragraphs in the body text that reference this table by its author-assigned number.
        tnum = _table_number(caption)
        if tnum:
            discussions = find_table_discussions(doc, tnum, caption)
            if discussions:
                normalized["discussions"] = discussions
        # Opt-in geometric ground-truth check.
        # --vector-correct patches cell values in place using pdfplumber's reading and adds a `_geometric_corrections` flag. 
        # --vector-check only attaches a `_geometric_warning`. Either flag triggers the pdfplumber read.
        if _VECTOR_CORRECT:
            apply_geometric_corrections(normalized, pdf_path, page, bbox)
        elif _VECTOR_CHECK:
            attach_geometric_warning(normalized, pdf_path, page, bbox)
        out["tables"].append(normalized)
    # Post-process: fold consecutive continuation-pages into one logical table
    # BEFORE the TOC pass, so the TOC heuristic sees the consolidated shape.
    out["tables"] = merge_continued_tables(out["tables"])
    # Post-process: tag tables that look like Table of Contents or List of Tables/Figures, so downstream consumers can filter them out.
    for t in out["tables"]:
        if is_toc_like_table(t):
            t["_kind"] = "toc"
    if failed:
        out["failed_tables"] = failed
    return out


def _print_usage():
    print(
        "Usage: python textract.py [--no-openai] [--no-cache] [--vector-check|--vector-correct]\n"
        "                          [--confidence-check] [--confidence-threshold=N] <pdf>\n\n"
        "Options:\n"
        "  --no-openai             Force local-only routing (skip OpenAI API calls even if API key is set).\n"
        "                          Useful for an offline run. \n"
        "  --no-cache              Disable the OpenAI result cache (skip reads and writes).\n"
        "                          Default: cached OpenAI results are used when available; fresh OpenAI successes\n"
        "                          are written to .cache/openai/ so a re-run after quota recovery only re-tries\n"
        "                          the tables that previously fell back to local.\n"
        "  --vector-check          Compare each table's LLM output against a pdfplumber geometric extraction\n"
        "                          straight from the PDF text layer. Any row whose X-cell count disagrees gets\n"
        "                          a `_geometric_warning` attached. It does not modify the extracted data. Requires pdfplumber\n"
        "  --vector-correct        Like --vector-check, AND patch the LLM's mismatched cells in place using\n"
        "                          pdfplumber's reading as ground truth. Each correction is logged under\n"
        "                          `_geometric_corrections` for audit. Implies --vector-check. Skipped per-row\n"
        "                          when leaf count != cell count (structural disagreement -> no patching\n"
        "  --confidence-check      Human-in-the-loop scoring: request per-token logprobs from OpenAI, compute\n"
        "                          the mean token probability, and tag results below --confidence-threshold\n"
        "                          with `_confidence.needs_review=true` so the UI can flag the table.\n"
        "                          Only applies to OpenAI-routed tables (cloud:openai). Local fallbacks have\n"
        "                          no logprobs and so don't carry a confidence field.\n"
        "  --confidence-threshold=N  Float in (0, 1]; mean-token-probability cutoff for needing a human.\n"
        "                          Default: 0.7. Only used when --confidence-check is on.\n"
        "  --help                  Show this message and exit.\n\n"
        "Equivalent env vars:\n"
        "  NO_OPENAI=1, NO_CACHE=1, VECTOR_CHECK=1, VECTOR_CORRECT=1, CONFIDENCE_CHECK=1, CONFIDENCE_THRESHOLD=0.7\n",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raw_args = sys.argv[1:]
    flags = {a for a in raw_args if a.startswith("--")}
    positional = [a for a in raw_args if not a.startswith("--")]

    if "--help" in flags or not positional:
        _print_usage()
        sys.exit(0 if "--help" in flags else 1)

    if "--no-openai" in flags:
        _NO_OPENAI = True
        print("[--no-openai] forcing local-only routing", file=sys.stderr, flush=True)
    if "--no-cache" in flags:
        os.environ["NO_CACHE"] = "1"
        print("[--no-cache] OpenAI result cache disabled (no reads or writes)",
              file=sys.stderr, flush=True)
    if "--vector-check" in flags:
        _VECTOR_CHECK = True
        print("[--vector-check] pdfplumber ground-truth check enabled on every grid table",
              file=sys.stderr, flush=True)
    if "--vector-correct" in flags:
        _VECTOR_CORRECT = True
        _VECTOR_CHECK = True
        print("[--vector-correct] mismatched cells will be patched from pdfplumber's reading",
              file=sys.stderr, flush=True)
    # --confidence-threshold=N must be parsed BEFORE --confidence-check is printed so the
    # threshold value in the log line is accurate.
    for a in flags:
        if a.startswith("--confidence-threshold="):
            try:
                _CONFIDENCE_THRESHOLD = float(a.split("=", 1)[1])
            except ValueError:
                print(f"[--confidence-threshold] ignored, not a float: {a!r}",
                      file=sys.stderr, flush=True)
    if "--confidence-check" in flags:
        _CONFIDENCE_CHECK = True
        print(f"[--confidence-check] logprobs-based scoring enabled "
              f"(threshold={_CONFIDENCE_THRESHOLD})",
              file=sys.stderr, flush=True)

    pdf_path = positional[0]
    if not Path(pdf_path).exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(extract_pdf(pdf_path), indent=2))
