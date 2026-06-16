"""Backend for table-anchored PDF viewer.

API surface — one POST, one GET:
  POST /api/pdf             -> upload a PDF, run extraction, and get the extracted JSON back in the same response. No id needed — 
  it's derived from the filename. `?progress=1` streams SSE progress lines instead (used by the viewer's upload drawer).
  GET  /api/pdf/{pdf_name}  -> the extracted JSON output for an already-uploaded PDF. In Swagger the name is a dropdown of what's 
  on disk. `?format=pdf` returns the raw PDF bytes instead (used by the viewer's renderer).

Plus the app itself:
  GET  /                   -> the single-page web app
  GET  /static/*           -> its assets
"""
import asyncio
import json as _json
import shutil
import sys
import webbrowser
from collections import deque
from pathlib import Path

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print(
        "Missing deps. Run:\n"
        "  /opt/miniconda3/bin/pip install fastapi uvicorn python-multipart",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent          # .../viewer/
PDF_DIR = ROOT.parent                            # .../pdfread/
STATIC_DIR = ROOT / "static"
TEXTRACT_PY = PDF_DIR / "textract.py"

app = FastAPI(title="Table-anchored PDF viewer")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache_dev_assets(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path == "/openapi.json" or path.startswith("/static") \
            or path.startswith("/api/pdf"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _clean_id(source_id: str) -> str:
    """Swagger users tend to paste ids wrapped in quotes; strip them (and stray whitespace)."""
    return source_id.strip().strip("'\"")


def _resolve_id(source_id: str) -> str:
    """Resolve loose user input to an uploaded PDF's stem.
    Accepts: exact stem, stem with .pdf suffix, case-insensitive match, or any substring that
    matches exactly one PDF ("30-40" -> "PHST001-101_Protocol_V3.0_15Oct2025-30-40").
    Raises 400 when ambiguous (listing the candidates) and 404 when nothing matches."""
    raw = _clean_id(source_id)
    if raw.lower().endswith(".pdf"):
        raw = raw[:-4]
    stems = [p.stem for p in sorted(PDF_DIR.glob("*.pdf"))]
    if raw in stems:
        return raw
    low = raw.lower()
    ci = [s for s in stems if s.lower() == low]
    if len(ci) == 1:
        return ci[0]
    sub = [s for s in stems if low in s.lower()]
    if len(sub) == 1:
        return sub[0]
    if len(sub) > 1:
        raise HTTPException(
            400, f"ambiguous name {source_id!r} — matches: {sub}. Be more specific.")
    raise HTTPException(
        404, f"unknown PDF {source_id!r} — upload it first via POST /api/pdf, "
             f"or pick from the dropdown in /docs")


@app.get("/", include_in_schema=False)   # the web app itself; not part of the API
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/pdf/{pdf_name}")
@app.head("/api/pdf/{pdf_name}", include_in_schema=False)  # viewer probes output-readiness via HEAD
def get_pdf(
    pdf_name: str,
    # Hidden from the docs: the API always returns JSON. The viewer's renderer passes
    # ?format=pdf internally to fetch the raw PDF bytes for display.
    format: str = Query("json", include_in_schema=False),
):
    """The extracted JSON for an uploaded PDF. Any unique fragment of the filename selects
    the PDF."""
    source_id = _resolve_id(pdf_name)
    pdf_path = PDF_DIR / f"{source_id}.pdf"
    if format == "pdf":
        # FileResponse adds Accept-Ranges and handles Range requests, which PDF.js uses for progressive loading.
        return FileResponse(pdf_path, media_type="application/pdf")
    json_path = PDF_DIR / f"{source_id}_out.json"
    if not (json_path.is_file() and json_path.stat().st_size > 0):
        raise HTTPException(
            404, f"no output for {source_id!r} yet — POST the PDF to /api/pdf to run extraction")
    return FileResponse(json_path, media_type="application/json")


def _sse(event: str | None, data: str) -> bytes:
    """Encode one SSE message. `event` may be None for the default 'message' event."""
    parts = []
    if event:
        parts.append(f"event: {event}")
    for line in data.split("\n"):
        parts.append(f"data: {line}")
    parts.append("")   # blank line terminates the message
    return ("\n".join(parts) + "\n").encode("utf-8")


async def _stream_extract(source_id: str):
    """Spawn `textract.py --vector-correct --confidence-check <pdf>` and yield SSE-encoded stderr lines as they arrive. Sends a 
    final `done` event with the subprocess exit code so the client knows when the output JSON is ready."""
    pdf_path = PDF_DIR / f"{source_id}.pdf"
    json_path = PDF_DIR / f"{source_id}_out.json"
    if not pdf_path.exists():
        yield _sse("error", f"PDF not found: {pdf_path.name}")
        return
    if not TEXTRACT_PY.exists():
        yield _sse("error", f"textract.py not found at {TEXTRACT_PY}")
        return

    yield _sse(None, f"[server] starting textract on {pdf_path.name}")
    # stdout goes straight to the *_out.json file so we don't have to buffer the result in memory.
    out_file = open(json_path, "wb")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(TEXTRACT_PY),
            "--vector-correct", "--confidence-check",
            str(pdf_path),
            cwd=str(PDF_DIR),
            stdout=out_file,
            stderr=asyncio.subprocess.PIPE,
        )
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            yield _sse(None, line.decode(errors="replace").rstrip())
        code = await proc.wait()
    finally:
        out_file.close()

    payload = _json.dumps({"code": code, "id": source_id})
    yield _sse("done", payload)


async def _run_extract(source_id: str):
    """Run textract to completion (no streaming). Returns (exit_code, last_log_lines)."""
    pdf_path = PDF_DIR / f"{source_id}.pdf"
    json_path = PDF_DIR / f"{source_id}_out.json"
    tail = deque(maxlen=15)
    out_file = open(json_path, "wb")
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(TEXTRACT_PY),
            "--vector-correct", "--confidence-check",
            str(pdf_path),
            cwd=str(PDF_DIR),
            stdout=out_file,
            stderr=asyncio.subprocess.PIPE,
        )
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            tail.append(line.decode(errors="replace").rstrip())
        code = await proc.wait()
    finally:
        out_file.close()
    return code, list(tail)


@app.post("/api/pdf")
async def upload_and_extract(file: UploadFile = File(...), progress: bool = False):
    """Upload a PDF, run extraction, and get the extracted JSON back — all in one call.
    The id is derived from the filename; you never need to supply one. The request blocks
    until extraction finishes (minutes on a large PDF), then the response body IS the JSON.

    With ?progress=1 the response instead streams textract's progress as Server-Sent Events,
    ending with a `done` event — that mode is what the viewer's upload drawer uses."""
    name = (file.filename or "").strip()
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    # Prevent path traversal / sneaky filenames by keeping just the basename.
    safe_name = Path(name).name
    dest = PDF_DIR / safe_name
    # Overwrites if a PDF with the same name exists.
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if progress:
        return StreamingResponse(
            _stream_extract(dest.stem),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    code, tail = await _run_extract(dest.stem)
    json_path = PDF_DIR / f"{dest.stem}_out.json"
    if code != 0 or not (json_path.is_file() and json_path.stat().st_size > 0):
        raise HTTPException(
            500, f"extraction failed (exit code {code}); last log lines: {tail}")
    return FileResponse(json_path, media_type="application/json")


# Swagger dropdown for the source id.
# Regenerate the OpenAPI schema on every /openapi.json request and inject the PDFs currently on disk as an enum for 
# GET /api/pdf/{source_id}. Swagger UI renders enums as a <select> dropdown, so the id is picked from a list instead of typed. 
def _openapi_with_pdf_dropdown():
    schema = get_openapi(title=app.title, version="1.0.0", routes=app.routes)
    stems = [p.stem for p in sorted(PDF_DIR.glob("*.pdf"))]
    try:
        for param in schema["paths"]["/api/pdf/{pdf_name}"]["get"]["parameters"]:
            if param["name"] == "pdf_name":
                param["schema"]["enum"] = stems
                param["description"] = "Pick an uploaded PDF (refresh the page to see new uploads)"
    except KeyError:
        pass
    return schema


app.openapi = _openapi_with_pdf_dropdown


if __name__ == "__main__":
    port = 8765
    print(f"\nServing on http://localhost:{port}\n  PDF dir: {PDF_DIR}\n", file=sys.stderr)
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
