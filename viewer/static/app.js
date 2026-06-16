/* Table-anchored PDF viewer
 *   Left pane:  PDF.js renders pages as you scroll. 
 *   Right pane: tree of extracted tables. Click a card -> scroll PDF to that page + box highlight.
 */
(() => {
const pdfjsLib = window["pdfjs-dist/build/pdf"];
pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

const state = {
    sources: [],
    sourceId: null,
    pdfDoc: null,
    data: null,
    pages: [],           // [{pageNumber, container, canvas, overlay, viewport, pdfHeight, page, rendered, placeholder}]
    scale: 1.25,
    observer: null,      // IntersectionObserver, recreated on each zoom
};

const $picker = document.getElementById("source-picker");
const $status = document.getElementById("status");
const $viewer = document.getElementById("pdf-viewer");
const $pdfPane = document.getElementById("pdf-pane");
const $tree   = document.getElementById("tree");
const $tip    = document.getElementById("tooltip");
const $zoomIn = document.getElementById("zoom-in");
const $zoomOut = document.getElementById("zoom-out");
const $zoomReadout = document.getElementById("zoom-readout");
const $ctx = document.getElementById("context-popup");
const $ctxTitle = $ctx.querySelector(".ctx-title");
const $ctxMeta  = $ctx.querySelector(".ctx-meta");
const $ctxBody  = $ctx.querySelector(".ctx-body");
const $ctxMin = $ctx.querySelector(".ctx-min");
// Minimize / restore toggle. Collapses the popup to just its title row so it stays in view without blocking content. Re-click
// expands again. The "–" / "+" mirrors the state.
$ctxMin.addEventListener("click", (e) => {
    e.stopPropagation();
    const min = $ctx.classList.toggle("minimized");
    $ctxMin.textContent = min ? "+" : "–";
    $ctxMin.setAttribute("aria-label", min ? "Restore context panel" : "Minimize context panel");
    $ctxMin.setAttribute("title", min ? "Restore" : "Minimize");
});


/* Upload + extract — drives the server's /api/upload + /api/extract endpoints so the user can drop in a fresh PDF without leaving
 the browser. EventSource streams the textract stderr line-by-line into the drawer; on `done` the dropdown refreshes and switches to the new file. */
const $uploadBtn   = document.getElementById("upload-btn");
const $uploadInput = document.getElementById("upload-input");
const $exDrawer    = document.getElementById("extract-drawer");
const $exFilename  = $exDrawer.querySelector(".ex-filename");
const $exLog       = $exDrawer.querySelector(".ex-log");
const $exStatus    = $exDrawer.querySelector(".ex-status");
const $exClose     = $exDrawer.querySelector(".ex-close");

$uploadBtn.addEventListener("click", () => $uploadInput.click());
$uploadInput.addEventListener("change", async () => {
    const file = $uploadInput.files[0];
    if (!file) return;
    $uploadInput.value = "";   // so picking the same file again re-fires `change`
    await uploadAndExtract(file);
});
// Minimize / restore toggle — mirrors the context popup. Collapses the drawer to just its title bar so the log doesn't block the 
// PDF; click again to restore.
$exClose.addEventListener("click", () => {
    const min = $exDrawer.classList.toggle("minimized");
    $exClose.textContent = min ? "+" : "−";
    $exClose.setAttribute("aria-label", min ? "Restore" : "Minimize");
    $exClose.setAttribute("title",      min ? "Restore" : "Minimize");
});

async function uploadAndExtract(file) {
    openDrawer(file.name);
    setExStatus("uploading + extracting… this can take several minutes on a large PDF");
    const fd = new FormData();
    fd.append("file", file);
    let resp;
    try {
        // Single POST: with ?progress=1 the response body IS the server sent progress stream. (EventSource can't do POST, so we 
        // parse the stream by hand with a reader.)
        resp = await fetch("/api/pdf?progress=1", { method: "POST", body: fd });
    } catch (e) {
        setExStatus("upload failed: " + e.message, "error");
        return;
    }
    if (!resp.ok || !resp.body) {
        const text = await resp.text().catch(() => "");
        setExStatus(`upload failed (${resp.status}): ${text.slice(0, 200)}`, "error");
        return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        // Server sent messages are separated by a blank line.
        let sep;
        while ((sep = buf.indexOf("\n\n")) >= 0) {
            const rawEvent = buf.slice(0, sep);
            buf = buf.slice(sep + 2);
            await handleExtractEvent(rawEvent);
        }
    }
}

async function handleExtractEvent(raw) {
    let event = "message";
    const dataLines = [];
    for (const line of raw.split("\n")) {
        if (line.startsWith("event:"))     event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
    }
    const data = dataLines.join("\n");
    if (event === "error") {
        appendExLog(data);
        setExStatus(data, "error");
    } else if (event === "done") {
        let payload;
        try { payload = JSON.parse(data); } catch { payload = { code: -1 }; }
        if (payload.code === 0) {
            setExStatus("done. switching to the new source…", "ok");
            await refreshSources(payload.id);
            setExStatus("done", "ok");
        } else {
            setExStatus(`textract exited with code ${payload.code} — check the log above`, "error");
        }
    } else {
        appendExLog(data);
    }
}

async function refreshSources(autoSelectId) {
    const sources = await fetchSources();
    state.sources = sources;
    $picker.innerHTML = sources.map(s =>
        `<option value="${s.id}">${s.label} — ${s.pdf}</option>`
    ).join("");
    if (autoSelectId && sources.some(s => s.id === autoSelectId)) {
        $picker.value = autoSelectId;
        await loadSource(autoSelectId);
    }
}

function openDrawer(filename) {
    $exDrawer.hidden = false;
    // Restore from any prior minimized state so the new extraction is visible immediately.
    $exDrawer.classList.remove("minimized");
    $exClose.textContent = "−";
    $exClose.setAttribute("aria-label", "Minimize");
    $exClose.setAttribute("title", "Minimize");
    $exFilename.textContent = filename;
    $exLog.textContent = "";
    setExStatus("");
}
function appendExLog(text) {
    const atBottom = $exLog.scrollTop + $exLog.clientHeight >= $exLog.scrollHeight - 4;
    $exLog.textContent += text + "\n";
    // Only auto-scroll if the user was already at the bottom — lets them scroll up to read earlier lines without being yanked back down.
    if (atBottom) $exLog.scrollTop = $exLog.scrollHeight;
}
function setExStatus(text, kind) {
    $exStatus.textContent = text;
    $exStatus.className = "ex-status" + (kind ? " ex-status--" + kind : "");
}


/** API is just POST /api/pdf + GET /api/pdf/{pdf_name}), so the available PDFs come from the OpenAPI schema — the server injects 
 * them as the pdf_name enum on every /openapi.json fetch. A HEAD probe per PDF tells us which ones have extracted output (200) vs not yet (404). */
async function fetchSources() {
    let stems = [];
    try {
        const spec = await fetch("/openapi.json").then(r => r.json());
        const params = spec.paths["/api/pdf/{pdf_name}"].get.parameters;
        stems = params.find(p => p.name === "pdf_name").schema.enum || [];
    } catch (e) {
        console.warn("could not read PDF list from /openapi.json:", e);
        return [];
    }
    const checks = await Promise.all(stems.map(s =>
        fetch(`/api/pdf/${encodeURIComponent(s)}`, { method: "HEAD" })
            .then(r => r.ok).catch(() => false)
    ));
    return stems
        .map((s, i) => ({ id: s, pdf: s + ".pdf", label: s, has_output: checks[i] }))
        .filter(s => s.has_output);
}


async function bootstrap() {
    const sources = await fetchSources();
    state.sources = sources;
    if (!sources.length) {
        $status.textContent = "No PDF+JSON pairs found in the parent directory.";
        return;
    }
    $picker.innerHTML = sources.map(s =>
        `<option value="${s.id}">${s.label} — ${s.pdf}</option>`
    ).join("");
    $picker.addEventListener("change", () => loadSource($picker.value));
    $zoomIn.addEventListener("click", () => setScale(state.scale * 1.15));
    $zoomOut.addEventListener("click", () => setScale(state.scale / 1.15));
    await loadSource(sources[0].id);
}


async function loadSource(id) {
    state.sourceId = id;
    $status.textContent = "loading…";
    teardownPages();
    $tree.innerHTML = "";
    $ctx.hidden = true;   // stale context from previous source

    const [data, pdfBytes] = await Promise.all([
        fetch(`/api/pdf/${id}`).then(r => r.json()),
        fetch(`/api/pdf/${id}?format=pdf`).then(r => r.arrayBuffer()),
    ]);
    state.data = data;
    state.pdfDoc = await pdfjsLib.getDocument({data: pdfBytes}).promise;
    $status.textContent = `${state.pdfDoc.numPages} pages · ${data.tables.length} tables`;

    await buildPagePlaceholders();
    setupLazyRender();
    renderTree();
}


/* PDF rendering */

function teardownPages() {
    if (state.observer) {
        state.observer.disconnect();
        state.observer = null;
    }
    $viewer.innerHTML = "";
    state.pages = [];
}


/** Create a correctly-sized placeholder for every page. This gives the scroll container the right total height up front so the 
 * IntersectionObserver works correctly and scroll-to-page is accurate even before any canvas exists. */
async function buildPagePlaceholders() {
    for (let i = 1; i <= state.pdfDoc.numPages; i++) {
        const page = await state.pdfDoc.getPage(i);
        const viewport = page.getViewport({scale: state.scale});

        const container = document.createElement("div");
        container.className = "pdf-page is-placeholder";
        container.dataset.pageNumber = i;
        container.style.width  = viewport.width  + "px";
        container.style.height = viewport.height + "px";

        // Skeleton label while the canvas hasn't rendered yet.
        const placeholder = document.createElement("div");
        placeholder.className = "page-placeholder";
        placeholder.textContent = `Page ${i}`;
        container.appendChild(placeholder);

        const overlay = document.createElement("div");
        overlay.className = "overlay";
        container.appendChild(overlay);

        $viewer.appendChild(container);

        state.pages.push({
            pageNumber: i, container, overlay, viewport, placeholder, page,
            canvas: null, rendered: false,
            pdfHeight: viewport.viewBox[3] - viewport.viewBox[1],
        });
    }
    $zoomReadout.textContent = Math.round(state.scale * 100) + "%";
}


/** Set up the IntersectionObserver that renders pages on demand. rootMargin of 1 viewport height means a page starts rendering 
 * before it scrolls into view. */
function setupLazyRender() {
    const obs = new IntersectionObserver(entries => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            const i = parseInt(entry.target.dataset.pageNumber, 10);
            const pe = state.pages[i - 1];
            if (pe && !pe.rendered) renderPage(pe);
        }
    }, {
        root: $pdfPane,
        rootMargin: "1000px 0px",   // start render ~1 viewport ahead of scroll
        threshold: 0,
    });
    for (const pe of state.pages) obs.observe(pe.container);
    state.observer = obs;
}


async function renderPage(pe) {
    if (pe.rendered) return;
    pe.rendered = true;   // mark first to avoid a double-render race if observer fires twice
    try {
        const canvas = document.createElement("canvas");
        canvas.width  = pe.viewport.width;
        canvas.height = pe.viewport.height;
        // Insert canvas BEFORE the overlay so highlights still float on top.
        pe.container.insertBefore(canvas, pe.overlay);
        pe.canvas = canvas;
        await pe.page.render({
            canvasContext: canvas.getContext("2d"),
            viewport: pe.viewport,
        }).promise;
        pe.placeholder.remove();
        pe.container.classList.remove("is-placeholder");
        // Make the page's link annotations clickable (TOC entries, "see Table N" cross-refs, external URLs).
        renderLinkLayer(pe).catch(err =>
            console.warn(`page ${pe.pageNumber} link layer failed:`, err));
    } catch (err) {
        // If render fails (rare), leave placeholder visible and reset the flag so a future intersection can retry.
        pe.rendered = false;
        console.warn(`page ${pe.pageNumber} render failed:`, err);
    }
}


/** Overlay one positioned <a> per link annotation on the page. The overlay div has
 * pointer-events:none (so it doesn't block text/canvas interaction); each link re-enables
 * pointer-events on itself. Internal destinations scroll the lazy-rendered page list;
 * external URLs open a new tab. */
async function renderLinkLayer(pe) {
    const annots = await pe.page.getAnnotations();
    for (const a of annots) {
        if (a.subtype !== "Link" || !a.rect) continue;
        // a.rect is [x1, y1, x2, y2] in PDF user space (y-up); the viewport transform
        // handles the flip + zoom scale for us.
        const r = pe.viewport.convertToViewportRectangle(a.rect);
        const el = document.createElement("a");
        el.className = "pdf-link";
        el.style.left   = Math.min(r[0], r[2]) + "px";
        el.style.top    = Math.min(r[1], r[3]) + "px";
        el.style.width  = Math.abs(r[2] - r[0]) + "px";
        el.style.height = Math.abs(r[3] - r[1]) + "px";
        if (a.url) {
            el.href = a.url;
            el.target = "_blank";
            el.rel = "noopener noreferrer";
            el.title = a.url;
        } else if (a.dest) {
            el.href = "#";
            el.title = "Jump to linked section";
            el.addEventListener("click", (e) => {
                e.preventDefault();
                goToDest(a.dest);
            });
        } else {
            continue;
        }
        pe.overlay.appendChild(el);
    }
}


/** Resolve a PDF internal destination (named string or explicit array) to a page and
 * scroll to it. The IntersectionObserver lazy-renders the target page on arrival. */
async function goToDest(dest) {
    try {
        const d = typeof dest === "string"
            ? await state.pdfDoc.getDestination(dest)
            : dest;
        if (!d || !d[0]) return;
        const pageIndex = await state.pdfDoc.getPageIndex(d[0]);
        const pe = state.pages[pageIndex];
        if (pe) pe.container.scrollIntoView({behavior: "smooth", block: "start"});
    } catch (err) {
        console.warn("could not resolve PDF link destination:", err);
    }
}


async function setScale(newScale) {
    state.scale = Math.max(0.5, Math.min(3.0, newScale));
    teardownPages();
    await buildPagePlaceholders();
    setupLazyRender();
}


/* Anchor a PDF page when a table is clicked */

function scrollToTable(t) {
    const pe = state.pages.find(p => p.pageNumber === t.page);
    if (!pe) return;
    pe.container.scrollIntoView({behavior: "smooth", block: "start"});
    // The observer will fire on scroll-end and render the page if it isn't yet.
    // Failed tables get the red highlight variant so the box itself signals "needs review".
    drawHighlight(pe, t._bbox, t._failed ? "failed" : "");
}


function drawHighlight(pe, bbox, variant) {
    pe.overlay.querySelectorAll(".hl-rect").forEach(el => el.remove());
    if (!bbox) return;
    // Docling bbox is in PDF points with y-up origin. Convert to top-left CSS coords:
    // pageH = y1 - y0; if bbox.top > bbox.bottom, source is y-up — invert.
    const pageH = pe.pdfHeight;
    const scale = pe.viewport.scale;
    const yLo = Math.min(bbox.top, bbox.bottom);
    const yHi = Math.max(bbox.top, bbox.bottom);
    const yUp = bbox.top > bbox.bottom;
    const yPageTop = yUp ? pageH - yHi : yLo;
    const rect = document.createElement("div");
    rect.className = "hl-rect" + (variant ? " hl-rect--" + variant : "");
    rect.style.left   = (bbox.left * scale) + "px";
    rect.style.top    = (yPageTop * scale) + "px";
    rect.style.width  = ((bbox.right - bbox.left) * scale) + "px";
    rect.style.height = ((yHi - yLo) * scale) + "px";
    pe.overlay.appendChild(rect);
    // Force-restart the pulse animation on repeat clicks.
    rect.style.animation = "none"; void rect.offsetWidth; rect.style.animation = "";
}


/* Tree pane */

// Source-type glyphs for the backend pill (lucide outlines, stroke = currentColor).
const ICON_CLOUD =
    '<svg class="backend-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/></svg>';
const ICON_DRIVE =
    '<svg class="backend-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<line x1="22" x2="2" y1="12" y2="12"/>' +
    '<path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>' +
    '<line x1="6" x2="6.01" y1="16" y2="16"/><line x1="10" x2="10.01" y1="16" y2="16"/></svg>';

function renderTree() {
    $tree.innerHTML = "";
    state.data.tables.forEach((t, idx) => {
        const card = document.createElement("div");
        card.className = "table-card";
        if (t._failed) card.classList.add("failed");
        card.dataset.idx = idx;

        const head = document.createElement("div");
        head.className = "head";

        const twisty = chevronEl();   

        const title = document.createElement("span");
        title.className = "title";
        if (t.title) {
            title.textContent = t.title;
        } else {
            title.textContent = `(table ${idx + 1})`;
            title.classList.add("fallback");
        }
        const backend = document.createElement("span");
        if (t._failed) {
            // Failed extraction (timeout / parse error): red FAILED pill replaces the
            // backend pill — there's no backend result to attribute.
            backend.className = "fail-badge";
            backend.textContent = "FAILED";
            backend.title = `Extraction failed: ${t._failed.reason} — flagged for human review`;
        } else {
            // Neutral pill + icon: the cloud/hard-drive glyph carries the source-type signal
            // instead of a loud background color.
            const backendKind = (t._backend || "").startsWith("cloud") ? "cloud" : "local";
            backend.className = `backend ${backendKind}`;
            backend.innerHTML =
                (backendKind === "cloud" ? ICON_CLOUD : ICON_DRIVE) + backendKind;
        }
        const page = document.createElement("span");
        page.className = "page-badge";
        page.textContent = `p.${t.page ?? "?"}`;

        head.append(twisty, title, backend);
        // Human-in-the-loop: tables where the LLM scored below the confidence threshold get a red REVIEW pill so the human 
        // reviewer knows where to focus. Hover for the score.
        if (t._confidence && t._confidence.needs_review) {
            const review = document.createElement("span");
            review.className = "review-badge";
            review.textContent = "REVIEW";
            const score = (t._confidence.score * 100).toFixed(1);
            const thresh = (t._confidence.threshold * 100).toFixed(0);
            review.title = `Mean token confidence ${score}% < threshold ${thresh}%`;
            head.appendChild(review);
        }
        if (t._kind === "toc") {
            const kind = document.createElement("span");
            kind.className = "kind-badge";
            kind.textContent = "TOC";
            head.appendChild(kind);
        }
        head.append(page);   

        // Popup only opens on an EXPLICIT click on the card's header row.
        head.addEventListener("click", () => {
            card.classList.toggle("open");
            scrollToTable(t);
            showContextPopup(t, idx);
        });

        const body = document.createElement("div");
        body.className = "body";

        if (t._failed) {
            const note = document.createElement("div");
            note.className = "fail-note";
            note.textContent =
                `Extraction failed (${t._failed.reason}). ` +
                `No data was captured for this table — review it manually in the PDF. ` +
                `Click the card header to jump to its location.`;
            body.appendChild(note);
        } else if (t.data !== undefined) {
            const label = document.createElement("div");
            label.className = "section-title"; label.textContent = "data";
            body.append(label, renderJson(t.data, t));
        }

        // Footnotes are available in the bottom-left context popup on click. 
        if (Array.isArray(t.discussions) && t.discussions.length) {
            const label = document.createElement("div");
            label.className = "section-title"; label.textContent = "discussed in";
            body.appendChild(label);
            t.discussions.forEach(d => {
                const div = document.createElement("div");
                div.className = "discussion";
                const preview = d.text.length > 110 ? d.text.slice(0, 110) + "…" : d.text;
                div.innerHTML = `<span class="page-tag">p.${d.page}</span>${escapeHtml(preview)}`;
                div.addEventListener("mouseenter", e => showTip(e, d.text));
                div.addEventListener("mouseleave", hideTip);
                div.addEventListener("click", e => {
                    e.stopPropagation();
                    const pe = state.pages.find(p => p.pageNumber === d.page);
                    if (pe) pe.container.scrollIntoView({behavior: "smooth", block: "start"});
                });
                body.appendChild(div);
            });
        }

        card.append(head, body);
        $tree.appendChild(card);
    });
}


function renderJson(node, table, depth = 0) {
    const ul = document.createElement("ul");
    ul.className = "json";

    const addLeaf = (key, val) => {
        const li = document.createElement("li");
        const k = key === null ? "" : `<span class="key">${escapeHtml(key)}</span>: `;
        let v;
        if (val === null) v = `<span class="nil">null</span>`;
        else if (typeof val === "string") v = `<span class="str">"${escapeHtml(val)}"</span>`;
        else if (typeof val === "number") v = `<span class="num">${val}</span>`;
        else if (typeof val === "boolean") v = `<span class="bool">${val}</span>`;
        else v = `<span>${escapeHtml(String(val))}</span>`;
        li.innerHTML = k + v;
        li.style.cursor = "pointer";
        li.addEventListener("click", e => {
            e.stopPropagation();
            scrollToTable(table);
        });
        ul.appendChild(li);
    };

    const addContainer = (key, val) => {
        const li = document.createElement("li");
        const isArr = Array.isArray(val);
        const label = key === null ? "" : `<span class="key">${escapeHtml(key)}</span>: `;
        const bracket = isArr ? `[${val.length}]` : `{${Object.keys(val).length}}`;
        li.innerHTML = `<span class="collapse"></span>${label}<span class="bracket">${bracket}</span>`;
        const child = renderJson(val, table, depth + 1);
        li.appendChild(child);
        li.querySelector(".collapse").addEventListener("click", e => {
            e.stopPropagation();
            li.classList.toggle("collapsed");
        });
        if (depth >= 2) li.classList.add("collapsed");
        ul.appendChild(li);
    };

    if (Array.isArray(node)) {
        node.forEach((v, i) => {
            if (v !== null && typeof v === "object") addContainer(i, v);
            else addLeaf(i, v);
        });
    } else if (node !== null && typeof node === "object") {
        Object.entries(node).forEach(([k, v]) => {
            if (v !== null && typeof v === "object") addContainer(k, v);
            else addLeaf(k, v);
        });
    } else {
        addLeaf(null, node);
    }
    return ul;
}


function showTip(evt, text) {
    $tip.textContent = text;
    $tip.hidden = false;
    const pad = 12;
    const x = Math.min(evt.clientX + pad, window.innerWidth - 380);
    const y = Math.min(evt.clientY + pad, window.innerHeight - 200);
    $tip.style.left = x + "px";
    $tip.style.top  = y + "px";
}
function hideTip() { $tip.hidden = true; }


/* Context popup shows in the bottom-left corner whenever a table card is clicked. Sourced from the JSON.*/
function showContextPopup(t, fallbackIdx) {
    if (t.title) {
        $ctxTitle.textContent = t.title;
        $ctxTitle.classList.remove("fallback");
    } else {
        $ctxTitle.textContent = `(table ${fallbackIdx + 1})`;
        $ctxTitle.classList.add("fallback");
    }

    // Meta row: page + rough size + presence counts
    const rows = Array.isArray(t.data) ? t.data.length
               : (t.data && typeof t.data === "object") ? Object.keys(t.data).length
               : 0;
    const fnCount = t.footnotes ? Object.keys(t.footnotes).length : 0;
    const dCount  = Array.isArray(t.discussions) ? t.discussions.length : 0;
    const parts = [
        `Page ${t.page ?? "?"}`,
        `${rows} ${rows === 1 ? "row" : "rows"}`,
    ];
    if (fnCount) parts.push(`${fnCount} footnote${fnCount === 1 ? "" : "s"}`);
    if (dCount)  parts.push(`${dCount} reference${dCount === 1 ? "" : "s"}`);
    $ctxMeta.textContent = parts.join(" · ");
    $ctxBody.innerHTML = "";

    // Failed extraction: lead with the red review note so the human knows why there's no data.
    if (t._failed) {
        const fail = document.createElement("div");
        fail.className = "ctx-failed";
        fail.textContent =
            `Extraction failed: ${t._failed.reason}. ` +
            `This table needs manual review — the red box on the page marks its location.`;
        $ctxBody.appendChild(fail);
    }

    if (dCount > 0) {
        const heading = document.createElement("div");
        heading.className = "ctx-section-title";
        heading.textContent = "Narrative context";
        $ctxBody.appendChild(heading);
        const lead = document.createElement("p");
        lead.className = "ctx-paragraph";
        lead.textContent = t.discussions[0].text;
        $ctxBody.appendChild(lead);

        if (dCount > 1) {
            const more = document.createElement("div");
            more.className = "ctx-more";
            more.textContent = `+ ${dCount - 1} more reference${dCount - 1 === 1 ? "" : "s"} in the right pane`;
            $ctxBody.appendChild(more);
        }
    }

    if (fnCount > 0) {
        const heading = document.createElement("div");
        heading.className = "ctx-section-title";
        heading.textContent = "Footnotes";
        $ctxBody.appendChild(heading);
        for (const [marker, text] of Object.entries(t.footnotes)) {
            const div = document.createElement("div");
            div.className = "ctx-footnote";
            const m = document.createElement("span");
            m.className = "marker"; m.textContent = marker;
            div.appendChild(m);
            div.appendChild(document.createTextNode(text));
            $ctxBody.appendChild(div);
        }
    }

    // Human-in-the-loop: surface the model's self-reported confidence so the reviewer can see why the table was flagged, plus the
    //  eight least-confident tokens.
    if (t._confidence) {
        const c = t._confidence;
        const heading = document.createElement("div");
        heading.className = "ctx-section-title";
        heading.textContent = "Model confidence";
        $ctxBody.appendChild(heading);

        const score = document.createElement("div");
        score.className = "ctx-confidence" + (c.needs_review ? " ctx-confidence--review" : " ctx-confidence--ok");
        const pct = (c.score * 100).toFixed(1);
        const threshPct = c.threshold ? (c.threshold * 100).toFixed(0) : "—";
        score.innerHTML = `
            <div class="conf-row">
                <span class="conf-label">Score</span>
                <span class="conf-value">${pct}%</span>
            </div>
            <div class="conf-row">
                <span class="conf-label">Threshold</span>
                <span class="conf-value">${threshPct}%</span>
            </div>
            <div class="conf-row">
                <span class="conf-label">Verdict</span>
                <span class="conf-value">${c.needs_review ? "needs human review" : "ok"}</span>
            </div>
        `;
        $ctxBody.appendChild(score);

        // worst_tokens is still computed and stored in _confidence for audit / debugging,
        // but not rendered here -- the raw token fragments (mostly JSON punctuation) didn't
        // give the reviewer anything actionable.
    }

    if (dCount === 0 && fnCount === 0 && !t._confidence && !t._failed) {
        const empty = document.createElement("div");
        empty.className = "ctx-empty";
        empty.textContent = "No narrative references or footnotes were extracted for this table.";
        $ctxBody.appendChild(empty);
    }

    // Always expand when showing a new table — previous minimized state from a different table would be misleading now that the
    // title bar shows different content.
    $ctx.classList.remove("minimized");
    $ctxMin.textContent = "–";
    $ctxMin.setAttribute("aria-label", "Minimize context panel");
    $ctx.hidden = false;
}


function escapeHtml(s) {
    return String(s)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function chevronEl() {
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("class", "chevron");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "none");
    svg.setAttribute("stroke", "currentColor");
    svg.setAttribute("stroke-width", "2.5");
    svg.setAttribute("stroke-linecap", "round");
    svg.setAttribute("stroke-linejoin", "round");
    svg.setAttribute("aria-hidden", "true");
    const poly = document.createElementNS(svgNS, "polyline");
    poly.setAttribute("points", "9 6 15 12 9 18");
    svg.appendChild(poly);
    return svg;
}


bootstrap().catch(err => {
    console.error(err);
    $status.textContent = "Error: " + err.message;
});
})();
