# rag/build_kb_clean.py
# One-step KB builder:
# PDFs -> slide extraction -> cleanup -> chunks -> data/kb/*
#
# Run:
#   python rag/build_kb_clean.py
#
# Requirements:
#   pip install pymupdf

import os
import re
import json
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple

import fitz  # PyMuPDF


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def list_pdfs(pdfs_dir: str) -> List[str]:
    pdfs = []
    for root, _, files in os.walk(pdfs_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    return sorted(pdfs)


def safe_id(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_")


def infer_lecture_id(pdf_path: str) -> str:
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    return safe_id(stem)


def norm_ws(s: str) -> str:
    s = (s or "").replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def is_page_counter(line: str) -> bool:
    t = (line or "").strip()
    return bool(re.fullmatch(r"\d{1,3}(\s*/\s*\d{1,3})?", t))


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_text_for_embedding(lecture_id: str, title: str, body: str) -> str:
    parts = []
    if lecture_id:
        parts.append(f"Lecture: {lecture_id}")
    if title:
        parts.append(f"Title: {title}")
    if body:
        parts.append(body)
    return norm_ws("\n".join(parts))


@dataclass
class Span:
    text: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float


def extract_spans(page: fitz.Page) -> List[Span]:
    d = page.get_text("dict")
    spans: List[Span] = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for sp in line.get("spans", []):
                t = (sp.get("text") or "").strip()
                if not t:
                    continue
                bbox = sp.get("bbox", [0, 0, 0, 0])
                spans.append(
                    Span(
                        text=t,
                        size=float(sp.get("size", 0.0)),
                        x0=float(bbox[0]),
                        y0=float(bbox[1]),
                        x1=float(bbox[2]),
                        y1=float(bbox[3]),
                    )
                )
    return spans


def parse_slide_structured(page: fitz.Page) -> Tuple[str, str]:
    spans = extract_spans(page)
    if not spans:
        return ("", "")

    rect = page.rect
    top_band_y = rect.y0 + 0.25 * rect.height

    top_spans = [s for s in spans if s.y0 <= top_band_y]
    title = ""
    max_size = None
    if top_spans:
        max_size = max(s.size for s in top_spans)
        title_spans = [s for s in top_spans if s.size >= max_size - 1.0]
        title_spans.sort(key=lambda s: (s.y0, s.x0))
        title = norm_ws(" ".join(s.text for s in title_spans))

    buckets: Dict[int, List[Span]] = {}
    for s in spans:
        if max_size is not None and s.y0 <= top_band_y and s.size >= max_size - 1.0:
            continue
        key = int(s.y0 // 6)
        buckets.setdefault(key, []).append(s)

    lines: List[str] = []
    for k in sorted(buckets.keys()):
        row = buckets[k]
        row.sort(key=lambda sp: sp.x0)
        line = norm_ws(" ".join(sp.text for sp in row))
        if line:
            lines.append(line)

    lines = [ln for ln in lines if not is_page_counter(ln)]
    body = norm_ws("\n".join(lines))
    return (title, body)


def build_boilerplate_sets(slides: List[Dict[str, Any]], df_ratio: float = 0.20) -> Dict[str, set]:
    by_lec = defaultdict(list)
    for s in slides:
        by_lec[s["lecture_id"]].append(s)

    boiler: Dict[str, set] = {}
    for lec, lec_slides in by_lec.items():
        df = Counter()
        n = len(lec_slides)

        for s in lec_slides:
            title = (s.get("title") or "").strip()
            seen = set(split_lines(s.get("raw_text", "")))
            if title:
                seen.add(title)
            for ln in seen:
                if is_page_counter(ln):
                    df[ln] += n
                else:
                    df[ln] += 1

        cutoff = max(2, int(n * df_ratio))
        boiler[lec] = {ln for ln, c in df.items() if c >= cutoff or is_page_counter(ln)}
    return boiler


def clean_slide(slide: Dict[str, Any], boiler_lines: set) -> Tuple[str, str]:
    title = (slide.get("title") or "").strip()
    raw_lines = split_lines(slide.get("raw_text", ""))

    kept = []
    for ln in raw_lines:
        if is_page_counter(ln):
            continue
        if ln in boiler_lines:
            continue
        kept.append(ln)

    if not title or title in boiler_lines or is_page_counter(title):
        title = kept[0] if kept else ""

    if kept and title and kept[0] == title:
        body_lines = kept[1:]
    else:
        body_lines = kept

    return (norm_ws(title), norm_ws("\n".join(body_lines)))


# ---- renamed “B1” → slide_level ----
def chunks_slide_level(slide: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{
        "chunk_id": slide["slide_id"],
        "chunk_type": "slide_level",
        "lecture_id": slide["lecture_id"],
        "source_pdf": slide["source_pdf"],
        "slide_number": slide["slide_number"],
        "slide_id": slide["slide_id"],
        "title": slide.get("title", ""),
        "chunk_part": 1,
        "text": slide["text_for_embedding"],
    }]


def chunks_split(slide: Dict[str, Any], max_tokens: int = 350) -> List[Dict[str, Any]]:
    text = slide["text_for_embedding"]
    segs = [s.strip() for s in re.split(r"\n\s*\n", text) if s.strip()]
    if len(segs) <= 1:
        segs = [s.strip() for s in text.split("\n") if s.strip()]

    out: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_t = 0
    part = 1

    def flush():
        nonlocal part, buf, buf_t
        if buf:
            out.append({
                "chunk_id": f"{slide['slide_id']}_p{part}",
                "chunk_type": "passage_level",
                "lecture_id": slide["lecture_id"],
                "source_pdf": slide["source_pdf"],
                "slide_number": slide["slide_number"],
                "slide_id": slide["slide_id"],
                "title": slide.get("title",""),
                "chunk_part": part,
                "text": norm_ws("\n".join(buf)),
            })
            part += 1
            buf, buf_t = [], 0

    for seg in segs:
        t = approx_tokens(seg)
        if t >= max_tokens:
            flush()
            out.append({
                "chunk_id": f"{slide['slide_id']}_p{part}",
                "chunk_type": "passage_level",
                "lecture_id": slide["lecture_id"],
                "source_pdf": slide["source_pdf"],
                "slide_number": slide["slide_number"],
                "slide_id": slide["slide_id"],
                "title": slide.get("title",""),
                "chunk_part": part,
                "text": norm_ws(seg),
            })
            part += 1
            continue

        if buf_t + t > max_tokens:
            flush()
        buf.append(seg)
        buf_t += t

    flush()
    return out


def process_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    lecture_id = infer_lecture_id(pdf_path)
    source_pdf = os.path.basename(pdf_path)

    doc = fitz.open(pdf_path)
    out: List[Dict[str, Any]] = []

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        slide_number = pno + 1
        slide_id = f"{lecture_id}_S{slide_number:03d}"

        title, body = parse_slide_structured(page)
        out.append({
            "slide_id": slide_id,
            "lecture_id": lecture_id,
            "slide_number": slide_number,
            "source_pdf": source_pdf,
            "title": title,
            "raw_text": norm_ws("\n".join([title, body]).strip()) if title else body,
        })

    doc.close()
    return out


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    pdfs_dir = os.path.join(root, "data", "pdfs")

    kb_dir = os.path.join(root, "data", "kb")
    slides_out = os.path.join(kb_dir, "slides_clean.jsonl")

    # renamed outputs
    slide_level_out = os.path.join(kb_dir, "chunks_slide_level.jsonl")
    passage_out = os.path.join(kb_dir, "chunks_passage_level.jsonl")

    debug_out = os.path.join(root, "data", "debug", "slides_extracted.jsonl")

    pdfs = list_pdfs(pdfs_dir)
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {pdfs_dir}")

    extracted: List[Dict[str, Any]] = []
    for pdf in pdfs:
        extracted.extend(process_pdf(pdf))

    write_jsonl(debug_out, extracted)

    boiler = build_boilerplate_sets(extracted, df_ratio=0.20)

    cleaned: List[Dict[str, Any]] = []
    for s in extracted:
        clean_title, clean_body = clean_slide(s, boiler_lines=boiler[s["lecture_id"]])
        s2 = dict(s)
        s2["title"] = clean_title
        s2["body"] = clean_body
        s2["text_for_embedding"] = build_text_for_embedding(s2["lecture_id"], clean_title, clean_body)
        cleaned.append(s2)

    write_jsonl(slides_out, cleaned)

    cleaned_sorted = sorted(cleaned, key=lambda x: (x["lecture_id"], x["slide_number"]))

    slide_level, passage_level = [], []
    for s in cleaned_sorted:
        slide_level.extend(chunks_slide_level(s))
        passage_level.extend(chunks_split(s, max_tokens=350))

    write_jsonl(slide_level_out, slide_level)
    write_jsonl(passage_out, passage_level)

    print("✅ KB build complete")
    print(f"Slides: {len(cleaned)} -> {slides_out}")
    print(f"slide_level chunks: {len(slide_level)} -> {slide_level_out}")
    print(f"passage_level chunks: {len(passage_level)} -> {passage_out}")
    print(f"Debug extracted (pre-clean): {debug_out}")


if __name__ == "__main__":
    main()