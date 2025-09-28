#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, json, base64, argparse, glob, uuid, textwrap, time, random, re
from dataclasses import dataclass
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import faiss
import fitz  # PyMuPDF
from PIL import Image
from tqdm import tqdm
import concurrent.futures as cf
import google.generativeai as genai


# ---------- CONFIG ----------

EMBED_MODEL  = "models/text-embedding-004"
# Use a model name that exists for this API/version. "gemini-1.5-flash-latest" returned 404
# when calling generate_content. ListModels showed "models/gemini-flash-latest" is available,
# so switch to that. If you prefer a different model from `genai.list_models()` pick it here.
VISION_MODEL = "models/gemini-flash-latest"   # multimodal, fast & available
TOP_K        = 6
CHUNK_CHARS  = 1800
CHUNK_OVERLAP = 250


# ---------- UTIL ----------

def chunk_text(txt: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    txt = " ".join(txt.split())
    if not txt:
        return []
    chunks, i = [], 0
    step = max(1, size - overlap)
    while i < len(txt):
        chunks.append(txt[i:i+size])
        i += step
    return chunks

def normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype("float32", copy=False)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return arr / n

def guess_mime(path: str) -> str:
    p = str(path).lower()
    if p.endswith(".png"):  return "image/png"
    if p.endswith(".jpg") or p.endswith(".jpeg"): return "image/jpeg"
    if p.endswith(".webp"): return "image/webp"
    return "image/png"


# ---------- EMBEDDINGS (Gemini) ----------

def embed_texts(texts: List[str], workers: int = 6, max_retries: int = 5) -> np.ndarray:
    """
    Parallel embeddings via google-generativeai.
    """
    def one(t: str) -> np.ndarray:
        for attempt in range(max_retries):
            try:
                r = genai.embed_content(model=EMBED_MODEL, content=t)
                return np.asarray(r["embedding"], dtype="float32")
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep((2 ** attempt) + random.random() * 0.5)

    vecs = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for v in tqdm(ex.map(one, texts), total=len(texts), desc="[embed]"):
            vecs.append(v)

    return normalize_rows(np.vstack(vecs))

def embed_one(text: str) -> np.ndarray:
    v = genai.embed_content(model=EMBED_MODEL, content=text)["embedding"]
    v = np.asarray(v, dtype="float32")[None, :]  # shape (1, d)
    return normalize_rows(v)


# ---------- INDEX (FAISS + simple metadata) ----------

@dataclass
class IndexPaths:
    index_bin: Path
    meta_json: Path

def build_index(papers_dir: Path, out_dir: Path) -> IndexPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = IndexPaths(out_dir/"faiss.index", out_dir/"meta.json")

    pdf_paths = sorted(glob.glob(str(papers_dir / "**/*.pdf"), recursive=True))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found in {papers_dir}")

    all_chunks: List[str] = []
    meta: List[Dict] = []

    print(f"[index] reading & chunking {len(pdf_paths)} PDFs ...")
    for p in tqdm(pdf_paths):
        doc = fitz.open(p)
        for i in range(len(doc)):
            page = doc[i]
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            for ci, c in enumerate(chunk_text(text)):
                all_chunks.append(c)
                meta.append({"file": str(p), "page": i + 1, "chunk_id": ci})
        doc.close()

    print(f"[index] embedding {len(all_chunks)} chunks ...")
    X = embed_texts(all_chunks)

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)   # cosine if inputs are normalized
    index.add(X)

    faiss.write_index(index, str(idx_path.index_bin))
    idx_path.meta_json.write_text(json.dumps({"chunks": all_chunks, "meta": meta}))
    print(f"[index] saved → {idx_path.index_bin} & {idx_path.meta_json}")

    return idx_path

def load_index(idx: IndexPaths):
    if not idx.index_bin.exists() or not idx.meta_json.exists():
        raise SystemExit("Index files not found. Build it first with --rebuild.")
    index = faiss.read_index(str(idx.index_bin))
    data = json.loads(idx.meta_json.read_text())
    return index, data["chunks"], data["meta"]

def search(index, chunks: List[str], meta: List[Dict], qvec: np.ndarray, top_k: int = TOP_K):
    D, I = index.search(qvec.astype("float32"), top_k)
    out = []
    for score, i in zip(D[0].tolist(), I[0].tolist()):
        if i == -1:
            continue
        out.append({"score": float(score), "text": chunks[i], "meta": meta[i]})
    return out


# ---------- GEMINI HELPERS ----------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)

def _parse_json_safely(txt: str) -> Dict:
    txt = txt.strip()
    m = _JSON_FENCE.search(txt)
    if m:
        txt = m.group(1)
    try:
        return json.loads(txt)
    except Exception:
        # return plain text if model didn’t produce valid JSON
        return {"raw_text": txt}

def _image_parts(image_paths: List[str]) -> List[Dict]:
    parts = []
    for p in image_paths:
        b = Path(p).read_bytes()
        parts.append({"mime_type": guess_mime(p), "data": b})
    return parts

def describe_images_with_gemini(image_paths: List[str]) -> Dict:
    """
    Produce a structured JSON description of the figures.
    """
    model = genai.GenerativeModel(VISION_MODEL)
    prompt = textwrap.dedent("""
        You are assisting with pulmonary drug-delivery analysis.
        Look at the figures (OpenFOAM outputs). Produce a compact JSON with keys:
        {
          "methods_compared": ["MDI","DPI","Nebulizer"] (if applicable),
          "key_findings": [bullet strings],
          "metrics": {
            "final_deposition_by_method": {"MDI": number, "DPI": number, "Nebulizer": number},
            "peak_local_deposition": {"MDI": number, "DPI": number, "Nebulizer": number}
          },
          "clinical_implications": [bullet strings]
        }
        Only return JSON. Omit fields if unknown.
    """).strip()

    parts = _image_parts(image_paths)
    resp = model.generate_content([*parts, prompt], request_options={"timeout": 300})
    return _parse_json_safely(resp.text or "")

def diagnose_with_gemini(image_paths: List[str], description_json: Dict, retrieved: List[Dict], full_medical_eval: bool = False) -> str:
    """
    Send images + structured description + retrieved evidence back to Gemini for the final assessment.
    """
    model = genai.GenerativeModel(VISION_MODEL)

    evidence = []
    for r in retrieved:
        cite = f"{Path(r['meta']['file']).name} p.{r['meta']['page']}"
        evidence.append(f"[{cite}] score={r['score']:.3f}\n{r['text']}")

    # Build user payload with description and evidence
    user_payload = {
        "description_json": description_json,
        "retrieved_context": evidence[:TOP_K]
    }

    parts = _image_parts(image_paths)

    if full_medical_eval:
        # Request a single-paragraph clinical-style evaluation. Include an explicit
        # disclaimer and avoid giving prescriptive medical orders — recommend clinician review.
        system = textwrap.dedent("""
            You are an expert pulmonary drug-delivery clinician and consultant.
            Produce a single concise paragraph (3-6 sentences) that reads like a clinical evaluation
            for a medical record. Use the images, the provided structured description (description_json),
            and the retrieved evidence to support your statements. Include: a one-sentence summary of
            overall findings, 1-2 sentences interpreting likely anatomic distribution and clinical implications,
            and 1-2 sentences suggesting next diagnostic or management steps (non-prescriptive).
            Always include a final sentence that the analysis is automated, not a substitute for clinical judgment,
            and recommend consultation with a licensed clinician before any treatment decisions. Cite supporting
            evidence inline like [filename p.X] when relevant. Keep it a single paragraph.
        """).strip()

        resp = model.generate_content(
            [*parts, system, "PATIENT DATA: \n" + json.dumps(user_payload, indent=2)],
            request_options={"timeout": 600}
        )
        return (resp.text or "").strip()

    else:
        system = textwrap.dedent("""
            Act as a pulmonary drug-delivery specialist.
            Provide an evidence-based assessment of likely drug deposition,
            method suitability (MDI vs DPI vs Nebulizer), and practical guidance.
            Cite the retrieved papers inline like [filename p.X].
            Be concise and actionable. Not medical advice—recommend clinician confirmation.
        """).strip()

        resp = model.generate_content(
            [*parts, system, "USER DATA:\n" + json.dumps(user_payload, indent=2)],
            request_options={"timeout": 600}
        )
        return resp.text or ""


def generate_patient_summary(image_paths: List[str], description_json: Dict, retrieved: List[Dict]) -> Dict:
    """
    Ask the LLM to produce a patient-friendly paragraph, a one-line summary, and 3 simple next steps.
    Returns a dict: {patient_paragraph, one_line, next_steps: []}
    """
    model = genai.GenerativeModel(VISION_MODEL)

    # Build brief evidence list for the prompt
    evidence = []
    for r in retrieved[:6]:
        cite = f"{Path(r['meta']['file']).name} p.{r['meta']['page']}"
        evidence.append(f"[{cite}] {r['text'][:200].replace('\n',' ')}...")

    system = textwrap.dedent("""
        You are a helpful communicator translating technical imaging and simulation results into
        clear, empathetic language for a patient. Produce a JSON object only, with keys:
        {
          "patient_paragraph": "A 2-4 sentence easy-to-read paragraph a patient can understand.",
          "one_line": "A single-sentence summary in plain language.",
          "next_steps": ["three simple actionable steps in plain language"]
        }
        Use no medical jargon; if technical terms are necessary, briefly define them. Keep sentences short.
    """).strip()

    user_payload = {
        "description": description_json,
        "evidence_examples": evidence
    }

    parts = _image_parts(image_paths)
    resp = model.generate_content([
        *parts,
        system,
        "PATIENT PROMPT DATA:\n" + json.dumps(user_payload, indent=2)
    ], request_options={"timeout": 300})

    out = _parse_json_safely(resp.text or "")
    # Normalize keys (ensure fields exist)
    patient_paragraph = out.get("patient_paragraph") if isinstance(out, dict) else None
    one_line = out.get("one_line") if isinstance(out, dict) else None
    next_steps = out.get("next_steps") if isinstance(out, dict) else None

    return {
        "patient_paragraph": patient_paragraph or (resp.text or ""),
        "one_line": one_line or "",
        "next_steps": next_steps or []
    }


# ---------- PIPELINE ----------

def run_pipeline(
    images: List[str],
    papers_dir: Path,
    index_dir: Path,
    out_path: Path,
    top_k: int = TOP_K,
    rebuild: bool = False,
    full_medical_eval: bool = False
):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Missing GOOGLE_API_KEY env var.")
    genai.configure(api_key=api_key)

    idx = IndexPaths(index_dir/"faiss.index", index_dir/"meta.json")
    if rebuild or not idx.index_bin.exists() or not idx.meta_json.exists():
        print("[index] building / rebuilding …")
        build_index(papers_dir, index_dir)

    index, chunks, meta = load_index(idx)

    print("[vision] describing images with Gemini …")
    description = describe_images_with_gemini(images)
    description_text = json.dumps(description, ensure_ascii=False)

    print("[retrieval] embedding description & searching evidence …")
    qvec = embed_one(description_text)
    hits = search(index, chunks, meta, qvec, top_k=top_k)

    print("[gemini] composing final assessment …")
    diagnosis = diagnose_with_gemini(images, description, hits, full_medical_eval=full_medical_eval)

    out = {
        "images": images,
        "description": description,
        "retrieved": hits,
        "diagnosis": diagnosis
    }
    # If a full clinical paragraph was requested, include it explicitly as a separate field
    if full_medical_eval:
        out["clinical_evaluation"] = diagnosis
        print("[gemini] composing patient-facing summary …")
        patient_summary = generate_patient_summary(images, description, hits)
        out["patient_summary"] = patient_summary
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote: {out_path}")


# ---------- CLI ----------

def parse_args():
    ap = argparse.ArgumentParser("Agentic RAG for pulmonary drug-delivery assessment")
    ap.add_argument("--papers", required=True, type=Path, help="Folder with research PDFs")
    ap.add_argument("--images", required=True, help="Comma-separated paths to result images")
    ap.add_argument("--index-dir", type=Path, default=Path("./rag_index"))
    ap.add_argument("--out", type=Path, default=Path("./diagnosis.json"))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--rebuild", action="store_true", help="Force rebuild the vector index")
    ap.add_argument("--full-eval", action="store_true", help="Produce a single-paragraph clinical evaluation")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    imgs = [s.strip() for s in args.images.split(",") if s.strip()]
    if not imgs:
        raise SystemExit("Provide at least one image path with --images img1.png,img2.png")
    run_pipeline(
        images=imgs,
        papers_dir=args.papers,
        index_dir=args.index_dir,
        out_path=args.out,
        top_k=args.top_k,
        rebuild=args.rebuild,
        full_medical_eval=args.full_eval,
    )
