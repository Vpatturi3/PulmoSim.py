#!/usr/bin/env python3
import os
import io
import sys
import uuid
import shutil
import zipfile
import tempfile
from typing import List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import numpy as np
import json
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import trimesh
import subprocess

# Local imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import ct_to_mesh as ctm


OUTPUT_ROOT = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def _save_uploads_to_temp(uploaded_files: List[UploadFile]) -> Tuple[Optional[str], Optional[str]]:
    """
    Save uploaded files to a temporary location and return a tuple:
    (input_path, cleanup_dir)

    - If a single NIfTI is uploaded → returns path to file
    - If a single ZIP is uploaded → extracts to dir and returns dir path
    - If multiple DICOM files (or a single .dcm) uploaded → saves into a dir and returns dir path
    """
    if not uploaded_files:
        return None, None

    temp_root = tempfile.mkdtemp(prefix="pulsrv_")

    if len(uploaded_files) == 1:
        uf = uploaded_files[0]
        name_lower = uf.filename.lower()

        if name_lower.endswith(".nii") or name_lower.endswith(".nii.gz"):
            nii_path = os.path.join(temp_root, uf.filename)
            with open(nii_path, "wb") as f:
                shutil.copyfileobj(uf.file, f)
            return nii_path, temp_root

        if name_lower.endswith(".zip"):
            zip_path = os.path.join(temp_root, uf.filename)
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(uf.file, f)
            extract_dir = os.path.join(temp_root, "dicom_series")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            return extract_dir, temp_root

        if name_lower.endswith(".dcm"):
            dicom_dir = os.path.join(temp_root, "dicom_single")
            os.makedirs(dicom_dir, exist_ok=True)
            dicom_path = os.path.join(dicom_dir, uf.filename)
            with open(dicom_path, "wb") as f:
                shutil.copyfileobj(uf.file, f)
            return dicom_dir, temp_root

    # Multiple files → assume DICOM
    dicom_dir = os.path.join(temp_root, "dicom_series")
    os.makedirs(dicom_dir, exist_ok=True)
    for uf in uploaded_files:
        # Some browsers send relative directory paths in filename when selecting folders
        # Normalize and ensure we don't escape the target root
        rel = os.path.normpath(uf.filename).lstrip(os.sep)
        dst = os.path.join(dicom_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            shutil.copyfileobj(uf.file, f)
    return dicom_dir, temp_root


def _process_case(input_path: str,
                  iso_mm: float,
                  lung_hu_low: int,
                  lung_hu_high: int,
                  decimate: float,
                  airway_seed_zyx: Optional[Tuple[int, int, int]] = None):
    img, _ = ctm.read_volume(input_path)
    img_iso = ctm.resample_isotropic(img, iso=iso_mm, interp=ctm.sitk.sitkLinear)
    lungs_mask = ctm.segment_lungs(img_iso, hu_low=lung_hu_low, hu_high=lung_hu_high, keep_components=2)

    job_id = str(uuid.uuid4())
    out_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(out_dir, exist_ok=True)

    lungs_stl = os.path.join(out_dir, "lungs.stl")
    ctm.mask_to_mesh_stl(lungs_mask, lungs_stl, decimate_ratio=decimate, verbose=False)

    airway_stl = None
    if airway_seed_zyx is not None:
        try:
            airway_mask = ctm.segment_airway_seeded(img_iso, airway_seed_zyx)
            arr = ctm.sitk.GetArrayFromImage(airway_mask)
            arr = ctm.morphology.remove_small_objects(arr.astype(bool), min_size=500).astype(np.uint8)
            airway_mask = ctm.sitk.GetImageFromArray(arr)
            airway_mask.CopyInformation(img_iso)
            airway_stl = os.path.join(out_dir, "airway.stl")
            ctm.mask_to_mesh_stl(airway_mask, airway_stl, decimate_ratio=decimate, verbose=False)
        except Exception:
            airway_stl = None

    meta = {
        "shape_zyx": list(ctm.sitk.GetArrayFromImage(img_iso).shape),
        "spacing_xyz_mm": list(img_iso.GetSpacing()),
        "size_xyz_vox": list(img_iso.GetSize()),
    }

    return job_id, lungs_stl, airway_stl, meta


app = FastAPI(title="PulmoSim API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/files", StaticFiles(directory=OUTPUT_ROOT), name="files")

# Demo asset: serve a built-in STL for homepage preview
DEMO_STL = os.path.join(CURRENT_DIR, "longen met bronchiaalboom.stl")

@app.get("/demo/longen")
async def demo_lungs():
    if os.path.exists(DEMO_STL):
        return FileResponse(DEMO_STL, media_type="model/stl")
    return JSONResponse(status_code=404, content={"error": "demo STL not found"})


@app.post("/process")
async def process(
    files: List[UploadFile] = File(...),
    iso: float = Form(1.0),
    lung_hu_low: int = Form(-1000),
    lung_hu_high: int = Form(-400),
    decimate: float = Form(0.5),
    airway_enabled: bool = Form(False),
    airway_seed_z: Optional[int] = Form(None),
    airway_seed_y: Optional[int] = Form(None),
    airway_seed_x: Optional[int] = Form(None),
):
    input_path, cleanup_dir = _save_uploads_to_temp(files)
    if input_path is None:
        return JSONResponse(status_code=400, content={"error": "No files provided"})

    try:
        seed = None
        if airway_enabled and None not in (airway_seed_z, airway_seed_y, airway_seed_x):
            seed = (int(airway_seed_z), int(airway_seed_y), int(airway_seed_x))

        job_id, lungs_stl, airway_stl, meta = _process_case(
            input_path=input_path,
            iso_mm=float(iso),
            lung_hu_low=int(lung_hu_low),
            lung_hu_high=int(lung_hu_high),
            decimate=float(decimate),
            airway_seed_zyx=seed,
        )

        base = f"/files/{job_id}"
        resp = {
            "job_id": job_id,
            "lungs_url": f"{base}/lungs.stl",
            "airway_url": (f"{base}/airway.stl" if airway_stl else None),
            "meta": meta,
        }
        return resp
    except Exception as e:
        # Map processing failures (e.g., unreadable inputs) to a 400 with message
        return JSONResponse(status_code=400, content={"error": str(e)})
    finally:
        if cleanup_dir and os.path.isdir(cleanup_dir):
            try:
                shutil.rmtree(cleanup_dir)
            except Exception:
                pass


@app.post("/run_local")
async def run_local(
    dicom_dir: str = Form(...),
    iso: float = Form(1.0),
    lung_hu_low: int = Form(-1000),
    lung_hu_high: int = Form(-400),
    decimate: float = Form(0.5),
    airway_enabled: bool = Form(False),
    airway_seed_z: Optional[int] = Form(None),
    airway_seed_y: Optional[int] = Form(None),
    airway_seed_x: Optional[int] = Form(None),
):
    # Use an existing DICOM folder path that is accessible to the server machine
    if not os.path.exists(dicom_dir) or not os.path.isdir(dicom_dir):
        return JSONResponse(status_code=400, content={"error": "dicom_dir must be an existing directory on the server"})

    try:
        seed = None
        if airway_enabled and None not in (airway_seed_z, airway_seed_y, airway_seed_x):
            seed = (int(airway_seed_z), int(airway_seed_y), int(airway_seed_x))

        job_id, lungs_stl, airway_stl, meta = _process_case(
            input_path=dicom_dir,
            iso_mm=float(iso),
            lung_hu_low=int(lung_hu_low),
            lung_hu_high=int(lung_hu_high),
            decimate=float(decimate),
            airway_seed_zyx=seed,
        )

        base = f"/files/{job_id}"
        return {
            "job_id": job_id,
            "lungs_url": f"{base}/lungs.stl",
            "airway_url": (f"{base}/airway.stl" if airway_stl else None),
            "meta": meta,
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


def _simulate_deposition_from_stl(stl_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Load STL mesh
    mesh = trimesh.load(stl_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("STL did not contain a single mesh")
    mesh.remove_unreferenced_vertices()

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    # Normalize coordinates to [0,1] for scalar fields
    mins = V.min(axis=0)
    maxs = V.max(axis=0)
    span = np.clip(maxs - mins, 1e-6, None)
    Vn = (V - mins) / span

    # Synthetic deposition fields per inhaler (proxy for real CFD)
    # Favor inferior regions (low z) for MDI, mid-basal for DPI, diffuse for Nebulizer
    z = Vn[:, 2]
    y = Vn[:, 1]
    x = Vn[:, 0]
    mdi_scalar = np.clip(1.2 * (1.0 - z) ** 1.1, 0.0, 1.0)
    dpi_scalar = np.clip(np.exp(-((z - 0.45) ** 2) / 0.02) * (0.6 + 0.4 * (1 - np.abs(x - 0.5))), 0.0, 1.0)
    neb_scalar = np.clip(0.7 * (1.0 - 0.6 * z) * (0.8 + 0.2 * (1 - np.abs(y - 0.5))), 0.0, 1.0)

    fields = {
        "mdi": {"name": "Metered Dose Inhaler", "scalar": mdi_scalar, "cmap": "Oranges"},
        "dpi": {"name": "Dry Powder Inhaler", "scalar": dpi_scalar, "cmap": "Blues"},
        "neb": {"name": "Nebulizer", "scalar": neb_scalar, "cmap": "Greens"},
    }

    image_paths = {}
    metrics = {}

    for key, spec in fields.items():
        scalars = spec["scalar"]
        # Face-wise mean scalar
        face_vals = scalars[F].mean(axis=1)
        cmap = cm.get_cmap(spec["cmap"])  # type: ignore
        colors = cmap((face_vals - face_vals.min()) / max(1e-8, (face_vals.max() - face_vals.min())))
        # Slight greyscale base tint
        base_alpha = 0.25
        colors[:, 3] = np.clip(0.35 + 0.65 * (face_vals - face_vals.min()) / max(1e-8, (face_vals.max() - face_vals.min())), 0.35, 0.98)

        fig = plt.figure(figsize=(6, 6), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        poly = Poly3DCollection(V[F], facecolors=colors, linewidths=0.05, edgecolors=(0, 0, 0, 0.05))
        ax.add_collection3d(poly)
        ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])
        ax.set_axis_off()
        ax.view_init(elev=10, azim=-90)
        ax.set_title(f"{spec['name']}\n", pad=8)

        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02, orientation='horizontal')
        cb.set_label('Deposition')

        fig.tight_layout()
        out_png = os.path.join(out_dir, f"{key}.png")
        fig.savefig(out_png, transparent=False)
        plt.close(fig)
        image_paths[key] = out_png

        # Metrics: mean scalar as total proxy
        total = float(np.clip(face_vals.mean(), 0.0, 1.0))
        metrics[key] = {"name": spec["name"], "total": total}

    # Determine best (max total)
    best_key = max(metrics.items(), key=lambda kv: kv[1]["total"])[0]

    summary = {
        "created": datetime.utcnow().isoformat() + "Z",
        "stl": os.path.abspath(stl_path),
        "images": {k: os.path.abspath(v) for k, v in image_paths.items()},
        "metrics": metrics,
        "best": best_key,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


@app.post("/simulate_deposition")
async def simulate_deposition(
    job_id: Optional[str] = Form(None),
    lungs_stl: Optional[UploadFile] = File(None),
):
    try:
        if lungs_stl is None and not job_id:
            return JSONResponse(status_code=400, content={"error": "Provide either job_id or lungs_stl"})

        if job_id:
            out_dir = os.path.join(OUTPUT_ROOT, job_id)
            stl_path = os.path.join(out_dir, "lungs.stl")
            if not os.path.exists(stl_path):
                return JSONResponse(status_code=404, content={"error": f"lungs.stl not found for job_id {job_id}"})
        else:
            # Save uploaded STL into a new job dir
            job_id = str(uuid.uuid4())
            out_dir = os.path.join(OUTPUT_ROOT, job_id)
            os.makedirs(out_dir, exist_ok=True)
            stl_path = os.path.join(out_dir, "lungs.stl")
            with open(stl_path, "wb") as f:
                shutil.copyfileobj(lungs_stl.file, f)

        summary = _simulate_deposition_from_stl(stl_path, out_dir)
        base = f"/files/{job_id}"
        images_rel = {k: f"{base}/{k}.png" for k in summary["images"].keys()}
        return {
            "job_id": job_id,
            "images": images_rel,
            "metrics": summary["metrics"],
            "best": summary["best"],
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/chatbot_answer")
async def chatbot_answer(job_id: str = Form(...), q: Optional[str] = Form(None)):
    try:
        out_dir = os.path.join(OUTPUT_ROOT, job_id)
        if not os.path.isdir(out_dir):
            return JSONResponse(status_code=404, content={"error": "Unknown job_id"})

        # Ensure deposition images exist
        imgs = [os.path.join(out_dir, f) for f in ("mdi.png", "dpi.png", "neb.png")]
        if not all(os.path.exists(p) for p in imgs):
            return JSONResponse(status_code=400, content={"error": "Run /simulate_deposition first for this job_id"})

        # If diagnosis.json missing, run the RAG pipeline now
        diag_json = os.path.join(out_dir, "diagnosis.json")
        if not os.path.exists(diag_json):
            script = os.path.join(CURRENT_DIR, "run_rag.sh")
            env = os.environ.copy()
            env.setdefault("INDEX_DIR", os.path.join(CURRENT_DIR, "rag_db"))
            env.setdefault("PAPERS_DIR", os.path.join(CURRENT_DIR, "articles"))
            env["IMAGES_PATHS"] = ",".join(imgs)
            env["OUT_JSON"] = diag_json
            proc = subprocess.run(["bash", script], cwd=CURRENT_DIR, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                return JSONResponse(status_code=400, content={"error": proc.stderr or proc.stdout})

        with open(diag_json, "r") as f:
            diag = json.load(f)

        # Also attempt to read summary for best inhaler label
        best_id = None
        best_name = None
        summary_path = os.path.join(out_dir, "summary.json")
        if os.path.exists(summary_path):
            try:
                summary = json.load(open(summary_path, "r"))
                best_id = summary.get("best")
                if best_id:
                    best_name = summary.get("metrics", {}).get(best_id, {}).get("name")
            except Exception:
                pass

        message = diag.get("diagnosis") or "Analysis generated."
        if best_name:
            message = f"Best inhaler from simulation: {best_name}.\n\n" + message
        return {"message": message, "best_id": best_id, "best_name": best_name}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/rag_assess")
async def rag_assess(
    job_id: str = Form(...),
    papers_dir: str = Form("./articles"),
    top_k: int = Form(8),
):
    try:
        out_dir = os.path.join(OUTPUT_ROOT, job_id)
        if not os.path.isdir(out_dir):
            return JSONResponse(status_code=404, content={"error": "Unknown job_id"})

        # Expect three images from simulate_deposition
        imgs = [os.path.join(out_dir, f) for f in ("mdi.png", "dpi.png", "neb.png")]
        for p in imgs:
            if not os.path.exists(p):
                return JSONResponse(status_code=400, content={"error": "Run simulate_deposition first"})

        # Invoke the RAG pipeline via run_rag.sh to keep parity with CLI
        script = os.path.join(CURRENT_DIR, "run_rag.sh")
        env = os.environ.copy()
        env.setdefault("INDEX_DIR", os.path.join(CURRENT_DIR, "rag_db"))
        env.setdefault("PAPERS_DIR", papers_dir)
        env["IMAGES_PATHS"] = ",".join(imgs)
        env["OUT_JSON"] = os.path.join(out_dir, "diagnosis.json")

        # Non-interactive execution
        proc = subprocess.run(["bash", script], cwd=CURRENT_DIR, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            return JSONResponse(status_code=400, content={"error": proc.stderr or proc.stdout})

        with open(env["OUT_JSON"], "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


