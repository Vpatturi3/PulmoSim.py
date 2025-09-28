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

    # Multiple files → assume DICOM. Preserve all files uniquely to avoid overwrites
    dicom_dir = os.path.join(temp_root, "dicom_series")
    os.makedirs(dicom_dir, exist_ok=True)
    for idx, uf in enumerate(uploaded_files):
        # Many browsers do not preserve relative paths in filename; avoid collisions
        base = os.path.basename(uf.filename) or f"file_{idx:06d}"
        dst = os.path.join(dicom_dir, f"{idx:06d}_{base}")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)


