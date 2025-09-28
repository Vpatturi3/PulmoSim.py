# PulmoSim.py

PulmoSim converts a chest CT scan (DICOM series or NIfTI) into a 3D STL mesh of the lungs, optionally segments the airway, and visualizes the result in a lightweight frontend. A placeholder physics overlay previews potential deposition patterns for different particle sizes.

## Quickstart (React + FastAPI)

### 1) Backend

```bash
pip install -r requirements.txt
python server.py  # starts FastAPI at http://localhost:8000
```

API:

- `POST /process` form-data: `files` (one or many), `iso`, `lung_hu_low`, `lung_hu_high`, `decimate`, optional `airway_enabled`, `airway_seed_z`, `airway_seed_y`, `airway_seed_x`.
- Returns JSON with `lungs_url` and optional `airway_url` (served under `/files/{job_id}/...`).

### 2) Frontend

```bash
cd frontend
npm install
npm run dev  # http://localhost:5173
```

Create `frontend/.env` (optional):

```env
VITE_API_BASE=http://localhost:8000
```

## CLI (Batch) Usage

```bash
python3 ct_to_mesh.py \
  --input /path/to/dicom_folder_or_nii \
  --out lungs.stl \
  --iso 1.0 \
  --lung-hu-low -1000 \
  --lung-hu-high -400 \
  --decimate 0.5 \
  [--airway-out airway.stl --airway-seed z y x]
```

## Notes

- DICOM uploads: The app accepts a ZIP with a DICOM series or multiple `.dcm` files. It will attempt to read the first series found.
- Large scans: Processing may take a minute; consider increasing `--iso` (e.g., 1.5–2.0 mm) for faster turnaround.
- Physics overlay: Currently a placeholder coloring. Replace with your physics model when available.

## Removing Streamlit

This repo previously included a Streamlit prototype (`app.py`). The new stack replaces it with React + FastAPI. You can remove `app.py` if not needed.
