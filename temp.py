#!/usr/bin/env python3
import os, argparse, numpy as np
import SimpleITK as sitk
import nibabel as nib
from skimage import measure, morphology
from skimage.segmentation import clear_border
from skimage.morphology import ball, binary_erosion, binary_dilation, remove_small_objects
import trimesh
import inspect  # you use this later in decimation
#from trimesh.creation import tube



import SimpleITK as sitk
import re
import inspect

from trimesh import repair as trepair
from trimesh.smoothing import filter_taubin


from skimage.filters import sato
from skimage.morphology import ball, binary_erosion, binary_dilation, binary_closing, remove_small_objects
from skimage.measure import label

from skimage.filters import sato

from skimage.filters import sato
from skimage.morphology import ball, binary_closing, remove_small_objects
from skimage.measure import label
import numpy as np
import SimpleITK as sitk
import numpy as np, trimesh

def make_seed_marker(img_iso, center_zyx, size_mm=20.0):
    """
    Big '+' plus a small sphere at the seed (z,y,x on RESAMPLED image).
    size_mm controls the cross-arm length; thickness scales automatically.
    """
    sx, sy, sz = img_iso.GetSpacing()           # (x,y,z) mm/voxel
    cz, cy, cx = [float(v) for v in center_zyx] # z,y,x indices
    p = np.array([cx*sx, cy*sy, cz*sz], dtype=float)

    arm = float(size_mm)
    thk = max(1.0, arm * 0.12)                  # thicker so it’s visible

    bars = [
        trimesh.creation.box(extents=(arm, thk, thk)),  # X
        trimesh.creation.box(extents=(thk, arm, thk)),  # Y
        trimesh.creation.box(extents=(thk, thk, arm)),  # Z
    ]
    for b in bars:
        b.apply_translation(p)

    # little sphere “dot” in the middle
    dot = trimesh.creation.icosphere(subdivisions=2, radius=max(arm * 0.15, 2.0))
    dot.apply_translation(p)

    m = trimesh.util.concatenate(bars + [dot])
    m.remove_degenerate_faces(); m.remove_unreferenced_vertices(); m.fix_normals()
    return m




def segment_airway_tree(img_iso,
                        seed_zyx=None,
                        hu_low=-1024, hu_high=-650,
                        sigmas=(0.8, 1.2, 1.8, 2.5),
                        pct=99.3,
                        min_size=1500):
    import numpy as np
    from skimage.filters import sato
    from skimage.morphology import ball, binary_closing, remove_small_objects
    from skimage.measure import label
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(img_iso)  # (z,y,x)
    Z, Y, X = arr.shape

    # 1) Body mask
    body = arr > -320
    body = binary_closing(body, ball(5))
    body = remove_small_objects(body, min_size=80_000)
    lab = label(body, connectivity=1)
    if lab.max() > 0:
        cc = np.bincount(lab.ravel()); cc[0] = 0
        body = (lab == cc.argmax())
    body_itk = sitk.GetImageFromArray(body.astype(np.uint8))
    body_itk.CopyInformation(img_iso)

    # 2) Air inside body
    air_in = ((arr >= hu_low) & (arr <= hu_high)) & body

    # 3) Seeds (user or auto)
    seeds = []
    if seed_zyx is not None:
        z, y, x = map(int, seed_zyx)
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            seeds.append((x, y, z))

    if not seeds:
        aclip = np.clip(arr, -1000, 400).astype(np.float32)
        inv = (aclip.max() - aclip)
        inv = (inv - inv.min()) / (inv.max() - inv.min() + 1e-6)
        tubeness = sato(inv, sigmas=sigmas, black_ridges=False)
        pool = tubeness[air_in] if air_in.any() else tubeness.ravel()
        thr = np.percentile(pool, pct)
        strong = (tubeness >= thr) & (air_in if air_in.any() else np.ones_like(tubeness, bool))
        idx = np.argwhere(strong)
        if idx.size:
            scores = tubeness[idx[:, 0], idx[:, 1], idx[:, 2]]
            for zz, yy, xx in idx[np.argsort(scores)[-5:]]:
                seeds.append((int(xx), int(yy), int(zz)))

    if not seeds:
        out = sitk.Image(img_iso.GetSize(), sitk.sitkUInt8)
        out.CopyInformation(img_iso)
        return out

    # 4) Region growing constrained by HU and body
    rg = sitk.ConnectedThreshold(img_iso, seedList=seeds, lower=hu_low, upper=hu_high)
    rg = sitk.Mask(rg, body_itk, outsideValue=0)
    rg = sitk.Cast(rg > 0, sitk.sitkUInt8)

    # ---- SimpleITK-compat morphology (no extra kwargs) ----
    try:
        # Preferred (short) API
        rg = sitk.BinaryMorphologicalOpening(rg, [1, 1, 1], sitk.sitkBall)
        rg = sitk.BinaryMorphologicalClosing(rg, [1, 1, 1], sitk.sitkBall)
    except TypeError:
        # Fallback: opening = erode→dilate, closing = dilate→erode
        er = sitk.BinaryErode(rg, [1, 1, 1], sitk.sitkBall, 1)
        rg = sitk.BinaryDilate(er, [1, 1, 1], sitk.sitkBall, 1)
        di = sitk.BinaryDilate(rg, [1, 1, 1], sitk.sitkBall, 1)
        rg = sitk.BinaryErode(di, [1, 1, 1], sitk.sitkBall, 1)

    # 5) Clean & keep largest component
    arr_rg = sitk.GetArrayFromImage(rg).astype(bool)
    arr_rg = remove_small_objects(arr_rg, min_size=min_size)
    lab = label(arr_rg, connectivity=1)
    if lab.max() > 0:
        cc = np.bincount(lab.ravel()); cc[0] = 0
        arr_rg = (lab == cc.argmax())

    out = sitk.GetImageFromArray(arr_rg.astype(np.uint8))
    out.CopyInformation(img_iso)
    return out


def repair_and_smooth_mesh(mesh: trimesh.Trimesh,
                           smooth_iters: int = 0,
                           strong: bool = False,
                           verbose: bool = True) -> trimesh.Trimesh:
    # Light cleanup
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    # Winding / normals
    try: trepair.fix_winding(mesh)
    except Exception: pass
    try: trepair.fix_inversion(mesh)
    except Exception: pass

    # ---- SAFE STITCH (avoid crash when no boundaries) ----
    try:
        nb = int(getattr(mesh, "edges_boundary", np.zeros((0,2))).shape[0])
        if nb > 0:
            trepair.stitch(mesh)   # multibody arg not required for most cases
        else:
            if verbose: print("[repair] no boundary edges; skipping stitch")
    except Exception as e:
        if verbose: print(f"[warn] stitch skipped: {e}")

    # Fill small holes (use method on mesh; more robust across versions)
    try:
        mesh.fill_holes()
    except Exception as e:
        if verbose: print(f"[warn] fill_holes skipped: {e}")

    # Optional smoothing (geometry only; doesn't change topology)
    if smooth_iters and smooth_iters > 0:
        filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=int(smooth_iters))
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()

    # Optional strong watertight pass
    if strong:
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(mesh.vertices.copy(), mesh.faces.copy())
            mf.repair(verbose=verbose, joincomp=True, remove_smallest_components=False)
            mesh = trimesh.Trimesh(mf.v, mf.f, process=True)
        except Exception as e:
            if verbose: print(f"[warn] strong repair skipped/failed: {e}")

    # Final tidy
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    try: trepair.fix_normals(mesh)
    except Exception: pass

    if verbose:
        print(f"[repair] watertight={mesh.is_watertight} | euler={mesh.euler_number} | "
              f"components={len(mesh.split(only_watertight=False))}")
    return mesh

def make_ring_mesh(img_iso, center_zyx, radius_mm, axis='z', thickness_mm=2.0, height_mm=1.0, segments=128):
    """
    Build a thin 3D ring (a washer) as a trimesh.Trimesh.
    - center_zyx: center in RESAMPLED voxel indices (z, y, x)
    - radius_mm: ring radius in millimeters
    - axis: plane normal for the ring ('z' => ring lies in XY plane)
    - thickness_mm: radial thickness of the ring
    - height_mm: ring height (thickness along the normal direction)
    """
    sx, sy, sz = img_iso.GetSpacing()            # (x,y,z) mm/voxel
    cz, cy, cx = [float(v) for v in center_zyx]
    cx_mm, cy_mm, cz_mm = cx*sx, cy*sy, cz*sz

    r_out = float(radius_mm)
    r_in  = max(0.1, r_out - float(thickness_mm))
    h     = float(height_mm)
    n     = int(max(12, segments))

    ang = np.linspace(0.0, 2*np.pi, n, endpoint=False)
    def circle_xy(r, z):
        x = cx_mm + r*np.cos(ang)
        y = cy_mm + r*np.sin(ang)
        z = np.full_like(ang, z, dtype=float)
        return np.column_stack([x, y, z])
    def circle_xz(r, y):
        x = cx_mm + r*np.cos(ang)
        z = cz_mm + r*np.sin(ang)
        y = np.full_like(ang, y, dtype=float)
        return np.column_stack([x, y, z])
    def circle_yz(r, x):
        y = cy_mm + r*np.cos(ang)
        z = cz_mm + r*np.sin(ang)
        x = np.full_like(ang, x, dtype=float)
        return np.column_stack([x, y, z])

    if axis == 'z':
        top_outer = circle_xy(r_out, cz_mm + h/2); top_inner = circle_xy(r_in, cz_mm + h/2)
        bot_outer = circle_xy(r_out, cz_mm - h/2); bot_inner = circle_xy(r_in, cz_mm - h/2)
    elif axis == 'y':
        top_outer = circle_xz(r_out, cy_mm + h/2); top_inner = circle_xz(r_in, cy_mm + h/2)
        bot_outer = circle_xz(r_out, cy_mm - h/2); bot_inner = circle_xz(r_in, cy_mm - h/2)
    else:  # 'x'
        top_outer = circle_yz(r_out, cx_mm + h/2); top_inner = circle_yz(r_in, cx_mm + h/2)
        bot_outer = circle_yz(r_out, cx_mm - h/2); bot_inner = circle_yz(r_in, cx_mm - h/2)

    V = np.vstack([top_outer, top_inner, bot_inner, bot_outer])
    idx_top_outer = np.arange(0, n)
    idx_top_inner = np.arange(n, 2*n)
    idx_bot_inner = np.arange(2*n, 3*n)
    idx_bot_outer = np.arange(3*n, 4*n)

    F = []
    for i in range(n):
        j = (i + 1) % n
        # top face (annulus)
        F.append([idx_top_outer[i], idx_top_outer[j], idx_top_inner[i]])
        F.append([idx_top_inner[i], idx_top_outer[j], idx_top_inner[j]])
        # bottom face (reverse winding)
        F.append([idx_bot_inner[i], idx_bot_outer[j], idx_bot_outer[i]])
        F.append([idx_bot_inner[i], idx_bot_inner[j], idx_bot_outer[j]])
        # outer wall
        F.append([idx_top_outer[i], idx_bot_outer[i], idx_bot_outer[j]])
        F.append([idx_top_outer[i], idx_bot_outer[j], idx_top_outer[j]])
        # inner wall (reverse orientation so normals point outward)
        F.append([idx_top_inner[i], idx_top_inner[j], idx_bot_inner[j]])
        F.append([idx_top_inner[i], idx_bot_inner[j], idx_bot_inner[i]])

    ring = trimesh.Trimesh(vertices=V, faces=np.array(F), process=True)
    ring.remove_degenerate_faces(); ring.remove_unreferenced_vertices(); ring.fix_normals()
    return ring


def clip_mask_radius(mask_itk, img_iso, radius_mm, center_zyx=None, axis=None):
    """
    Clip mask to a sphere (axis=None) or an infinite cylinder around `axis` ('x'/'y'/'z').
    radius_mm is in millimeters. center_zyx is in RESAMPLED indices (z,y,x).
    """
    arr = sitk.GetArrayFromImage(mask_itk).astype(bool)  # (z,y,x)
    Z, Y, X = arr.shape
    sx, sy, sz = img_iso.GetSpacing()  # (x,y,z) mm/voxel

    # center: COM of mask if not provided
    if center_zyx is None:
        coords = np.argwhere(arr)
        if coords.size:
            cz, cy, cx = coords.mean(axis=0)
        else:
            cz, cy, cx = Z/2.0, Y/2.0, X/2.0
    else:
        cz, cy, cx = [float(v) for v in center_zyx]

    zz = np.arange(Z)[:, None, None]
    yy = np.arange(Y)[None, :, None]
    xx = np.arange(X)[None, None, :]

    dz = (zz - cz) * sz
    dy = (yy - cy) * sy
    dx = (xx - cx) * sx

    if axis == 'z':          # cylinder about z → radial in x–y
        dist2 = dx*dx + dy*dy
    elif axis == 'y':        # cylinder about y → radial in x–z
        dist2 = dx*dx + dz*dz
    elif axis == 'x':        # cylinder about x → radial in y–z
        dist2 = dy*dy + dz*dz
    else:                    # sphere
        dist2 = dx*dx + dy*dy + dz*dz

    keep = dist2 <= (radius_mm * radius_mm)
    clipped = arr & keep

    out = sitk.GetImageFromArray(clipped.astype(np.uint8))
    out.CopyInformation(mask_itk)
    return out



def _series_meta(first_file):
    r = sitk.ImageFileReader()
    r.SetFileName(first_file)
    r.ReadImageInformation()
    def g(tag): 
        return r.GetMetaData(tag) if r.HasMetaDataKey(tag) else ""
    return {
        "Modality": g("0008|0060"),
        "SeriesDescription": g("0008|103e"),
        "SeriesNumber": g("0020|0011"),
        "SliceThickness": g("0018|0050"),
        "PixelSpacing": g("0028|0030"),
    }

def _pick_best_series(input_dir):
    reader = sitk.ImageSeriesReader()
    sids = reader.GetGDCMSeriesIDs(input_dir)
    if not sids:
        raise RuntimeError(f"No DICOM series found in: {input_dir}")

    best = None
    for sid in sids:
        files = reader.GetGDCMSeriesFileNames(input_dir, sid)
        meta = _series_meta(files[0])
        desc = (meta["SeriesDescription"] or "").lower()
        modality = (meta["Modality"] or "").upper()

        # score: prefer CT modality, many files, and avoid SEG/SCOUT/SECONDARY
        score = len(files)
        if modality == "CT": score += 100000
        if re.search(r"seg|segmentation|secondary|scout|localizer", desc):
            score -= 100000

        cand = {
            "sid": sid,
            "files": files,
            "n": len(files),
            "meta": meta,
            "score": score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best

def read_volume(input_path):
    """
    Read a CT volume from a DICOM folder or a NIfTI file.
    For DICOM directories with multiple series, auto-pick the largest CT series.
    """
    if os.path.isdir(input_path):
        try:
            best = _pick_best_series(input_path)
            files = best["files"]
            meta = best["meta"]
            print(f"[dicom] picked series: {best['n']} slices | "
                  f"Modality={meta['Modality']} | "
                  f"Desc='{meta['SeriesDescription']}' | "
                  f"SliceThickness={meta['SliceThickness']} | PixelSpacing={meta['PixelSpacing']}")
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            img = reader.Execute()
            return img, "dicom"
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM: {e}")
    else:
        # NIfTI
        ext = os.path.splitext(input_path)[1].lower()
        if ext in [".mhd", ".mha"]:
            img = sitk.ReadImage(input_path)         # reads the .mhd and its .raw
            return img, "mhd"
        else:
            # NIfTI
            nii = nib.load(input_path)
            data = nii.get_fdata().astype(np.float32)
            spacing = nii.header.get_zooms()  # (z,y,x)
            img = sitk.GetImageFromArray(data)
            img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
            return img, "nifti"


def resample_isotropic(img, iso=1.0, interp=sitk.sitkLinear):
    orig_spacing = np.array(list(img.GetSpacing()), dtype=np.float32)  # (x,y,z)
    orig_size = np.array(list(img.GetSize()), dtype=np.int64)         # (x,y,z)
    new_spacing = np.array([iso, iso, iso], dtype=np.float32)
    new_size = np.round(orig_size * (orig_spacing / new_spacing)).astype(int)
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interp)
    resampler.SetOutputSpacing(tuple(new_spacing.tolist()))
    resampler.SetSize([int(s) for s in new_size])
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetDefaultPixelValue(0)
    img_iso = resampler.Execute(img)
    return img_iso




from skimage.morphology import ball, binary_closing, remove_small_objects
from skimage.measure import label
import numpy as np
import SimpleITK as sitk

from skimage.morphology import ball, binary_closing, remove_small_objects
from skimage.measure import label

def segment_lung_lumen(img_iso,
                       air_low=-1024, air_high=-600,     # air threshold (tight)
                       body_thresh=-320,                 # body vs outside
                       body_close=7,                     # close neck/gaps
                       min_lung_vox=20000,               # remove tiny junk
                       keep_components=2):
    """Return a mask of the lung lumen (air inside the body, excluding outside air)."""
    arr = sitk.GetArrayFromImage(img_iso)  # z,y,x

    # 1) Body mask
    body = arr > body_thresh
    body = binary_closing(body, ball(body_close))
    body = remove_small_objects(body, min_size=80000)

    # 2) Outside air = complement of body that touches the volume border
    outside = ~body
    lab_out = label(outside, connectivity=1)
    Z, Y, X = outside.shape
    border = np.zeros_like(outside, dtype=bool)
    border[0,:,:]=border[-1,:,:]=True; border[:,0,:]=border[:,-1,:]=True; border[:,:,0]=border[:,:,-1]=True
    outside_labels = np.unique(lab_out[border])
    outside_border = np.isin(lab_out, outside_labels)

    # 3) Lung lumen = air that is NOT outside-border-connected
    air = (arr >= air_low) & (arr <= air_high)
    lumen = air & (~outside_border)
    lumen = remove_small_objects(lumen, min_size=min_lung_vox)

    # 4) Keep left/right lungs (2 largest components) and tidy
    lab = label(lumen, connectivity=1)
    if lab.max() == 0:
        raise ValueError("Lumen mask empty; try a higher air_high (e.g., -550).")
    counts = np.bincount(lab.ravel()); counts[0] = 0
    keep = counts.argsort()[::-1][:keep_components]
    lumen = np.isin(lab, keep)
    lumen = binary_closing(lumen, ball(2))

    out = sitk.GetImageFromArray(lumen.astype(np.uint8))
    out.CopyInformation(img_iso)
    return out


def segment_lungs(img_iso,
                  hu_low=-1000, hu_high=-320,         # air-ish threshold (less strict)
                  keep_components=2,
                  body_thresh=-250,                    # body includes soft tissue/bone
                  body_min_size=80000,                 # drop small junk
                  close_radius=6):                     # solidify lungs
    """
    Body-first method → lungs are the low-HU region *inside* the body,
    then we 'solidify' to get the full lung volume (not just airways).
    """
    arr = sitk.GetArrayFromImage(img_iso)  # z,y,x

    # 1) Body mask: anything denser than outside air
    body = arr > body_thresh
    body = binary_closing(body, ball(7))                      # seal neck/gaps
    body = remove_small_objects(body, min_size=body_min_size)

    lab_body = label(body, connectivity=1)
    if lab_body.max() == 0:
        raise ValueError("Body mask empty; raise body_thresh (e.g., -200).")
    counts = np.bincount(lab_body.ravel()); counts[0] = 0
    body = lab_body == counts.argmax()                        # keep largest body

    # 2) Air inside body → initial lung candidates
    air = (arr >= hu_low) & (arr <= hu_high)
    lung_air = air & body

    # 3) Solidify lungs: close gaps (vessels, fissures) and fill internal holes
    #    This turns the air-lumen into a solid parenchyma volume
    lung_solid = binary_closing(lung_air, ball(close_radius))
    lung_solid = remove_small_objects(lung_solid, min_size=20000)

    # 4) Keep left/right lungs (2 largest components)
    lab = label(lung_solid, connectivity=1)
    if lab.max() == 0:
        raise ValueError("Lung mask empty; relax hu_high to -300/-280 or raise close_radius.")
    cc = np.bincount(lab.ravel()); cc[0] = 0
    keep = cc.argsort()[::-1][:keep_components]
    lungs_np = np.isin(lab, keep)

    mask = sitk.GetImageFromArray(lungs_np.astype(np.uint8))
    mask.CopyInformation(img_iso)
    mask = sitk.BinaryFillhole(mask)  # final tidy
    return sitk.Cast(mask, sitk.sitkUInt8)




def segment_airway_seeded(img_iso, seed_zyx, low=-1024, high=-800, iters=2):
    """
    Very basic airway segmentation via seeded region growing around trachea.
    Provide seed in (z,y,x) coordinates on the resampled volume.
    """
    arr = sitk.GetArrayFromImage(img_iso)  # z,y,x
    z, y, x = [int(v) for v in seed_zyx]
    if not (0 <= z < arr.shape[0] and 0 <= y < arr.shape[1] and 0 <= x < arr.shape[2]):
        raise ValueError("Airway seed is out of bounds for the resampled volume.")

    # Region growing constrained by HU range (air)
    rg = sitk.Cast(rg > 0, sitk.sitkUInt8)  # ensure binary
    kernel = [1, 1, 1]  # radius (x,y,z) in voxels
    rg = sitk.BinaryMorphologicalOpening(
        rg,
        kernelRadius=kernel,
        foregroundValue=1,
        backgroundValue=0,
        safeBorder=True,
        kernelType=sitk.sitkBall,
    )
    for _ in range(iters):
        rg = sitk.BinaryMorphologicalClosing(
            rg,
            kernelRadius=kernel,
            foregroundValue=1,
            backgroundValue=0,
            safeBorder=True,
            kernelType=sitk.sitkBall,
        )
        rg = sitk.Cast(rg > 0, sitk.sitkUInt8)
    return rg

def apply_mask_to_ct(img_iso, lungs_mask, outside_hu=-1024.0):
    """Keep lung voxels; set everything else to outside_hu (air)."""
    m = sitk.Cast(lungs_mask, img_iso.GetPixelID())
    return sitk.Mask(img_iso, m, outsideValue=float(outside_hu))

def bbox_from_mask(mask_itk, pad_vox=8):
    """Return (index,size) in (x,y,z) for a tight bbox around mask, padded."""
    arr = sitk.GetArrayFromImage(mask_itk)  # z,y,x
    coords = np.argwhere(arr > 0)
    if coords.size == 0:
        raise ValueError("Mask empty; cannot compute bbox.")
    z0, y0, x0 = coords.min(axis=0); z1, y1, x1 = coords.max(axis=0) + 1
    Z, Y, X = arr.shape
    z0 = max(0, z0 - pad_vox); y0 = max(0, y0 - pad_vox); x0 = max(0, x0 - pad_vox)
    z1 = min(Z, z1 + pad_vox); y1 = min(Y, y1 + pad_vox); x1 = min(X, x1 + pad_vox)
    index = [int(x0), int(y0), int(z0)]
    size  = [int(x1 - x0), int(y1 - y0), int(z1 - z0)]
    return index, size


def mask_to_mesh_stl(mask_itk, out_path, decimate_ratio=0.5, smooth_iters=0,
                     repair_strong=False, verbose=True):
    """
    Convert a binary mask (SITK) to STL via marching cubes, repairs, optional smoothing.
    decimate_ratio = fraction of faces to KEEP (0.10..0.99)
    """
    dirpath = os.path.dirname(os.path.abspath(out_path))
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    mask_np = sitk.GetArrayFromImage(mask_itk)       # z,y,x
    spacing = mask_itk.GetSpacing()                  # (x,y,z)
    spacing_zyx = (spacing[2], spacing[1], spacing[0])

    if verbose:
        print(f"[mesh] marching cubes on mask of shape {mask_np.shape} with spacing (z,y,x)={spacing_zyx}")

    verts, faces, _, _ = measure.marching_cubes(
        mask_np.astype(np.uint8), level=0.5, spacing=spacing_zyx
    )

    # skimage gives verts in (z,y,x); convert to (x,y,z)
    verts_xyz = verts[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=False)

    # Basic repairs before decimation
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    # Safe decimation (optional)
    n_faces = int(len(mesh.faces))
    r_keep = float(decimate_ratio)
    if not (n_faces < 2000 or r_keep >= 0.999):
        r_keep = float(np.clip(r_keep, 0.10, 0.99))
        target_faces = int(np.clip(n_faces * r_keep, 100, n_faces - 1))
        dec_fn = getattr(mesh, "simplify_quadratic_decimation", None) \
                 or getattr(mesh, "simplify_quadric_decimation", None)
        try:
            if dec_fn is not None:
                sig = inspect.signature(dec_fn)
                if "face_count" in sig.parameters:
                    mesh = dec_fn(face_count=target_faces)
                elif "target_reduction" in sig.parameters:
                    reduction = float(np.clip(1.0 - r_keep, 0.01, 0.90))
                    mesh = dec_fn(target_reduction=reduction)
        except Exception as e:
            if verbose:
                print(f"[warn] decimation failed ({e}); continuing without decimation.")

    # **New**: run topology repair + optional smoothing
    mesh = repair_and_smooth_mesh(mesh, smooth_iters=int(smooth_iters),
                                  strong=bool(repair_strong), verbose=verbose)

    mesh.export(out_path)
    if verbose:
        print(f"[ok] wrote {out_path}  |  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")



def main():
    ap = argparse.ArgumentParser(description="Convert a chest CT into 3D STL meshes (lungs and/or airway tree).")
    ap.add_argument("--input", required=True, help="Path to DICOM folder OR a NIfTI (.nii/.nii.gz) file")

    # Lungs (optional if --airway-only)
    ap.add_argument("--out", type=str, default=None, help="Output STL for lungs (omit if --airway-only).")
    ap.add_argument("--airway-only", action="store_true", help="Skip lung mesh and only export the airway tree.")

    # Common controls
    ap.add_argument("--iso", type=float, default=1.0, help="Isotropic resample spacing in mm (default: 1.0)")
    ap.add_argument("--lung-hu-low", type=int, default=-1000, help="Lower HU for lung lumen threshold (default: -1000)")
    ap.add_argument("--lung-hu-high", type=int, default=-400,  help="Upper HU for lung lumen threshold (e.g. -600)")
    ap.add_argument("--decimate", type=float, default=0.5,     help="Face ratio after decimation (0.1–0.95, default 0.5)")
    ap.add_argument("--smooth-iters", type=int, default=6,     help="Taubin smoothing iterations (0 to disable).")
    ap.add_argument("--repair-strong", action="store_true",    help="Try stronger watertight repair (pymeshfix).")

    # Airway export (optional; can be used with or without lung mesh)
    ap.add_argument("--airway-out", type=str, default=None, help="Output STL for airway tree (optional).")
    ap.add_argument("--airway-seed", nargs=3, type=int, default=None,
                    help="Seed voxel for airway (z y x) in the resampled volume (optional).")

    # Aux saves
    ap.add_argument("--save-lung-mask", type=str, default=None, help="Write lung mask NIfTI (.nii/.nii.gz).")
    ap.add_argument("--save-masked-nii", type=str, default=None, help="Write lungs-only CT (outside set to -1024 HU).")
    ap.add_argument("--save-cropped-nii", type=str, default=None, help="Also write a cropped lungs-only CT.")
    ap.add_argument("--crop-pad", type=int, default=8, help="Padding (voxels) around lung bbox when cropping.")
    ap.add_argument("--outside-hu", type=float, default=-1024.0, help="HU value outside lung in masked volume.")

    # Optional geometric clip + visual ring
    ap.add_argument("--radius-mm", type=float, default=None,
                    help="Clip mask to this radius (mm). With --cyl-axis, uses a cylinder; else a sphere.")
    ap.add_argument("--cyl-axis", choices=['x', 'y', 'z'], default=None,
                    help="Axis for cylindrical clip. Omit for spherical clip.")
    ap.add_argument("--center-zyx", nargs=3, type=float, default=None,
                    help="Clip center in RESAMPLED voxel indices (z y x). Default = mask COM.")
    ap.add_argument("--draw-circle", action="store_true",
                    help="Overlay a thin ring into the lung STL for visual guidance (requires --out and --radius-mm).")
    ap.add_argument("--mark-seed", action="store_true",
                    help="Overlay a small '+' marker at --airway-seed in the output STL.")
    ap.add_argument("--mark-size-mm", type=float, default=4.0,
                    help="Marker size in mm (length of each bar).")

    args = ap.parse_args()

    # Require at least one output
    if not args.out and not args.airway_out:
        raise SystemExit("Provide at least one output: --out (lungs) and/or --airway-out (airway tree).")

    # Read & resample
    print(f"[io] reading {args.input} ...")
    img, src = read_volume(args.input)
    print(f"[io] source: {src} | size (x,y,z)={img.GetSize()} | spacing (x,y,z)={img.GetSpacing()}")
    print(f"[resample] to isotropic {args.iso} mm ...")
    img_iso = resample_isotropic(img, iso=args.iso, interp=sitk.sitkLinear)
    print(f"[resample] new size (x,y,z)={img_iso.GetSize()} | spacing (x,y,z)={img_iso.GetSpacing()}")

    # ---------------- LUNG LUMEN (optional if not airway-only) ----------------
    if (not args.airway_only) and args.out:
        print("[segment] lung lumen (air space) ...")
        lungs_mask = segment_lung_lumen(
            img_iso,
            air_low=args.lung_hu_low,
            air_high=args.lung_hu_high,
        )

        # Optional clip
        if args.radius_mm is not None:
            lungs_mask = clip_mask_radius(
                lungs_mask, img_iso,
                radius_mm=float(args.radius_mm),
                center_zyx=args.center_zyx,
                axis=args.cyl_axis
            )

        # Optional saves
        if args.save_lung_mask:
            sitk.WriteImage(lungs_mask, args.save_lung_mask)
            print(f"[ok] wrote lung mask: {args.save_lung_mask}")

        if args.save_masked_nii or args.save_cropped_nii:
            print("[volume] creating lungs-only CT ...")
            ct_masked = apply_mask_to_ct(img_iso, lungs_mask, outside_hu=args.outside_hu)
            if args.save_masked_nii:
                sitk.WriteImage(ct_masked, args.save_masked_nii)
                print(f"[ok] wrote masked NIfTI: {args.save_masked_nii}")
            if args.save_cropped_nii:
                idx, sz = bbox_from_mask(lungs_mask, pad_vox=args.crop_pad)
                roi = sitk.RegionOfInterest(ct_masked, size=sz, index=idx)
                sitk.WriteImage(roi, args.save_cropped_nii)
                print(f"[ok] wrote cropped NIfTI: {args.save_cropped_nii}")

        # Mesh lungs
        print("[mesh] lungs → STL ...")
        mask_to_mesh_stl(
            lungs_mask, args.out,
            decimate_ratio=args.decimate,
            smooth_iters=args.smooth_iters,
            repair_strong=args.repair_strong
        )

        # Optional ring overlay (only if we produced a lung STL)
        if args.draw_circle:
            if args.radius_mm is None:
                raise ValueError("--radius-mm is required when using --draw-circle")
            # Default center = COM of lung mask
            if args.center_zyx is None:
                arr_mask = sitk.GetArrayFromImage(lungs_mask).astype(bool)
                coords = np.argwhere(arr_mask)
                if coords.size:
                    cz, cy, cx = coords.mean(axis=0)
                else:
                    Z, Y, X = arr_mask.shape
                    cz, cy, cx = Z/2.0, Y/2.0, X/2.0
                center_zyx = (float(cz), float(cy), float(cx))
            else:
                center_zyx = tuple(args.center_zyx)

            ring = make_ring_mesh(
                img_iso, center_zyx,
                radius_mm=float(args.radius_mm),
                axis='z', thickness_mm=2.0, height_mm=1.0, segments=128
            )
            base = trimesh.load(args.out)
            merged = trimesh.util.concatenate([base, ring])
            merged.export(args.out)
            print("[ok] drew circle and merged into:", args.out)

    # ---------------- AIRWAY TREE (optional) ----------------
    if args.airway_out:
        print(f"[airway] segmenting airway tree {'(auto-seed)' if args.airway_seed is None else ''} ...")
        airway_mask = segment_airway_tree(
            img_iso,
            seed_zyx=tuple(args.airway_seed) if args.airway_seed else None,
            hu_low=-1024, hu_high=-500
        )
        # Optional: apply the same clip to airway mask if radius provided
        if args.radius_mm is not None:
            airway_mask = clip_mask_radius(
                airway_mask, img_iso,
                radius_mm=float(args.radius_mm),
                center_zyx=args.center_zyx,
                axis=args.cyl_axis
            )
        print("[mesh] airway tree → STL ...")
        mask_to_mesh_stl(
            airway_mask, args.airway_out,
            decimate_ratio=max(args.decimate, 0.98),  # preserve detail
            smooth_iters=2,
            repair_strong=True
        )
    if args.airway_out and args.airway_seed and args.mark_seed:
        base = trimesh.load(args.airway_out)
        marker = make_seed_marker(
            img_iso,
            center_zyx=tuple(args.airway_seed),      # (z,y,x) on the resampled image
            size_mm=float(args.mark_size_mm),
            thickness_mm=max(0.6, float(args.mark_size_mm) * 0.2),
        )
        merged = trimesh.util.concatenate([base, marker])
        merged.export(args.airway_out)
        print(f"[ok] added seed marker at ZYX={tuple(args.airway_seed)} to {args.airway_out}")

    print("[done]")



if __name__ == "__main__":
    main()
