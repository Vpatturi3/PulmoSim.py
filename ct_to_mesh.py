#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Imaging + IO
import SimpleITK as sitk
import nibabel as nib

# Geometry
from skimage import measure, morphology
import trimesh


def read_volume(input_path):
    """
    Read a CT volume from a DICOM folder or a NIfTI file.
    Returns: (sitk.Image in original spacing), source_type str
    """
    if os.path.isdir(input_path):
        # Assume DICOM series
        reader = sitk.ImageSeriesReader()
        try:
            series_ids = reader.GetGDCMSeriesIDs(input_path)
            if not series_ids:
                raise RuntimeError("No DICOM series found in directory.")
            series_files = reader.GetGDCMSeriesFileNames(input_path, series_ids[0])
            reader.SetFileNames(series_files)
            img = reader.Execute()
            return img, "dicom"
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM: {e}")
    else:
        # Assume NIfTI
        try:
            nii = nib.load(input_path)
            data = nii.get_fdata().astype(np.float32)
            # Nibabel returns affine that encodes spacing; SimpleITK is simpler downstream
            spacing = nii.header.get_zooms()  # (z, y, x)
            # Create SITK image with correct spacing (SITK expects (x,y,z))
            img = sitk.GetImageFromArray(data)  # z,y,x indexing
            img.SetSpacing((float(spacing[2]), float(spacing[1]), float(spacing[0])))
            return img, "nifti"
        except Exception as e:
            raise RuntimeError(f"Failed to read NIfTI: {e}")


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


def segment_lungs(img_iso, hu_low=-1000, hu_high=-400, keep_components=2):
    """
    Simple, robust lung mask: HU threshold -> largest 2 components -> hole close.
    Returns a binary SITK image (uint8).
    """
    # Ensure Hounsfield-like range (some NIfTI may be scaled oddly; assume already HU if DICOM)
    arr = sitk.GetArrayFromImage(img_iso)  # z,y,x
    mask_np = (arr >= hu_low) & (arr <= hu_high)

    mask = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    mask.CopyInformation(img_iso)

    # Remove small speckles and outside air: keep largest connected components
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    labels = list(stats.GetLabels())
    labels_sorted = sorted(labels, key=lambda l: stats.GetPhysicalSize(l), reverse=True)[:keep_components]
    keep = sitk.Image(cc.GetSize(), sitk.sitkUInt8)
    keep.CopyInformation(cc)
    for l in labels_sorted:
        keep = keep | sitk.Equal(cc, l)

    # Close small holes and smooth a bit
    closed = sitk.BinaryMorphologicalClosing(keep, [2, 2, 2])
    filled = sitk.BinaryFillhole(closed)
    return sitk.Cast(filled, sitk.sitkUInt8)


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
    rg = sitk.ConnectedThreshold(img_iso, seedList=[(x, y, z)], lower=low, upper=high)
    # Optional morphological opening/closing to tidy up
    kern = sitk.BinaryBallStructuringElement(1)  # radius 1
    rg = sitk.BinaryMorphologicalOpening(rg, kern)
    for _ in range(iters):
        rg = sitk.BinaryMorphologicalClosing(rg, kern)
    rg = sitk.Cast(rg > 0, sitk.sitkUInt8)
    return rg


def mask_to_mesh_stl(mask_itk, out_path, decimate_ratio=0.5, smooth_iters=0, verbose=True):
    """
    Convert a binary mask (SITK) to STL via marching cubes, basic repair & decimation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    mask_np = sitk.GetArrayFromImage(mask_itk)  # z,y,x
    spacing = mask_itk.GetSpacing()             # (x,y,z)
    # skimage marching_cubes expects spacing per axis of the array order (z, y, x)
    spacing_zyx = (spacing[2], spacing[1], spacing[0])

    if verbose:
        print(f"[mesh] marching cubes on mask of shape {mask_np.shape} with spacing (z,y,x)={spacing_zyx}")

    # Marching cubes at 0.5 (between 0 and 1)
    verts, faces, _, _ = measure.marching_cubes(mask_np.astype(np.uint8), level=0.5, spacing=spacing_zyx)

    # skimage returns verts in (z,y,x) world; swap to (x,y,z) for sanity
    verts_xyz = verts[:, ::-1]

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)

    # Basic repairs
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fill_holes()
    mesh.fix_normals()

    # Optional smoothing (Laplacian-like via Taubin needs VTK; skip here to keep deps light)
    # Decimate (0.0=no faces left, 1.0=no decimation; so clamp between 0.1 and 0.95)
    # Decimate safely: keep between 10%..99% of faces and never exceed current face count
    n_faces = int(len(mesh.faces))
    r = float(decimate_ratio)

    # If user asks for ~no decimation or the mesh is small, skip
    if r >= 0.99 or n_faces < 2000:
        pass
    else:
        r = float(np.clip(r, 0.10, 0.99))          # keep this fraction of faces
        target = int(max(min(n_faces - 1, n_faces * r), 100))  # [100 .. n_faces-1]
        mesh = mesh.simplify_quadratic_decimation(target)

    # Ensure watertight if possible
    mesh.fill_holes()
    if not mesh.is_watertight and verbose:
        print("[warn] mesh not fully watertight; consider post-processing in MeshLab/Blender.")

    mesh.export(out_path)
    if verbose:
        print(f"[ok] wrote {out_path}  |  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")


def main():
    ap = argparse.ArgumentParser(description="Convert a chest CT into 3D STL meshes (lungs, optional airway).")
    ap.add_argument("--input", required=True, help="Path to DICOM folder OR a NIfTI (.nii/.nii.gz) file")
    ap.add_argument("--out", required=True, help="Output STL for lungs (e.g., lungs.stl)")
    ap.add_argument("--iso", type=float, default=1.0, help="Isotropic resample spacing in mm (default: 1.0)")
    ap.add_argument("--lung-hu-low", type=int, default=-1000, help="Lower HU for lung threshold (default: -1000)")
    ap.add_argument("--lung-hu-high", type=int, default=-400, help="Upper HU for lung threshold (default: -400)")
    ap.add_argument("--decimate", type=float, default=0.5, help="Face ratio after decimation (0.1–0.95, default 0.5)")

    # Optional airway
    ap.add_argument("--airway-out", type=str, default=None, help="Output STL for airway (optional)")
    ap.add_argument("--airway-seed", nargs=3, type=int, default=None,
                    help="Seed voxel for airway (z y x) in the resampled volume")

    args = ap.parse_args()

    # Read
    print(f"[io] reading {args.input} ...")
    img, src = read_volume(args.input)
    print(f"[io] source: {src} | size (x,y,z)={img.GetSize()} | spacing (x,y,z)={img.GetSpacing()}")

    # Resample
    print(f"[resample] to isotropic {args.iso} mm ...")
    img_iso = resample_isotropic(img, iso=args.iso, interp=sitk.sitkLinear)
    print(f"[resample] new size (x,y,z)={img_iso.GetSize()} | spacing (x,y,z)={img_iso.GetSpacing()}")

    # Segment lungs
    print("[segment] lungs via HU threshold and connected components ...")
    lungs_mask = segment_lungs(img_iso, hu_low=args.lung_hu_low, hu_high=args.lung_hu_high, keep_components=2)

    # Mesh lungs
    print("[mesh] lungs → STL ...")
    mask_to_mesh_stl(lungs_mask, args.out, decimate_ratio=args.decimate)

    # Optional airway
    if args.airway_out:
        if args.airway_seed is None:
            print("[airway] --airway-out was provided but no --airway-seed; skipping airway.")
        else:
            print(f"[segment] airway via seeded region grow at {tuple(args.airway_seed)} ...")
            try:
                airway_mask = segment_airway_seeded(img_iso, args.airway_seed)
                # Remove tiny components and keep largest airway tree
                arr = sitk.GetArrayFromImage(airway_mask)
                arr = morphology.remove_small_objects(arr.astype(bool), min_size=500).astype(np.uint8)
                airway_mask = sitk.GetImageFromArray(arr)
                airway_mask.CopyInformation(img_iso)

                print("[mesh] airway → STL ...")
                mask_to_mesh_stl(airway_mask, args.airway_out, decimate_ratio=args.decimate)
            except Exception as e:
                print(f"[airway][warn] airway segmentation failed: {e}")

    print("[done]")


if __name__ == "__main__":
    main()
