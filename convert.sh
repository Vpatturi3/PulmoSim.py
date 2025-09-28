python - <<'PY'
import SimpleITK as sitk, sys, os
src = "/Users/neelb/Documents/PulmoSim.py/subset1/1.3.6.1.4.1.14519.5.2.1.6279.6001.100684836163890911914061745866.mhd"   # <- change this
img = sitk.ReadImage(src)
out = os.path.splitext(src)[0] + ".nii.gz"
sitk.WriteImage(img, out)
print("Wrote:", out)
PY