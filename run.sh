python3 ct_to_mesh.py \
  --input "/Users/neelb/Documents/PulmoSim.py/data/manifest-1759009369418/LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-NA-NA-98329/3000522.000000-NA-04919" \
  --out lungs_fixed_ring.stl \
  --iso 1.0 \
  --lung-hu-low -1000 --lung-hu-high -450 \
  --decimate 0.5 \
  --radius-mm 105 --cyl-axis z  \
  --smooth-iters 8 --repair-strong