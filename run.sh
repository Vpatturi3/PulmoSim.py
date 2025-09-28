python ct_to_mesh.py \
  --input "/Users/neelb/Documents/PulmoSim.py/data/manifest-1759009369418/LIDC-IDRI/LIDC-IDRI-0002/01-01-2000-NA-NA-98329/3000522.000000-NA-04919" \
  --airway-out airway_auto.stl \
  --airway-only \
  --iso 1.0 \
  --decimate 0.98 \
  --smooth-iters 2 \
  --repair-strong
