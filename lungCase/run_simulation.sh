#!/bin/bash
# OpenFOAM simulation runner script

echo "=== Starting OpenFOAM Lung Simulation ==="
source /opt/openfoam11/etc/bashrc

# Change to case directory
cd /home/openfoam/case

echo "=== Checking mesh quality ==="
checkMesh > log.checkMesh 2>&1
cat log.checkMesh | tail -20

echo "=== Initializing flow field ==="
potentialFoam -initialiseUBCs > log.potentialFoam 2>&1

echo "=== Running steady-state solver ==="
simpleFoam > log.simpleFoam 2>&1

echo "=== Simulation Results ==="
tail -20 log.simpleFoam

echo "=== Checking convergence ==="
grep "Time = " log.simpleFoam | tail -5

echo "=== OpenFOAM simulation complete ==="