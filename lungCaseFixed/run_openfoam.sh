#!/bin/bash
# OpenFOAM Lung Simulation Runner

echo "=== OpenFOAM Lung Airflow Simulation ==="
source /opt/openfoam11/etc/bashrc

# Check if we're in the right directory
if [ ! -f "constant/lungMesh.msh" ]; then
    echo "Error: lungMesh.msh not found in constant/"
    exit 1
fi

echo "=== Step 1: Converting mesh ==="
cd constant
gmshToFoam lungMesh.msh
if [ $? -ne 0 ]; then
    echo "Mesh conversion failed!"
    exit 1
fi
cd ..

echo "=== Step 2: Checking mesh quality ==="
checkMesh > checkMesh.log 2>&1
grep -E "(Mesh OK|FAILED|cells|faces|points)" checkMesh.log

echo "=== Step 3: Initializing flow field ==="
foamRun -solver potentialFoam -func initialiseUBCs > potentialFoam.log 2>&1

echo "=== Step 4: Running CFD simulation ==="
foamRun -solver incompressibleFluid > simulation.log 2>&1 &
FOAM_PID=$!

# Monitor progress
echo "Simulation running (PID: $FOAM_PID)..."
sleep 5

# Show last few iterations
echo "=== Recent Progress ==="
tail -20 simulation.log

echo "=== Simulation Status ==="
if ps -p $FOAM_PID > /dev/null; then
    echo "Still running... Check simulation.log for progress"
    echo "To monitor: tail -f simulation.log"
else
    echo "Completed! Checking results..."
    ls -la [0-9]*/ 2>/dev/null && echo "Time directories created successfully"
fi

echo "=== Log files available ==="
ls -la *.log