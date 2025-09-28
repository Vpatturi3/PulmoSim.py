#!/bin/bash
source /opt/openfoam11/etc/bashrc
cd /home/openfoam/case

echo "Starting OpenFOAM diagnostics..." > /home/openfoam/case/diagnostics.log
echo "Current directory: $(pwd)" >> /home/openfoam/case/diagnostics.log
echo "Files present:" >> /home/openfoam/case/diagnostics.log
ls -la >> /home/openfoam/case/diagnostics.log

echo "Running checkMesh..." >> /home/openfoam/case/diagnostics.log
checkMesh >> /home/openfoam/case/diagnostics.log 2>&1

echo "CheckMesh complete, status: $?" >> /home/openfoam/case/diagnostics.log

if [ $? -eq 0 ]; then
    echo "Mesh OK, running potentialFoam..." >> /home/openfoam/case/diagnostics.log
    potentialFoam -initialiseUBCs >> /home/openfoam/case/diagnostics.log 2>&1
    
    echo "PotentialFoam complete, status: $?" >> /home/openfoam/case/diagnostics.log
    
    echo "Running simpleFoam for 10 iterations..." >> /home/openfoam/case/diagnostics.log
    simpleFoam >> /home/openfoam/case/diagnostics.log 2>&1
    
    echo "SimpleFoam complete, status: $?" >> /home/openfoam/case/diagnostics.log
else
    echo "Mesh check failed!" >> /home/openfoam/case/diagnostics.log
fi

echo "Final directory listing:" >> /home/openfoam/case/diagnostics.log
ls -la >> /home/openfoam/case/diagnostics.log