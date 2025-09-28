#!/usr/bin/env python3
"""
Simplified Robust Lung Mesh Generator
Focuses on reliability over advanced features
"""

import argparse
import sys
import os
import math
import numpy as np
import trimesh
import gmsh
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_lung_mesh(input_stl, case_dir="lungCaseFixed", mesh_size=3.0):
    """
    Simple, robust lung mesh generation
    """
    case_path = Path(case_dir)
    
    try:
        # Step 1: Basic STL validation
        logger.info(f"Loading STL: {input_stl}")
        mesh = trimesh.load_mesh(input_stl)
        
        if hasattr(mesh, 'is_empty') and mesh.is_empty:
            raise ValueError("Empty mesh")
            
        logger.info(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Step 2: Basic cleanup only
        if hasattr(mesh, 'update_faces'):
            mesh.update_faces(mesh.unique_faces())
        if hasattr(mesh, 'remove_unreferenced_vertices'):
            mesh.remove_unreferenced_vertices()
            
        # Step 3: Simple gmsh meshing
        gmsh.initialize()
        gmsh.clear()
        
        # Conservative settings
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D
        
        # Size control
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size * 2.0)
        
        # Load STL directly
        logger.info("Loading STL into gmsh...")
        gmsh.merge(input_stl)
        
        # Simple approach - no surface classification
        logger.info("Creating volume...")
        surfaces = gmsh.model.getEntities(2)
        
        if not surfaces:
            raise ValueError("No surfaces found in STL")
            
        # Create surface loop and volume
        surface_ids = [s[1] for s in surfaces]
        surface_loop = gmsh.model.geo.addSurfaceLoop(surface_ids)
        volume = gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()
        
        # Physical groups
        gmsh.model.addPhysicalGroup(2, surface_ids, 1)
        gmsh.model.setPhysicalName(2, 1, "wall")
        gmsh.model.addPhysicalGroup(3, [volume], 2)
        gmsh.model.setPhysicalName(3, 2, "internal")
        
        # Generate mesh
        logger.info("Generating mesh...")
        gmsh.model.mesh.generate(3)
        
        # Simple optimization
        logger.info("Basic optimization...")
        gmsh.model.mesh.optimize("Netgen", niter=3)
        
        # Save mesh
        case_path.mkdir(exist_ok=True)
        (case_path / "constant").mkdir(exist_ok=True)
        mesh_file = case_path / "constant" / "lungMesh.msh"
        
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(mesh_file))
        
        # Stats
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()
        
        logger.info(f"Mesh created: {len(nodes[0])} nodes")
        
        gmsh.finalize()
        
        # Create basic OpenFOAM case
        create_openfoam_case(case_path)
        
        logger.info(f"Complete case created in: {case_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        if 'gmsh' in globals():
            gmsh.finalize()
        return False

def create_openfoam_case(case_path):
    """Create basic OpenFOAM case structure"""
    
    # Create directories
    (case_path / "0").mkdir(exist_ok=True)
    (case_path / "system").mkdir(exist_ok=True)
    (case_path / "constant").mkdir(exist_ok=True)
    
    # controlDict
    with open(case_path / "system" / "controlDict", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     foamRun;
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         100;
deltaT          1;

writeControl    timeStep;
writeInterval   20;
writeFormat     ascii;
""")

    # fvSolution with pressure reference
    with open(case_path / "system" / "fvSolution", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.1;
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 2;
    
    pRefCell        0;
    pRefValue       0;
    
    residualControl
    {
        p               1e-4;
        U               1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
    }
}
""")

    # fvSchemes
    with open(case_path / "system" / "fvSchemes", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
""")

    # Initial conditions
    with open(case_path / "0" / "U", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0.1 0 0);

boundaryField
{
    wall
    {
        type            noSlip;
    }
}
""")

    with open(case_path / "0" / "p", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    wall
    {
        type            zeroGradient;
    }
}
""")

    # Transport properties
    with open(case_path / "constant" / "transportProperties", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1.5e-05;
""")
    
    # Turbulence properties
    with open(case_path / "constant" / "turbulenceProperties", "w") as f:
        f.write("""
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  laminar;
""")

    logger.info("OpenFOAM case files created")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Lung Mesh Generator')
    parser.add_argument('stl_file', help='Input STL file')
    parser.add_argument('--case-dir', default='lungCaseFixed', help='Output case directory')
    parser.add_argument('--mesh-size', type=float, default=3.0, help='Mesh size')
    
    args = parser.parse_args()
    
    success = simple_lung_mesh(args.stl_file, args.case_dir, args.mesh_size)
    sys.exit(0 if success else 1)