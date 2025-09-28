#!/usr/bin/env python3
"""
Advanced Lung Mesh Repair Pipeline with OpenFOAM Setup
Fixes mesh quality issues and prepares complete OpenFOAM case
"""

import argparse
import sys
import os
import math
import numpy as np
import trimesh
import gmsh
import shutil
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedLungMeshProcessor:
    def __init__(self, input_stl, case_dir="lungCase", mesh_size=2.0):
        self.input_stl = input_stl
        self.case_dir = Path(case_dir)
        self.mesh_size = mesh_size
        
    def repair_stl_advanced(self):
        """
        Advanced STL repair with multiple strategies
        """
        logger.info("Starting advanced STL repair...")
        
        # Load mesh
        mesh = trimesh.load_mesh(self.input_stl, process=True)
        
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, 'dump'):
                meshes = mesh.dump()
                mesh = trimesh.util.concatenate(meshes) if meshes else None
        
        if mesh is None or mesh.is_empty:
            raise ValueError("Failed to load valid mesh")
        
        logger.info(f"Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Step 1: Basic cleaning
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        mesh.remove_unreferenced_vertices()
        
        # Step 2: Smooth and regularize
        if not mesh.is_watertight:
            logger.info("Filling holes...")
            # Fill holes to make watertight
            mesh.fill_holes()
        
        # Step 3: Remesh for quality
        logger.info("Remeshing for quality...")
        # Subdivide large faces
        max_face_area = np.percentile([mesh.area_faces], 95)
        mesh = mesh.subdivide_to_size(max_edge=np.sqrt(max_face_area), max_iter=2)
        
        # Step 4: Final repair
        mesh.fix_normals()
        
        # Save repaired STL
        repaired_path = str(Path(self.input_stl).with_suffix('.repaired.stl'))
        mesh.export(repaired_path)
        logger.info(f"Repaired STL saved to: {repaired_path}")
        
        return repaired_path, mesh
    
    def create_quality_mesh(self, stl_path):
        """
        Create high-quality mesh with careful parameter tuning
        """
        logger.info("Creating quality mesh with gmsh...")
        
        gmsh.initialize()
        gmsh.clear()
        
        # Critical: Set options for mesh quality
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("General.Verbosity", 2)
        
        # Geometry tolerance - crucial for broken STLs
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
        gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
        gmsh.option.setNumber("Geometry.MatchMeshTolerance", 1e-6)
        
        # Mesh algorithm selection for quality
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay 2D
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D
        gmsh.option.setNumber("Mesh.RecombineAll", 0)  # Keep tets for now
        
        # Quality thresholds
        gmsh.option.setNumber("Mesh.QualityType", 2)  # SICN quality
        gmsh.option.setNumber("Mesh.OptimizeThreshold", 0.4)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        
        # Size field parameters
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size * 0.3)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size * 3.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumCircleNodes", 12)
        
        # Load STL
        gmsh.merge(stl_path)
        
        # Classify surfaces
        gmsh.model.mesh.classifySurfaces(math.pi/4, True, True)
        gmsh.model.mesh.createGeometry()
        
        # Get entities
        surfaces = gmsh.model.getEntities(2)
        volumes = gmsh.model.getEntities(3)
        
        if not volumes:
            # Create volume from surfaces
            logger.info("Creating volume from surfaces...")
            surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
            volume = gmsh.model.geo.addVolume([surface_loop])
            gmsh.model.geo.synchronize()
        else:
            volume = volumes[0][1]
        
        # Add size fields for gradual refinement
        logger.info("Setting up mesh size fields...")
        
        # Distance field from walls
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", [s[1] for s in surfaces])
        
        # Threshold field for size gradation
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", self.mesh_size * 0.5)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", self.mesh_size * 2.0)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0.5)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 10.0)
        
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
        
        # Physical groups for OpenFOAM
        all_surfaces = [s[1] for s in surfaces]
        gmsh.model.addPhysicalGroup(2, all_surfaces, 1)
        gmsh.model.setPhysicalName(2, 1, "wall")
        
        gmsh.model.addPhysicalGroup(3, [volume], 2)
        gmsh.model.setPhysicalName(3, 2, "internal")
        
        # Generate mesh with quality control
        logger.info("Generating 3D mesh...")
        gmsh.model.mesh.generate(3)
        
        # Optimize iteratively
        logger.info("Optimizing mesh quality...")
        for i in range(5):
            gmsh.model.mesh.optimize("Netgen")
            gmsh.model.mesh.optimize("Relocate3D")
            
            # Check quality
            _, element_tags = gmsh.model.mesh.getElementsByType(4)  # Tetrahedra
            if len(element_tags) > 0:
                qualities = []
                for tag in element_tags[:min(100, len(element_tags))]:
                    q = gmsh.model.mesh.getElementQuality(tag, "SICN")
                    qualities.append(q)
                
                min_q = min(qualities) if qualities else 0
                logger.info(f"  Iteration {i+1}: Min quality = {min_q:.3f}")
                
                if min_q > 0.3:  # Good enough
                    break
        
        # Save mesh
        mesh_path = self.case_dir / "constant" / "lungMesh.msh"
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  # OpenFOAM compatible
        gmsh.write(str(mesh_path))
        
        # Get statistics
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()
        
        stats = {
            "nodes": len(nodes[0]),
            "elements": sum(len(e) for e in elements[2]),
            "surfaces": len(surfaces),
            "volumes": 1
        }
        
        gmsh.finalize()
        
        logger.info(f"Mesh saved: {stats['nodes']} nodes, {stats['elements']} elements")
        return mesh_path, stats
    
    def setup_openfoam_case(self):
        """
        Create complete OpenFOAM case structure with proper configuration
        """
        logger.info("Setting up OpenFOAM case...")
        
        # Create directory structure
        dirs = [
            self.case_dir / "0",
            self.case_dir / "constant",
            self.case_dir / "system",
            self.case_dir / "constant" / "polyMesh"
        ]
        
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        # Create boundary conditions (0 directory)
        self._create_file(self.case_dir / "0" / "U", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volVectorField;
    object      U;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    wall
    {
        type            fixedValue;
        value           uniform (0 0 -0.1);  // Inlet velocity at top
    }
}

// ************************************************************************* //""")
        
        self._create_file(self.case_dir / "0" / "p", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       volScalarField;
    object      p;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    wall
    {
        type            zeroGradient;
    }
}

// ************************************************************************* //""")
        
        # Create transport properties
        self._create_file(self.case_dir / "constant" / "transportProperties", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

transportModel  Newtonian;

nu              1.5e-05;  // Air at room temperature

// ************************************************************************* //""")
        
        # Create turbulence properties (laminar for now)
        self._create_file(self.case_dir / "constant" / "momentumTransport", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "constant";
    object      momentumTransport;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  laminar;

// ************************************************************************* //""")
        
        # Create controlDict
        self._create_file(self.case_dir / "system" / "controlDict", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     foamRun;

solver          incompressibleFluid;

startFrom       latestTime;

startTime       0;

stopAt          endTime;

endTime         100;

deltaT          0.01;

writeControl    timeStep;

writeInterval   10;

purgeWrite      0;

writeFormat     binary;

writePrecision  6;

writeCompression off;

timeFormat      general;

timePrecision   6;

runTimeModifiable true;

adjustTimeStep  yes;

maxCo           0.9;

// ************************************************************************* //""")
        
        # Create fvSchemes
        self._create_file(self.case_dir / "system" / "fvSchemes", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited corrected 0.5;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited corrected 0.5;
}

// ************************************************************************* //""")
        
        # Create fvSolution with pressure reference
        self._create_file(self.case_dir / "system" / "fvSolution", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
        nCellsInCoarsestLevel 20;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.1;
    }

    UFinal
    {
        $U;
        relTol          0;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 2;
    residualControl
    {
        p               1e-5;
        U               1e-5;
    }
    
    // CRITICAL: Set pressure reference point
    pRefCell        0;      // Use cell 0 as reference
    pRefValue       0;      // Reference pressure value
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

// ************************************************************************* //""")
        
        # Create mesh quality dict
        self._create_file(self.case_dir / "system" / "meshQualityDict", """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  11                                     |
|   \\\\  /    A nd           | Website:  https://openfoam.org                  |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      meshQualityDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Relaxed quality criteria for complex geometries
maxNonOrthogonality 75;
maxBoundarySkewness 20;
maxInternalSkewness 8;
maxConcave 80;
minVol 1e-20;
minTetQuality -1e30;
minArea -1;
minTwist 0.01;
minDeterminant 0.001;
minFaceWeight 0.01;
minVolRatio 0.01;
minTriangleTwist -1;
nSmoothScale 4;
errorReduction 0.75;

// ************************************************************************* //""")
        
        logger.info(f"OpenFOAM case setup complete in {self.case_dir}")
    
    def _create_file(self, path, content):
        """Helper to create a file with content"""
        with open(path, 'w') as f:
            f.write(content)
    
    def convert_mesh_to_foam(self, msh_path):
        """
        Convert MSH to OpenFOAM format using gmshToFoam
        """
        logger.info("Converting mesh to OpenFOAM format...")
        
        # Create conversion script
        convert_script = self.case_dir / "convert.sh"
        with open(convert_script, 'w') as f:
            f.write(f"""#!/bin/bash
cd {self.case_dir}
gmshToFoam {msh_path}
checkMesh -allTopology -allGeometry > checkMesh.log 2>&1
echo "Mesh conversion complete. Check checkMesh.log for quality report."
""")
        
        convert_script.chmod(0o755)
        logger.info(f"Run '{convert_script}' after sourcing OpenFOAM to convert the mesh")
        
        return convert_script
    
    def process(self):
        """
        Complete processing pipeline
        """
        try:
            # Step 1: Repair STL
            repaired_stl, mesh_data = self.repair_stl_advanced()
            
            # Step 2: Create quality mesh
            mesh_path, stats = self.create_quality_mesh(repaired_stl)
            
            # Step 3: Setup OpenFOAM case
            self.setup_openfoam_case()
            
            # Step 4: Prepare conversion
            convert_script = self.convert_mesh_to_foam(mesh_path)
            
            logger.info("="*60)
            logger.info("Processing complete! Next steps:")
            logger.info("1. Source OpenFOAM environment:")
            logger.info("   source /opt/openfoam11/etc/bashrc")
            logger.info(f"2. Navigate to case directory:")
            logger.info(f"   cd {self.case_dir}")
            logger.info("3. Convert mesh:")
            logger.info(f"   ./convert.sh")
            logger.info("4. If mesh has issues, run:")
            logger.info("   foamMeshHealer")
            logger.info("5. Run simulation:")
            logger.info("   foamRun")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Advanced lung mesh processor for OpenFOAM")
    parser.add_argument("input_stl", help="Input STL file")
    parser.add_argument("--case-dir", default="lungCase", help="OpenFOAM case directory")
    parser.add_argument("--mesh-size", type=float, default=2.0, help="Target mesh size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_stl):
        logger.error(f"Input file not found: {args.input_stl}")
        sys.exit(1)
    
    processor = AdvancedLungMeshProcessor(
        args.input_stl,
        args.case_dir,
        args.mesh_size
    )
    
    success = processor.process()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()