#!/usr/bin/env python3
"""
Drug Delivery Presets System
Automated OpenFOAM case generation for 3 drug delivery methods:
1. MDI (Metered Dose Inhaler) - Slow deep breath
2. DPI (Dry Powder Inhaler) - Sharp forceful breath  
3. Nebulizer - Gentle normal breath
"""

import os
import shutil
import subprocess
from pathlib import Path
import time

class DrugDeliveryPresets:
    def __init__(self, base_mesh_path):
        """Initialize with path to .msh file"""
        self.base_mesh_path = Path(base_mesh_path)
        self.base_dir = self.base_mesh_path.parent
        self.mesh_name = self.base_mesh_path.stem
        
        # Define the 3 delivery methods
        self.delivery_methods = {
            'MDI': {
                'name': 'Metered Dose Inhaler',
                'description': 'Slow deep inhalation - medication sprayed with propellant',
                'velocity': 1.5,  # m/s
                'flow_rate': '30 L/min',
                'injection_type': 'cone',
                'particle_size': 3e-6,  # 3 microns
                'particle_density': 1000,  # kg/m³
                'injection_velocity': 30,  # m/s (high due to propellant)
                'duration': 0.1,  # seconds (short burst)
                'parcels_per_second': 10000,
                'cone_angle': 20
            },
            'DPI': {
                'name': 'Dry Powder Inhaler',
                'description': 'Sharp forceful inhalation - powder dispersed by airflow',
                'velocity': 4.5,  # m/s
                'flow_rate': '90 L/min',
                'injection_type': 'patchInjection',
                'particle_size': 4e-6,  # 4 microns
                'particle_density': 1400,  # kg/m³ (powder is denser)
                'injection_velocity': 4.5,  # Same as air velocity
                'duration': 0.5,  # seconds
                'parcels_per_second': 10000,
                'cone_angle': None
            },
            'Nebulizer': {
                'name': 'Nebulizer',
                'description': 'Gentle normal breath - fine mist continuously delivered',
                'velocity': 0.75,  # m/s
                'flow_rate': '15 L/min',
                'injection_type': 'patchInjection',
                'particle_size': 2.5e-6,  # 2.5 microns (finest droplets)
                'particle_density': 1000,  # kg/m³ (water-based)
                'injection_velocity': 1.0,  # Low velocity
                'duration': 5.0,  # seconds (long continuous)
                'parcels_per_second': 5000,
                'cone_angle': None
            }
        }
        
    def create_openfoam_case(self, method_key, case_dir):
        """Create complete OpenFOAM case for a delivery method"""
        method = self.delivery_methods[method_key]
        case_path = self.base_dir / case_dir
        
        print(f"🏗️  Creating {method['name']} case in {case_dir}")
        
        # Create directory structure
        self.create_case_structure(case_path)
        
        # Copy mesh
        shutil.copy2(self.base_mesh_path, case_path / 'constant' / 'polyMesh' / f'{self.mesh_name}.msh')
        
        # Generate configuration files
        self.write_velocity_field(case_path, method)
        self.write_pressure_field(case_path, method)
        self.write_particle_properties(case_path, method)
        self.write_control_dict(case_path, method)
        self.write_transport_properties(case_path)
        self.write_turbulence_properties(case_path)
        self.write_fv_schemes(case_path)
        self.write_fv_solution(case_path)
        
        print(f"✅ {method['name']} case ready!")
        
    def create_case_structure(self, case_path):
        """Create OpenFOAM case directory structure"""
        dirs = [
            '0',
            'constant/polyMesh',
            'system',
            'VTK'
        ]
        
        for d in dirs:
            (case_path / d).mkdir(parents=True, exist_ok=True)
            
    def write_velocity_field(self, case_path, method):
        """Write velocity boundary conditions (0/U file)"""
        u_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v11                                   |
|   \\  /    A nd           | Website:  www.OpenFOAM.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        // {method['description']} ({method['flow_rate']} -> ~{method['velocity']} m/s)
        value           uniform (0 0 {method['velocity']});
    }}

    outlets
    {{
        type            zeroGradient;
    }}

    wall
    {{
        type            noSlip;
    }}
}}

// ************************************************************************* //
"""
        with open(case_path / '0' / 'U', 'w') as f:
            f.write(u_content)
            
    def write_pressure_field(self, case_path, method):
        """Write pressure boundary conditions (0/p file)"""
        p_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v11                                   |
|   \\  /    A nd           | Website:  www.OpenFOAM.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}

    outlets
    {{
        type            fixedValue;
        value           uniform 0;
    }}

    wall
    {{
        type            zeroGradient;
    }}
}}

// ************************************************************************* //
"""
        with open(case_path / '0' / 'p', 'w') as f:
            f.write(p_content)
            
    def write_particle_properties(self, case_path, method):
        """Write particle injection properties"""
        if method['injection_type'] == 'cone':
            injection_config = f"""        cone
        {{
            type            cone;
            position        (0 0 10);    // Center of inlet patch
            direction       (0 0 1);     // Direction of injection
            coneAngle       {method['cone_angle']};
            U0              {method['injection_velocity']};
            duration        {method['duration']};
            parcelsPerSecond {method['parcels_per_second']};
            
            sizeDistribution
            {{
                type        logNormal;
                median      {method['particle_size']};
                sigma       0.4;
            }}
        }}"""
        else:  # patchInjection
            injection_config = f"""        patchInjection
        {{
            type            patchInjection;
            patchName       inlet;
            U0              {method['injection_velocity']};
            duration        {method['duration']};
            parcelsPerSecond {method['parcels_per_second']};
            
            sizeDistribution
            {{
                type        logNormal;
                median      {method['particle_size']};
                sigma       0.4;
            }}
        }}"""

        particle_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:  v11                                   |
|   \\  /    A nd           | Website:  www.OpenFOAM.com                      |
|    \\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      kinematicCloudProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

kinematicCloud
{{
    injectionModels
    (
{injection_config}
    );

    particleProperties
    {{
        rho             {method['particle_density']};
    }}
}}

// ************************************************************************* //
"""
        with open(case_path / 'constant' / 'kinematicCloudProperties', 'w') as f:
            f.write(particle_content)
            
    def write_control_dict(self, case_path, method):
        """Write simulation control parameters"""
        end_time = max(method['duration'] * 2, 1.0)  # Run longer than injection
        
        control_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     icoUncoupledKinematicParcelFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};

deltaT          0.01;
writeControl    timeStep;
writeInterval   10;
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""
        with open(case_path / 'system' / 'controlDict', 'w') as f:
            f.write(control_content)
            
    def write_transport_properties(self, case_path):
        """Write transport properties"""
        transport_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

nu              [0 2 -1 0 0 0 0] 1.5e-05;  // Air kinematic viscosity

// ************************************************************************* //
"""
        with open(case_path / 'constant' / 'transportProperties', 'w') as f:
            f.write(transport_content)
            
    def write_turbulence_properties(self, case_path):
        """Write turbulence model properties"""
        turb_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  laminar;

// ************************************************************************* //
"""
        with open(case_path / 'constant' / 'turbulenceProperties', 'w') as f:
            f.write(turb_content)
            
    def write_fv_schemes(self, case_path):
        """Write finite volume schemes"""
        schemes_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;
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

// ************************************************************************* //
"""
        with open(case_path / 'system' / 'fvSchemes', 'w') as f:
            f.write(schemes_content)
            
    def write_fv_solution(self, case_path):
        """Write solution control"""
        solution_content = """/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
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
    }

    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
}

PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}

// ************************************************************************* //
"""
        with open(case_path / 'system' / 'fvSolution', 'w') as f:
            f.write(solution_content)
            
    def create_all_presets(self):
        """Create all 3 delivery method cases"""
        print("🚀 Creating Drug Delivery Simulation Presets")
        print("=" * 60)
        
        for method_key in ['MDI', 'DPI', 'Nebulizer']:
            method = self.delivery_methods[method_key]
            case_dir = f"{self.mesh_name}_{method_key}_Case"
            
            print(f"\n📦 {method['name']} ({method_key})")
            print(f"   Flow Rate: {method['flow_rate']}")
            print(f"   Particle Size: {method['particle_size']*1e6:.1f} μm")
            print(f"   Delivery: {method['description']}")
            
            self.create_openfoam_case(method_key, case_dir)
            
        print(f"\n✅ All 3 delivery method cases created!")
        print(f"📁 Cases are in: {self.base_dir}")
        
        # Create run script
        self.create_run_all_script()
        
    def create_run_all_script(self):
        """Create script to run all simulations"""
        script_content = f"""#!/bin/bash
# Automated Drug Delivery Simulation Runner
# Runs all 3 delivery methods (MDI, DPI, Nebulizer)

echo "🫁 Starting Automated Drug Delivery Simulations"
echo "================================================"

BASE_DIR="{self.base_dir}"
MESH_NAME="{self.mesh_name}"

# Function to run simulation
run_simulation() {{
    local method=$1
    local case_dir="${{MESH_NAME}}_${{method}}_Case"
    
    echo ""
    echo "🚀 Running $method simulation..."
    echo "Case: $case_dir"
    
    cd "$BASE_DIR/$case_dir"
    
    # Convert mesh
    echo "📐 Converting mesh..."
    foamMeshToFluent -case . 2>/dev/null || echo "⚠️  Mesh conversion warning (continuing...)"
    
    # Run simulation
    echo "⚡ Running OpenFOAM solver..."
    icoUncoupledKinematicParcelFoam > "${{method}}_simulation.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $method simulation completed successfully!"
        echo "📊 Results saved in $case_dir/VTK/"
        echo "📝 Log: ${{method}}_simulation.log"
    else
        echo "❌ $method simulation failed - check log"
    fi
}}

# Run all 3 delivery methods
run_simulation "MDI"
run_simulation "DPI" 
run_simulation "Nebulizer"

echo ""
echo "🎯 All simulations completed!"
echo "📊 Ready for visualization and analysis"
echo ""
echo "Next steps:"
echo "1. Run heatmap generator for comparative analysis"
echo "2. Use ParaView to visualize particle trajectories"
echo "3. Generate publication-quality comparison plots"
"""

        script_path = self.base_dir / 'run_all_drug_delivery.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # Make executable
        os.chmod(script_path, 0o755)
        print(f"📜 Created run script: {script_path}")

def create_openfoam_cases(mesh_name):
    """
    Create OpenFOAM cases for all three delivery methods
    This function is used by the main pipeline
    """
    mesh_path = f"{mesh_name}.msh"
    if not os.path.exists(mesh_path):
        print(f"⚠️  Mesh file not found: {mesh_path}")
        return False
    
    try:
        generator = DrugDeliveryPresets(mesh_path)
        generator.create_all_presets()
        return True
    except Exception as e:
        print(f"❌ Case creation failed: {e}")
        return False

def main():
    """Main interface"""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python drug_delivery_presets.py <mesh_file.msh>")
        print("Example: python drug_delivery_presets.py lung.msh")
        return
        
    mesh_path = sys.argv[1]
    if not os.path.exists(mesh_path):
        print(f"❌ Mesh file not found: {mesh_path}")
        return
        
    print("🫁 Drug Delivery Simulation Preset Generator")
    print("=" * 50)
    print("Creates 3 different OpenFOAM cases for:")
    print("• MDI (Metered Dose Inhaler)")
    print("• DPI (Dry Powder Inhaler)")  
    print("• Nebulizer")
    print("")
    
    generator = DrugDeliveryPresets(mesh_path)
    generator.create_all_presets()

if __name__ == "__main__":
    main()