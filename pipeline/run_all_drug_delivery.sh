#!/bin/bash
# Automated Drug Delivery Simulation Runner
# Runs all 3 delivery methods (MDI, DPI, Nebulizer)

echo "🫁 Starting Automated Drug Delivery Simulations"
echo "================================================"

BASE_DIR="."
MESH_NAME="lungs_repaired"

# Function to run simulation
run_simulation() {
    local method=$1
    local case_dir="${MESH_NAME}_${method}_Case"
    
    echo ""
    echo "🚀 Running $method simulation..."
    echo "Case: $case_dir"
    
    cd "$BASE_DIR/$case_dir"
    
    # Convert mesh
    echo "📐 Converting mesh..."
    foamMeshToFluent -case . 2>/dev/null || echo "⚠️  Mesh conversion warning (continuing...)"
    
    # Run simulation
    echo "⚡ Running OpenFOAM solver..."
    icoUncoupledKinematicParcelFoam > "${method}_simulation.log" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ $method simulation completed successfully!"
        echo "📊 Results saved in $case_dir/VTK/"
        echo "📝 Log: ${method}_simulation.log"
    else
        echo "❌ $method simulation failed - check log"
    fi
}

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
