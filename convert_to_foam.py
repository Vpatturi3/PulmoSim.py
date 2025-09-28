import gmsh
import os
import sys

def convert_to_foam(mesh_file):
    """
    Convert the mesh to OpenFOAM format directly using Gmsh
    """
    # Initialize gmsh
    gmsh.initialize()
    
    # Read the mesh
    gmsh.open(mesh_file)
    
    # Set up OpenFOAM export options
    gmsh.option.setString("Mesh.Format", "msh2")  # Use MSH2 format which is more compatible
    
    # Save as OpenFOAM format
    foam_dir = "lungCase"
    if not os.path.exists(foam_dir):
        os.makedirs(foam_dir)
    
    foam_file = os.path.join(foam_dir, "my_lung_mesh_quality.foam")
    gmsh.write(foam_file)
    
    # Finalize gmsh
    gmsh.finalize()
    
    print(f"Mesh converted and saved to OpenFOAM format in {foam_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mesh_file = sys.argv[1]
    else:
        mesh_file = "my_lung_mesh_quality.msh"
    
    if not os.path.exists(mesh_file):
        print(f"Error: {mesh_file} not found")
        sys.exit(1)
    
    convert_to_foam(mesh_file)