#!/usr/bin/env python3
"""
Lung Airways STL to MSH Converter
Converts STL files of lung airways to labeled volumetric meshes for OpenFOAM.
Handles non-watertight STL files with automatic repair.
"""

import argparse
import sys
import os
import numpy as np
import trimesh
import gmsh
from scipy.spatial import KDTree
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress trimesh warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

class LungMeshConverter:
    def __init__(self, input_stl, output_msh, mesh_size=2.0, repair_aggressive=True):
        """
        Initialize the converter.
        
        Args:
            input_stl: Path to input STL file
            output_msh: Path to output MSH file
            mesh_size: Target mesh element size (smaller = finer mesh)
            repair_aggressive: Use aggressive repair strategies
        """
        self.input_stl = input_stl
        self.output_msh = output_msh
        self.mesh_size = mesh_size
        self.repair_aggressive = repair_aggressive
        self.repaired_stl = None
        
    def load_and_repair_stl(self):
        """
        Load STL file and attempt to repair it.
        Returns the repaired mesh or raises an exception.
        """
        logger.info(f"Loading STL file: {self.input_stl}")
        
        try:
            # Load mesh with automatic processing
            mesh = trimesh.load_mesh(self.input_stl, process=True, validate=True)
            
            if not isinstance(mesh, trimesh.Trimesh):
                # If it's a scene, try to combine all geometries
                if hasattr(mesh, 'dump'):
                    meshes = mesh.dump()
                    if len(meshes) > 0:
                        mesh = trimesh.util.concatenate(meshes)
                    else:
                        raise ValueError("No valid geometry found in file")
                else:
                    raise ValueError("Unable to extract valid mesh from file")
            
            logger.info(f"Initial mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            logger.info(f"Initial watertight status: {mesh.is_watertight}")
            
            # Apply repair strategies
            if not mesh.is_watertight:
                logger.info("Mesh is not watertight. Attempting repairs...")
                mesh = self.repair_mesh(mesh)
            
            # Final validation
            if mesh.is_empty:
                raise ValueError("Mesh is empty after processing")
            
            # Save repaired mesh
            self.repaired_stl = self.input_stl.replace('.stl', '_repaired.stl')
            mesh.export(self.repaired_stl)
            logger.info(f"Repaired mesh saved to: {self.repaired_stl}")
            
            return mesh
            
        except Exception as e:
            logger.error(f"Failed to load/repair STL: {str(e)}")
            raise
    
    def repair_mesh(self, mesh):
        """
        Apply various repair strategies to fix non-watertight meshes.
        """
        original_watertight = mesh.is_watertight
        
        # Strategy 1: Remove duplicate and degenerate faces
        mesh.remove_duplicate_faces()
        mesh.remove_degenerate_faces()
        logger.info("Removed duplicate and degenerate faces")
        
        # Strategy 2: Fill holes
        if not mesh.is_watertight:
            mesh.fill_holes()
            logger.info("Attempted to fill holes")
        
        # Strategy 3: Fix normals
        mesh.fix_normals()
        logger.info("Fixed face normals")
        
        # Strategy 4: Remove unreferenced vertices
        mesh.remove_unreferenced_vertices()
        
        # Strategy 5: Aggressive repair using convex hull or voxelization
        if self.repair_aggressive and not mesh.is_watertight:
            logger.info("Applying aggressive repair strategies...")
            
            # Try voxelization and marching cubes
            try:
                # Calculate appropriate voxel pitch
                bounds = mesh.bounds
                max_dim = np.max(bounds[1] - bounds[0])
                pitch = max_dim / 100  # 100 voxels along longest dimension
                
                voxelized = mesh.voxelized(pitch=pitch)
                mesh = voxelized.as_boxes().marching_cubes
                logger.info("Applied voxelization repair")
            except:
                # Fall back to convex hull
                try:
                    mesh = mesh.convex_hull
                    logger.info("Applied convex hull as last resort")
                except:
                    logger.warning("All repair attempts failed")
        
        # Final check
        logger.info(f"Repair complete. Watertight: {original_watertight} -> {mesh.is_watertight}")
        logger.info(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        return mesh
    
    def find_inlet_boundary(self, mesh):
        """
        Identify the inlet (trachea opening) by finding the largest opening
        at the highest Z-coordinate.
        Returns approximate center and radius of inlet.
        """
        logger.info("Identifying inlet boundary...")
        
        # Get vertices at high Z values (top 10% of model)
        z_values = mesh.vertices[:, 2]
        z_threshold = np.percentile(z_values, 90)
        top_vertices_mask = z_values > z_threshold
        top_vertices = mesh.vertices[top_vertices_mask]
        
        if len(top_vertices) == 0:
            # Fallback: use highest points
            z_threshold = np.percentile(z_values, 75)
            top_vertices_mask = z_values > z_threshold
            top_vertices = mesh.vertices[top_vertices_mask]
        
        # Find boundary edges in the top region
        edges = mesh.edges_unique
        edge_points = mesh.vertices[edges]
        
        # Filter edges that have at least one vertex in top region
        top_edges_mask = np.any(edge_points[:, :, 2] > z_threshold, axis=1)
        top_edges = edges[top_edges_mask]
        
        # Cluster boundary points to find the main opening
        if len(top_edges) > 0:
            # Get unique vertices from top edges
            top_edge_vertices = np.unique(top_edges.flatten())
            boundary_points = mesh.vertices[top_edge_vertices]
            
            # Find center of the largest cluster (assumed to be inlet)
            # Use centroid of points in top region
            inlet_center = np.mean(boundary_points, axis=0)
            
            # Estimate inlet radius
            distances = np.linalg.norm(boundary_points - inlet_center, axis=1)
            inlet_radius = np.percentile(distances, 75)  # Use 75th percentile for robustness
        else:
            # Fallback: use top center
            inlet_center = np.array([
                np.mean(mesh.vertices[:, 0]),
                np.mean(mesh.vertices[:, 1]),
                np.max(mesh.vertices[:, 2])
            ])
            inlet_radius = np.max(mesh.bounds[1][:2] - mesh.bounds[0][:2]) / 4
        
        logger.info(f"Inlet identified at: {inlet_center}, radius: {inlet_radius:.2f}")
        return inlet_center, inlet_radius
    
    def create_volumetric_mesh(self):
        """
        Create a labeled volumetric mesh using gmsh.
        """
        logger.info("Creating volumetric mesh with gmsh...")
        
        # Initialize gmsh
        gmsh.initialize()
        gmsh.clear()
        
        # Set options for better mesh quality
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 4)  # Frontal-Delaunay
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.mesh_size * 0.5)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.mesh_size * 2.0)
        
        try:
            # Load repaired STL
            if not self.repaired_stl or not os.path.exists(self.repaired_stl):
                raise ValueError("Repaired STL file not found")
            
            # Import STL as a surface
            gmsh.merge(self.repaired_stl)
            
            # Create surface loop from all surfaces
            surfaces = gmsh.model.getEntities(2)  # Get all 2D entities (surfaces)
            if not surfaces:
                raise ValueError("No surfaces found in STL")
            
            surface_tags = [s[1] for s in surfaces]
            
            # Try to load the mesh for inlet identification
            try:
                mesh = trimesh.load_mesh(self.repaired_stl)
                inlet_center, inlet_radius = self.find_inlet_boundary(mesh)
                inlet_found = True
            except:
                logger.warning("Could not identify inlet automatically")
                inlet_found = False
                inlet_center = None
                inlet_radius = None
            
            # Create surface loop and volume
            surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
            volume = gmsh.model.geo.addVolume([surface_loop])
            
            gmsh.model.geo.synchronize()
            
            # Label surfaces based on location
            if inlet_found:
                inlet_surfaces = []
                wall_surfaces = []
                
                for surf_tag in surface_tags:
                    # Get bounding box of surface
                    bbox = gmsh.model.getBoundingBox(2, surf_tag)
                    surf_center = np.array([
                        (bbox[0] + bbox[3]) / 2,
                        (bbox[1] + bbox[4]) / 2,
                        (bbox[2] + bbox[5]) / 2
                    ])
                    
                    # Check if surface is near inlet location
                    dist_to_inlet = np.linalg.norm(surf_center - inlet_center)
                    
                    if dist_to_inlet < inlet_radius * 1.5 and surf_center[2] > inlet_center[2] - inlet_radius:
                        inlet_surfaces.append(surf_tag)
                    else:
                        wall_surfaces.append(surf_tag)
                
                # Create physical groups
                if inlet_surfaces:
                    gmsh.model.addPhysicalGroup(2, inlet_surfaces, tag=1)
                    gmsh.model.setPhysicalName(2, 1, "inlet")
                    logger.info(f"Labeled {len(inlet_surfaces)} surfaces as inlet")
                
                if wall_surfaces:
                    gmsh.model.addPhysicalGroup(2, wall_surfaces, tag=2)
                    gmsh.model.setPhysicalName(2, 2, "wall")
                    logger.info(f"Labeled {len(wall_surfaces)} surfaces as wall")
            else:
                # Fallback: label all surfaces as wall
                gmsh.model.addPhysicalGroup(2, surface_tags, tag=2)
                gmsh.model.setPhysicalName(2, 2, "wall")
                logger.warning("Could not identify inlet - all surfaces labeled as wall")
            
            # Add volume physical group
            gmsh.model.addPhysicalGroup(3, [volume], tag=3)
            gmsh.model.setPhysicalName(3, 3, "internal_volume")
            logger.info("Created internal volume")
            
            # Set mesh size
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)
            
            # Generate 3D mesh
            logger.info("Generating 3D mesh... (this may take a while)")
            gmsh.model.mesh.generate(3)
            
            # Optimize mesh
            logger.info("Optimizing mesh quality...")
            gmsh.model.mesh.optimize("Netgen")
            
            # Save mesh
            gmsh.write(self.output_msh)
            logger.info(f"Mesh saved to: {self.output_msh}")
            
            # Get statistics
            nodes = gmsh.model.mesh.getNodes()
            elements = gmsh.model.mesh.getElements()
            logger.info(f"Final mesh: {len(nodes[0])} nodes, {sum(len(e) for e in elements[2])} elements")
            
        except Exception as e:
            logger.error(f"Meshing failed: {str(e)}")
            raise
        finally:
            gmsh.finalize()
    
    def convert(self):
        """
        Run the complete conversion pipeline.
        """
        try:
            # Step 1: Load and repair STL
            mesh = self.load_and_repair_stl()
            
            # Step 2: Create volumetric mesh with labels
            self.create_volumetric_mesh()
            
            logger.info("Conversion completed successfully!")
            
            # Clean up temporary files (optional)
            if self.repaired_stl and os.path.exists(self.repaired_stl):
                logger.info(f"Keeping repaired STL for reference: {self.repaired_stl}")
            
            return True
            
        except Exception as e:
            logger.error(f"Conversion failed: {str(e)}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert lung airway STL files to labeled volumetric meshes for OpenFOAM"
    )
    parser.add_argument("input_stl", help="Path to input STL file")
    parser.add_argument("output_msh", help="Path to output MSH file")
    parser.add_argument("--mesh-size", type=float, default=2.0,
                        help="Target mesh element size (default: 2.0)")
    parser.add_argument("--no-aggressive-repair", action="store_true",
                        help="Disable aggressive repair strategies")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input_stl):
        logger.error(f"Input file not found: {args.input_stl}")
        sys.exit(1)
    
    # Create converter
    converter = LungMeshConverter(
        input_stl=args.input_stl,
        output_msh=args.output_msh,
        mesh_size=args.mesh_size,
        repair_aggressive=not args.no_aggressive_repair
    )
    
    # Run conversion
    success = converter.convert()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()