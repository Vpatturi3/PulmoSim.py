#!/usr/bin/env python3
"""
Real-Time Interactive Lung Drug Delivery Simulator
3D visualization with animated drug particles and breathing lung model
"""

import numpy as np
import pyvista as pv
import time
import threading
from scipy.spatial import cKDTree

class InteractiveLungSimulator:
    def __init__(self):
        """Initialize the real-time lung drug delivery simulator"""
        self.setup_lung_geometry()
        self.setup_drug_particles()
        self.setup_physics()
        self.setup_visualization()
        
    def setup_lung_geometry(self):
        """Load and prepare lung 3D model"""
        try:
            # Try to load the lung STL file
            self.lung_mesh = pv.read('lungs_repaired.stl')
            print("✅ Loaded lung geometry from STL")
        except:
            # Create a simplified lung-like geometry
            print("⚠️  STL not found, creating simplified lung geometry")
            self.lung_mesh = self.create_simplified_lung()
            
        # Center and scale the lung
        center = self.lung_mesh.center
        self.lung_mesh.translate([-center[0], -center[1], -center[2]], inplace=True)
        self.lung_mesh.scale(0.1, inplace=True)  # Scale to reasonable size
        
        # Create breathing animation properties
        self.original_points = self.lung_mesh.points.copy()
        self.breathing_phase = 0.0
        self.breathing_rate = 0.5  # Hz (breaths per second)
        
    def create_simplified_lung(self):
        """Create a simplified lung-like geometry for demonstration"""
        # Create main bronchi structure
        cylinder1 = pv.Cylinder(center=[0, 0, 2], direction=[0, 0, 1], 
                               radius=0.3, height=4)
        
        # Left lung lobe
        ellipsoid1 = pv.Sphere(center=[-1.5, 0, -1], radius=2)
        ellipsoid1.scale([1, 0.8, 1.2], inplace=True)
        
        # Right lung lobe  
        ellipsoid2 = pv.Sphere(center=[1.5, 0, -1], radius=1.8)
        ellipsoid2.scale([1, 0.8, 1.3], inplace=True)
        
        # Combine geometries
        lung = cylinder1 + ellipsoid1 + ellipsoid2
        return lung.clean()
        
    def setup_drug_particles(self):
        """Initialize drug particle system"""
        self.num_particles = 1000
        
        # Initialize particle positions at trachea inlet
        inlet_center = np.array([0, 0, 4])
        inlet_radius = 0.2
        
        # Create particles in spherical distribution around inlet
        theta = np.random.uniform(0, 2*np.pi, self.num_particles)
        phi = np.random.uniform(0, np.pi, self.num_particles)
        r = np.random.uniform(0, inlet_radius, self.num_particles)
        
        x = inlet_center[0] + r * np.sin(phi) * np.cos(theta)
        y = inlet_center[1] + r * np.sin(phi) * np.sin(theta)
        z = inlet_center[2] + r * np.cos(phi)
        
        self.particle_positions = np.column_stack([x, y, z])
        
        # Particle properties
        self.particle_velocities = np.zeros_like(self.particle_positions)
        self.particle_ages = np.zeros(self.num_particles)
        self.max_particle_age = 10.0  # seconds
        
        # Drug particle visualization properties
        self.particle_colors = np.random.rand(self.num_particles)
        
    def setup_physics(self):
        """Configure physics simulation parameters"""
        self.gravity = np.array([0, 0, -0.5])  # Weak gravity
        self.air_resistance = 0.1
        self.turbulence_strength = 0.05
        self.dt = 0.02  # 50 FPS
        
        # Simplified flow field (based on your CFD results)
        self.create_flow_field()
        
    def create_flow_field(self):
        """Create a simplified 3D flow field based on CFD results"""
        # Create a grid for the flow field
        x = np.linspace(-3, 3, 20)
        y = np.linspace(-3, 3, 20)
        z = np.linspace(-2, 4, 30)
        
        self.flow_grid_x, self.flow_grid_y, self.flow_grid_z = np.meshgrid(x, y, z)
        
        # Simplified flow: inward toward lung center, downward component
        center = np.array([0, 0, -1])
        
        flow_x = -(self.flow_grid_x - center[0]) * 0.2
        flow_y = -(self.flow_grid_y - center[1]) * 0.2  
        flow_z = -(self.flow_grid_z - center[2]) * 0.3 - 0.5  # Downward + toward center
        
        # Add turbulence
        flow_x += np.random.normal(0, 0.1, flow_x.shape)
        flow_y += np.random.normal(0, 0.1, flow_y.shape)
        flow_z += np.random.normal(0, 0.1, flow_z.shape)
        
        self.flow_field = np.stack([flow_x.flatten(), flow_y.flatten(), flow_z.flatten()], axis=1)
        self.flow_positions = np.stack([self.flow_grid_x.flatten(), 
                                      self.flow_grid_y.flatten(), 
                                      self.flow_grid_z.flatten()], axis=1)
        
        # Create KDTree for fast interpolation
        self.flow_tree = cKDTree(self.flow_positions)
        
    def interpolate_flow_velocity(self, position):
        """Get flow velocity at a given position using nearest neighbor"""
        _, idx = self.flow_tree.query(position)
        return self.flow_field[idx]
        
    def update_particles(self):
        """Update particle positions and properties"""
        for i in range(self.num_particles):
            pos = self.particle_positions[i]
            
            # Get local flow velocity
            flow_vel = self.interpolate_flow_velocity(pos)
            
            # Add turbulence
            turbulence = np.random.normal(0, self.turbulence_strength, 3)
            
            # Update velocity (simplified physics)
            self.particle_velocities[i] = (
                flow_vel * 0.8 +  # Flow following
                self.gravity * 0.1 +  # Gravity
                turbulence +  # Random motion
                -self.particle_velocities[i] * self.air_resistance  # Damping
            )
            
            # Update position
            self.particle_positions[i] += self.particle_velocities[i] * self.dt
            
            # Update age
            self.particle_ages[i] += self.dt
            
            # Reset old particles
            if self.particle_ages[i] > self.max_particle_age:
                self.reset_particle(i)
                
    def reset_particle(self, idx):
        """Reset a particle to the inlet"""
        inlet_center = np.array([0, 0, 4])
        inlet_radius = 0.2
        
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi) 
        r = np.random.uniform(0, inlet_radius)
        
        x = inlet_center[0] + r * np.sin(phi) * np.cos(theta)
        y = inlet_center[1] + r * np.sin(phi) * np.sin(theta)
        z = inlet_center[2] + r * np.cos(phi)
        
        self.particle_positions[idx] = [x, y, z]
        self.particle_velocities[idx] = [0, 0, -0.5]  # Initial downward velocity
        self.particle_ages[idx] = 0.0
        
    def animate_breathing(self):
        """Animate lung breathing motion"""
        self.breathing_phase += self.breathing_rate * self.dt * 2 * np.pi
        
        # Breathing expansion factor (1.0 = normal, 1.1 = inhaled)
        expansion_factor = 1.0 + 0.1 * np.sin(self.breathing_phase)
        
        # Apply breathing motion
        center = self.lung_mesh.center
        expanded_points = (self.original_points - center) * expansion_factor + center
        self.lung_mesh.points = expanded_points
        
    def setup_visualization(self):
        """Setup PyVista plotter for real-time rendering"""
        self.plotter = pv.Plotter()
        self.plotter.set_background('black')
        
        # Add lung mesh
        self.lung_actor = self.plotter.add_mesh(
            self.lung_mesh, 
            color='pink', 
            opacity=0.3,
            smooth_shading=True,
            name='lung'
        )
        
        # Create particle mesh
        self.particle_mesh = pv.PolyData(self.particle_positions)
        self.particle_mesh.point_data['colors'] = self.particle_colors
        
        # Add particles  
        self.particle_actor = self.plotter.add_mesh(
            self.particle_mesh,
            scalars='colors',
            point_size=8.0,
            render_points_as_spheres=True,
            cmap='hot',
            name='particles'
        )
        
        # Add flow vectors (sample)
        sample_indices = np.arange(0, len(self.flow_positions), 50)
        sample_positions = self.flow_positions[sample_indices]
        sample_vectors = self.flow_field[sample_indices] * 2  # Scale for visibility
        
        self.plotter.add_arrows(
            sample_positions, 
            sample_vectors,
            mag=0.5,
            color='cyan',
            opacity=0.6,
            name='flow_field'
        )
        
        # Setup camera
        self.plotter.camera_position = [(8, 8, 8), (0, 0, 0), (0, 0, 1)]
        
        # Add title and info
        self.plotter.add_title("🫁 Real-Time Lung Drug Delivery Simulator", font_size=16)
        
        print("🎮 Controls:")
        print("   - Mouse: Rotate view")
        print("   - Scroll: Zoom")
        print("   - 'q': Quit")
        
    def run_simulation(self):
        """Run the real-time simulation"""
        print("🚀 Starting real-time drug delivery simulation...")
        
        # Animation callback
        def update_scene():
            self.update_particles()
            self.animate_breathing()
            
            # Update particle mesh
            self.particle_mesh.points = self.particle_positions
            
            # Update colors based on age (newer = brighter)
            age_colors = 1.0 - (self.particle_ages / self.max_particle_age)
            self.particle_mesh.point_data['colors'] = age_colors
            
        # Start animation timer  
        self.plotter.add_timer_event(duration=int(self.dt * 1000), callback=update_scene)
        
        # Show the simulation
        self.plotter.show(auto_close=False, interactive=True)
        
def main():
    """Main function to run the interactive lung simulator"""
    print("🫁 Interactive Lung Drug Delivery Simulator")
    print("=" * 50)
    
    try:
        simulator = InteractiveLungSimulator()
        simulator.run_simulation()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try installing missing packages: pip install pyvista scipy")

if __name__ == "__main__":
    main()