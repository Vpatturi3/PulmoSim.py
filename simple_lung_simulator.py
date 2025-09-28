#!/usr/bin/env python3
"""
Real-Time Interactive Lung Drug Delivery Simulator
Simplified version with working PyVista animation
"""

import numpy as np
import pyvista as pv
import time

class SimpleLungSimulator:
    def __init__(self):
        """Initialize the real-time lung drug delivery simulator"""
        self.setup_lung_geometry()
        self.setup_drug_particles()
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
        self.lung_mesh.scale(0.01, inplace=True)  # Scale to reasonable size for viewing
        
    def create_simplified_lung(self):
        """Create a simplified lung-like geometry"""
        # Create main trachea
        trachea = pv.Cylinder(center=[0, 0, 2], direction=[0, 0, 1], 
                             radius=0.5, height=4)
        
        # Left lung lobe
        left_lung = pv.Sphere(center=[-2, 0, -1], radius=3)
        left_lung.scale([1, 0.8, 1.2], inplace=True)
        
        # Right lung lobe  
        right_lung = pv.Sphere(center=[2, 0, -1], radius=2.8)
        right_lung.scale([1, 0.8, 1.3], inplace=True)
        
        # Combine all parts
        lung = trachea + left_lung + right_lung
        return lung.clean()
        
    def setup_drug_particles(self):
        """Initialize drug particle system"""
        self.num_particles = 500
        
        # Initialize particle positions at trachea inlet
        inlet_center = np.array([0, 0, 4])
        
        # Create particles randomly around inlet
        self.particle_positions = []
        for i in range(self.num_particles):
            # Random position near inlet
            x = inlet_center[0] + np.random.normal(0, 0.3)
            y = inlet_center[1] + np.random.normal(0, 0.3) 
            z = inlet_center[2] + np.random.uniform(-0.5, 0.5)
            
            self.particle_positions.append([x, y, z])
            
        self.particle_positions = np.array(self.particle_positions)
        
        # Particle properties
        self.particle_velocities = np.random.normal(0, 0.1, (self.num_particles, 3))
        self.particle_velocities[:, 2] -= 1.0  # Initial downward velocity
        
        # Colors based on particle type/age
        self.particle_colors = np.random.rand(self.num_particles)
        
    def update_particles(self):
        """Update particle positions with simple physics"""
        dt = 0.05
        
        for i in range(self.num_particles):
            # Simple gravity and air resistance
            gravity = np.array([0, 0, -2.0])
            air_resistance = -0.1 * self.particle_velocities[i]
            
            # Add some turbulence
            turbulence = np.random.normal(0, 0.2, 3)
            
            # Update velocity
            acceleration = gravity + air_resistance + turbulence
            self.particle_velocities[i] += acceleration * dt
            
            # Update position
            self.particle_positions[i] += self.particle_velocities[i] * dt
            
            # Reset particle if it goes too far
            if (np.linalg.norm(self.particle_positions[i]) > 10 or 
                self.particle_positions[i][2] < -8):
                self.reset_particle(i)
                
    def reset_particle(self, idx):
        """Reset a particle to the inlet"""
        inlet_center = np.array([0, 0, 4])
        
        # Random position near inlet
        x = inlet_center[0] + np.random.normal(0, 0.3)
        y = inlet_center[1] + np.random.normal(0, 0.3)
        z = inlet_center[2] + np.random.uniform(-0.5, 0.5)
        
        self.particle_positions[idx] = [x, y, z]
        self.particle_velocities[idx] = [
            np.random.normal(0, 0.1), 
            np.random.normal(0, 0.1), 
            -1.0 + np.random.normal(0, 0.2)
        ]
        
    def setup_visualization(self):
        """Setup PyVista plotter"""
        self.plotter = pv.Plotter()
        self.plotter.set_background('black')
        
        # Add lung mesh
        self.plotter.add_mesh(
            self.lung_mesh, 
            color='lightcoral', 
            opacity=0.4,
            smooth_shading=True
        )
        
        # Create initial particle mesh
        self.particle_mesh = pv.PolyData(self.particle_positions)
        self.particle_mesh.point_data['velocity_mag'] = np.linalg.norm(self.particle_velocities, axis=1)
        
        # Add particles
        self.particle_actor = self.plotter.add_mesh(
            self.particle_mesh,
            scalars='velocity_mag',
            point_size=12.0,
            render_points_as_spheres=True,
            cmap='plasma',
            clim=[0, 3]
        )
        
        # Set camera
        self.plotter.camera_position = [(15, 15, 10), (0, 0, 0), (0, 0, 1)]
        
        # Add title
        self.plotter.add_title("🫁 Real-Time Lung Drug Delivery Simulator\n(Press 'q' to quit)", font_size=14)
        
        print("🎮 Controls:")
        print("   - Mouse: Rotate, pan, zoom")  
        print("   - 'q': Quit simulation")
        print("   - 'r': Reset camera")
        
    def run_interactive_simulation(self):
        """Run simulation with manual updates"""
        print("🚀 Starting interactive drug delivery simulation...")
        print("💡 Move mouse to interact, particles will update automatically")
        
        # Show with callback for updates
        self.plotter.show(
            auto_close=False,
            interactive_update=True
        )
        
        # Manual animation loop
        frame_count = 0
        while not self.plotter.render_window.GetInteractor().GetDone():
            # Update particles every few frames for smoother interaction
            if frame_count % 3 == 0:  # Update every 3rd frame
                self.update_particles()
                
                # Update particle mesh
                self.particle_mesh.points = self.particle_positions
                velocity_magnitudes = np.linalg.norm(self.particle_velocities, axis=1)
                self.particle_mesh.point_data['velocity_mag'] = velocity_magnitudes
                
            # Render frame
            self.plotter.render()
            time.sleep(0.03)  # ~30 FPS
            frame_count += 1
            
    def run_simple_animation(self):
        """Run a simpler non-interactive animation"""
        print("🚀 Starting drug delivery animation...")
        
        # Animate for a fixed number of steps
        for step in range(1000):
            self.update_particles()
            
            # Update visualization every 10 steps
            if step % 10 == 0:
                self.particle_mesh.points = self.particle_positions
                velocity_mags = np.linalg.norm(self.particle_velocities, axis=1)
                self.particle_mesh.point_data['velocity_mag'] = velocity_mags
                
                # Save screenshot periodically
                if step % 100 == 0:
                    self.plotter.screenshot(f'drug_delivery_frame_{step:04d}.png')
                    print(f"📸 Saved frame {step}")
                    
        # Show final result
        self.plotter.show()

def main():
    """Main function"""
    print("🫁 Interactive Lung Drug Delivery Simulator")
    print("=" * 50)
    
    try:
        simulator = SimpleLungSimulator()
        
        print("\nChoose simulation mode:")
        print("1. Interactive (recommended)")
        print("2. Simple animation")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            simulator.run_simple_animation()
        else:
            simulator.run_interactive_simulation()
            
    except KeyboardInterrupt:
        print("\n⏹️ Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()