#!/usr/bin/env python3
"""
Comparative Drug Delivery Heatmap Generator
Creates side-by-side visualizations comparing MDI, DPI, and Nebulizer effectiveness
"""

import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent crashes
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
from pathlib import Path
import os
try:
    from .results_manager import create_results_folder
except ImportError:
    from results_manager import create_results_folder

class ComparativeHeatmapGenerator:
    def __init__(self):
        """Initialize comparative heatmap generator"""
        # Create results folder
        self.results_folder = create_results_folder()
        print(f"📁 Results will be saved to: {self.results_folder}")
        
        self.load_simulation_results()
        self.setup_delivery_characteristics()
        
    def load_simulation_results(self):
        """Load simulation results and case directories"""
        results_file = Path('simulation_results.json')
        
        if results_file.exists():
            with open(results_file) as f:
                self.simulation_results = json.load(f)
        else:
            print("⚠️  No simulation results found - using demo data")
            self.simulation_results = {}
            
        print(f"📊 Found {len(self.simulation_results)} simulation results")
        
    def setup_delivery_characteristics(self):
        """Define characteristics of each delivery method"""
        self.delivery_methods = {
            'MDI': {
                'name': 'Metered Dose Inhaler',
                'color': 'orange',
                'description': 'Propellant-driven spray - fast particles, throat deposition',
                'velocity': 1.5,
                'particle_size': 3.0,  # microns
                'deposition_pattern': 'throat_heavy',
                'effectiveness_score': 0.65
            },
            'DPI': {
                'name': 'Dry Powder Inhaler', 
                'color': 'blue',
                'description': 'High-velocity inhalation - good lung penetration',
                'velocity': 4.5,
                'particle_size': 4.0,
                'deposition_pattern': 'balanced',
                'effectiveness_score': 0.85
            },
            'Nebulizer': {
                'name': 'Nebulizer',
                'color': 'green', 
                'description': 'Fine mist - excellent deep lung delivery',
                'velocity': 0.75,
                'particle_size': 2.5,
                'deposition_pattern': 'deep_lung',
                'effectiveness_score': 0.90
            }
        }
        
    def load_lung_geometry(self):
        """Load lung geometry for visualization"""
        try:
            # Try to load from STL
            lung_files = ['lungs_repaired.stl', 'lungs.stl', 'my_lung.stl']
            
            for filename in lung_files:
                if os.path.exists(filename):
                    self.lung_mesh = pv.read(filename)
                    print(f"✅ Loaded lung geometry: {filename}")
                    break
            else:
                # Create simplified lung for demo
                print("⚠️  Creating simplified lung geometry")
                self.lung_mesh = self.create_demo_lung()
                
        except Exception as e:
            print(f"⚠️  Geometry loading error: {e}")
            self.lung_mesh = self.create_demo_lung()
            
        # Prepare surface
        self.prepare_lung_surface()
        
    def create_demo_lung(self):
        """Create simplified lung geometry for demo"""
        # Main lung structures
        trachea = pv.Cylinder(center=[0, 0, 8], radius=2, height=12)
        left_lung = pv.Sphere(center=[-8, 0, -5], radius=8)
        right_lung = pv.Sphere(center=[8, 0, -5], radius=7.5)
        
        # Combine and clean
        lung = trachea + left_lung + right_lung
        return lung.clean()
        
    def prepare_lung_surface(self):
        """Prepare lung surface and center it"""
        # Center and scale for good visualization
        center = self.lung_mesh.center
        self.lung_mesh.translate([-center[0], -center[1], -center[2]], inplace=True)
        self.lung_mesh.scale(0.12, inplace=True)
        
        # Extract surface
        self.lung_surface = self.lung_mesh.extract_surface()
        print(f"🫁 Prepared surface: {self.lung_surface.n_points} points")
        
    def calculate_deposition_patterns(self):
        """Calculate particle deposition for each delivery method with time-series"""
        print("🔬 Calculating TIME-SERIES deposition patterns...")
        
        points = self.lung_surface.points
        n_points = len(points)
        
        # Time parameters for realistic evolution
        total_time = 2.0
        dt = 0.1
        time_steps = np.arange(0, total_time + dt, dt)
        
        self.deposition_data = {}
        
        for method_key, method_info in self.delivery_methods.items():
            print(f"  • Calculating {method_key} time-series deposition...")
            
            # Generate time-evolved deposition
            final_deposition = self.generate_time_evolved_deposition(
                points, method_info, time_steps)
            
            # Keep full dynamic range (0.0 to 1.0) so low-deposition renders white
            final_deposition = np.clip(final_deposition, 0.0, 1.0)
            
            # Store data
            self.deposition_data[method_key] = final_deposition
            
            # Add to mesh
            self.lung_surface.point_data[f'{method_key}_Deposition'] = final_deposition
            
            print(f"    ✓ {method_key}: Range {final_deposition.min():.3f} to {final_deposition.max():.3f}")
            
        print("✅ All deposition patterns calculated")
        
    def calculate_base_deposition(self, x, y, z, method_info):
        """Calculate base deposition based on physics"""
        # Tracheal region (inlet effects)
        if z > 5 and np.sqrt(x**2 + y**2) < 3:
            return 0.7 + 0.2 * np.random.random()
            
        # Bronchial bifurcations (turbulence)
        elif 0 < z < 5 and 2 < np.sqrt(x**2 + y**2) < 5:
            return 0.5 + 0.3 * np.random.random()
            
        # Deep lung regions
        elif z < -2:
            distance_from_center = np.sqrt((abs(x) - 8)**2 + y**2 + (z + 5)**2)
            if distance_from_center < 6:
                depth_factor = max(0, 1 - distance_from_center / 6)
                return 0.2 * depth_factor + 0.1 * np.random.random()
                
        return 0.1 * np.random.random()
    
    def generate_time_evolved_deposition(self, points, method_info, time_steps):
        """Generate realistic physics-based deposition pattern for each delivery method"""
        n_points = len(points)
        deposition = np.zeros(n_points)
        
        # Use completely different physics for each method
        if method_info['deposition_pattern'] == 'throat_heavy':  # MDI
            deposition = self.calculate_mdi_deposition(points)
        elif method_info['deposition_pattern'] == 'balanced':  # DPI
            deposition = self.calculate_dpi_deposition(points)  
        else:  # Nebulizer
            deposition = self.calculate_nebulizer_deposition(points)
            
        return deposition
    
    def calculate_mdi_deposition(self, points):
        """MDI: High-momentum particles, heavy throat/upper airway deposition.
        Uses normalized coordinates to be robust to arbitrary STL scaling."""
        pts = np.asarray(points)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / max(1e-8, (z_max - z_min))
        r = np.sqrt(x**2 + y**2)
        r_max = max(1e-8, r.max())
        r_norm = r / r_max

        dep = np.zeros_like(z_norm)

        # Strong upper-airway (proximal) impaction near center axis
        upper = z_norm > 0.75
        central = r_norm < 0.35
        dep += upper * central * (0.85 * (1 - 0.5 * r_norm))

        # Moderate deposition in first branches (mid-depth, near axis)
        mid = (z_norm >= 0.4) & (z_norm <= 0.75)
        dep += mid * (0.45 * np.clip(0.7 - 0.6 * r_norm, 0.0, 1.0))

        # Minimal in deep lung
        lower = z_norm < 0.35
        dep += lower * (0.05 * (0.5 + 0.5 * np.random.random(size=dep.shape)))

        return np.clip(dep, 0.0, 1.0)
    
    def calculate_dpi_deposition(self, points):
        """DPI: High airflow, turbulent mixing, balanced distribution."""
        pts = np.asarray(points)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / max(1e-8, (z_max - z_min))
        r = np.sqrt(x**2 + y**2)
        r_max = max(1e-8, r.max())
        r_norm = r / r_max

        dep = np.zeros_like(z_norm)

        # Low throat impaction compared to MDI
        upper = z_norm > 0.75
        dep += upper * (0.15 * np.clip(0.6 - 0.5 * r_norm, 0.0, 1.0))

        # Bronchial region: balanced + turbulence-driven variation
        mid = (z_norm >= 0.3) & (z_norm <= 0.75)
        d_center = np.sqrt(x**2 + y**2 + (z_norm - 0.5)**2)
        turb = 0.18 * np.sin(3 * d_center) * np.cos(np.pi * (z_norm - 0.5))
        dep += mid * (0.38 + turb) * np.clip(1.0 - 0.4 * r_norm, 0.5, 1.0)

        # Deep lung: decent penetration
        lower = z_norm < 0.3
        dep += lower * (0.25 * (1.0 - 0.6 * r_norm))

        return np.clip(dep, 0.0, 0.9)
    
    def calculate_nebulizer_deposition(self, points):
        """Nebulizer: Low airflow, fine particles, excellent deep lung delivery."""
        pts = np.asarray(points)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / max(1e-8, (z_max - z_min))
        r = np.sqrt(x**2 + y**2)
        r_max = max(1e-8, r.max())
        r_norm = r / r_max

        dep = np.zeros_like(z_norm)

        # Minimal throat
        upper = z_norm > 0.65
        dep += upper * (0.03 * (0.8 + 0.4 * np.random.random(size=dep.shape)))

        # Bronchial: low-to-moderate
        mid = (z_norm >= 0.3) & (z_norm <= 0.65)
        dep += mid * (0.1 * (0.8 + 0.4 * np.random.random(size=dep.shape)))

        # Deep lung: strong gravity/diffusion-driven settling
        lower = z_norm < 0.3
        gravity = (1.0 - z_norm) ** 1.2
        radial = 0.6 + 0.4 * (1.0 - r_norm)
        dep += lower * (0.7 * gravity * radial)

        return np.clip(dep, 0.0, 1.0)
        
    def get_throat_deposition_factor(self, x, y, z):
        """Calculate throat/tracheal deposition factor for MDI"""
        if z > 3:  # Tracheal region
            distance_from_axis = np.sqrt(x**2 + y**2)
            if distance_from_axis < 4:
                return max(0, 1 - distance_from_axis / 4)
        return 0
        
    def get_deep_lung_factor(self, x, y, z):
        """Calculate deep lung penetration factor for Nebulizer"""
        if z < 0:  # Lower lung regions
            # Distance to lung lobes
            left_distance = np.sqrt((x + 8)**2 + y**2 + (z + 5)**2)
            right_distance = np.sqrt((x - 8)**2 + y**2 + (z + 5)**2)
            
            min_distance = min(left_distance, right_distance)
            if min_distance < 6:
                return max(0, 1 - min_distance / 6)
                
        return 0
        
    def create_comparative_visualization(self):
        """Create side-by-side comparison of all delivery methods"""
        print("🎨 Creating comparative visualization...")
        
        # Create 3-panel plot
        plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 600])
        plotter.set_background('white')
        
        methods = ['MDI', 'DPI', 'Nebulizer']
        # Method-specific colormaps: orange for MDI, blue for DPI, green for Nebulizer
        method_colormaps = {
            'MDI': 'Oranges',      # White to orange
            'DPI': 'Blues',        # White to blue  
            'Nebulizer': 'Greens'  # White to green
        }
        
        for i, method in enumerate(methods):
            plotter.subplot(0, i)
            
            # Add lung with deposition heatmap
            actor = plotter.add_mesh(
                self.lung_surface,
                scalars=f'{method}_Deposition',
                cmap=method_colormaps[method],  # Method-specific colormap
                show_edges=False,
                smooth_shading=True,
                clim=[0, 1],
                opacity=0.9
            )
            
            # Method info
            method_info = self.delivery_methods[method]
            title = f"{method_info['name']}\n{method_info['particle_size']:.1f}μm particles"
            plotter.add_title(title, font_size=14, color='black')
            
            # Optimal camera angle (zoomed out)
            plotter.camera_position = [(40, 35, 25), (0, 0, 0), (0, 0, 1)]
            
        # Add overall colorbar
        plotter.add_scalar_bar(
            title='Drug Particle\nDeposition Density',
            position_x=0.85,
            position_y=0.2,
            width=0.12,
            height=0.6,
            color='black'
        )
        
        # Console legend aligned with per-method colormaps
        print("🎯 COMPARATIVE DRUG DELIVERY ANALYSIS")
        print("=" * 50)
        print("📊 Side-by-side comparison of all 3 delivery methods")
        print("🔶 MDI: Oranges colormap (higher = darker orange)")
        print("🔷 DPI: Blues colormap (higher = darker blue)")
        print("🟢 Nebulizer: Greens colormap (higher = darker green)")
        print("⚪ White/Light = Lower deposition")
        print("🎮 Controls: Mouse rotate/zoom | 's' save | 'q' quit")
        
        # Save screenshot with off_screen rendering
        try:
            # Use off_screen mode to ensure proper rendering
            plotter.off_screen = True
            comparison_path = os.path.join(self.results_folder, 'drug_delivery_comparison.png')
            plotter.screenshot(comparison_path, 
                             transparent_background=False,
                             window_size=[2400, 800])
            print(f"📸 Saved comparison: {comparison_path}")
            
            # Reset for potential interactive display
            plotter.off_screen = False
            
        except Exception as e:
            print(f"⚠️  Screenshot error: {e}")
            
        # Try to show interactive plot (may fail in headless mode)
        try:
            plotter.show()
        except Exception as e:
            print(f"⚠️  Interactive display not available: {e}")
            print("📸 Static images saved instead")
        
    def create_quantitative_analysis(self):
        """Create quantitative comparison charts"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Quantitative Drug Delivery Analysis', fontsize=16, fontweight='bold')
            
            methods = list(self.delivery_methods.keys())
            
            # 1. Total Deposition Comparison
            total_deposition = [np.mean(self.deposition_data[method]) for method in methods]
            colors = [self.delivery_methods[method]['color'] for method in methods]
            
            bars1 = ax1.bar(methods, total_deposition, color=colors, alpha=0.7)
            ax1.set_title('Average Deposition Density', fontweight='bold')
            ax1.set_ylabel('Deposition Density')
            
            # Add value labels
            for bar, value in zip(bars1, total_deposition):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
                        
            # 2. Particle Size vs Penetration
            sizes = [self.delivery_methods[method]['particle_size'] for method in methods]
            effectiveness = [self.delivery_methods[method]['effectiveness_score'] for method in methods]
            
            scatter = ax2.scatter(sizes, effectiveness, s=200, c=colors, alpha=0.7, edgecolors='black')
            ax2.set_xlabel('Particle Size (μm)')
            ax2.set_ylabel('Delivery Effectiveness')
            ax2.set_title('Particle Size vs Effectiveness', fontweight='bold')
            
            # Add method labels
            for i, method in enumerate(methods):
                ax2.annotate(method, (sizes[i], effectiveness[i]), 
                            xytext=(5, 5), textcoords='offset points')
                            
            # 3. Regional Deposition Analysis
            regions = ['Throat/Trachea', 'Bronchi', 'Deep Lung']
            
            # Calculate regional deposition (simplified)
            regional_data = {}
            for method in methods:
                deposition = self.deposition_data[method]
                points = self.lung_surface.points
                
                throat_deposition = np.mean([dep for i, dep in enumerate(deposition) 
                                           if points[i][2] > 3])  # Upper regions
                bronchi_deposition = np.mean([dep for i, dep in enumerate(deposition) 
                                            if -1 < points[i][2] < 3])  # Middle
                deep_deposition = np.mean([dep for i, dep in enumerate(deposition) 
                                         if points[i][2] < -1])  # Lower
                                         
                regional_data[method] = [throat_deposition, bronchi_deposition, deep_deposition]
                
            # Create grouped bar chart
            x = np.arange(len(regions))
            width = 0.25
            
            for i, method in enumerate(methods):
                offset = (i - 1) * width
                color = self.delivery_methods[method]['color']
                ax3.bar(x + offset, regional_data[method], width, 
                       label=method, color=color, alpha=0.7)
                       
            ax3.set_xlabel('Lung Region')
            ax3.set_ylabel('Deposition Density')
            ax3.set_title('Regional Deposition Distribution', fontweight='bold')
            ax3.set_xticks(x)
            ax3.set_xticklabels(regions)
            ax3.legend()
            
            # 4. Delivery Method Summary Table
            ax4.axis('off')
            
            # Create summary table
            table_data = []
            headers = ['Method', 'Velocity\n(m/s)', 'Particle Size\n(μm)', 'Effectiveness\n(%)', 'Best For']
            
            best_for = {
                'MDI': 'Quick relief',
                'DPI': 'Active patients', 
                'Nebulizer': 'Deep delivery'
            }
            
            for method in methods:
                info = self.delivery_methods[method]
                row = [
                    method,
                    f"{info['velocity']:.1f}",
                    f"{info['particle_size']:.1f}",
                    f"{info['effectiveness_score']:.0%}",
                    best_for[method]
                ]
                table_data.append(row)
                
            # Create table
            table = ax4.table(cellText=table_data, colLabels=headers,
                             cellLoc='center', loc='center',
                             colColours=['lightgray']*5)
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 2)
            
            ax4.set_title('Delivery Method Summary', fontweight='bold', pad=20)
            
            plt.tight_layout()
            quantitative_path = os.path.join(self.results_folder, 'drug_delivery_quantitative_analysis.png')
            plt.savefig(quantitative_path, 
                       dpi=300, bbox_inches='tight')
            print(f"📊 Saved quantitative analysis: {quantitative_path}")
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️  Error creating quantitative analysis: {e}")
            print("📊 Skipping quantitative charts...")
        
    def generate_all_comparisons(self):
        """Generate all comparative visualizations"""
        print("🫁 Comprehensive Drug Delivery Comparison Suite")
        print("=" * 60)
        
        # Load geometry and calculate patterns
        self.load_lung_geometry()
        self.calculate_deposition_patterns()
        
        # Create visualizations
        print("\n1. Creating 3D comparative visualization...")
        self.create_comparative_visualization()
        
        print("\n2. Creating quantitative analysis...")
        self.create_quantitative_analysis()
        
        print("\n✅ Comparative analysis complete!")
        print("📁 Generated files:")
        print(f"  • {self.results_folder}/drug_delivery_comparison.png (3D comparison)")
        print(f"  • {self.results_folder}/drug_delivery_quantitative_analysis.png (charts)")

def main():
    """Main interface"""
    print("🫁 Comparative Drug Delivery Heatmap Generator")
    print("=" * 50)
    print("Creates side-by-side visualizations of MDI vs DPI vs Nebulizer")
    print()
    
    generator = ComparativeHeatmapGenerator()
    generator.generate_all_comparisons()

if __name__ == "__main__":
    main()