#!/usr/bin/env python3
"""
Drug Delivery CFD Visualization Script
Visualizes OpenFOAM simulation results for lung drug delivery analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re
import os

def parse_openfoam_field(filepath):
    """Parse OpenFOAM field files (U, p, etc.)"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the data section
    start_marker = "internalField   nonuniform"
    if start_marker in content:
        # Find the start of data
        start_idx = content.find(start_marker)
        start_idx = content.find("(", start_idx)
        end_idx = content.find(");", start_idx)
        
        data_str = content[start_idx+1:end_idx]
        
        # Parse based on field type
        if "vector" in content:
            # Vector field (velocity)
            vectors = []
            lines = data_str.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('(') and line.endswith(')'):
                    coords = line[1:-1].split()
                    if len(coords) == 3:
                        try:
                            vectors.append([float(x) for x in coords])
                        except:
                            pass
            return np.array(vectors)
        else:
            # Scalar field (pressure)
            scalars = []
            lines = data_str.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('('):
                    try:
                        scalars.append(float(line))
                    except:
                        pass
            return np.array(scalars)
    
    return None

def analyze_drug_delivery_results():
    """Analyze and visualize drug delivery simulation results"""
    
    print("🫁 Drug Delivery CFD Analysis")
    print("=" * 50)
    
    # Load velocity field
    velocity_file = "100/U"
    pressure_file = "100/p"
    
    if not os.path.exists(velocity_file):
        print("❌ Results not found. Make sure 100/U exists.")
        return
    
    # Parse velocity data
    print("📊 Loading velocity field...")
    velocities = parse_openfoam_field(velocity_file)
    
    if velocities is not None and len(velocities) > 0:
        # Calculate velocity statistics
        vel_magnitude = np.sqrt(np.sum(velocities**2, axis=1))
        
        print(f"✅ Loaded {len(velocities)} velocity vectors")
        print(f"📈 Velocity Statistics:")
        print(f"   - Min: {vel_magnitude.min():.4f} m/s")
        print(f"   - Max: {vel_magnitude.max():.4f} m/s") 
        print(f"   - Mean: {vel_magnitude.mean():.4f} m/s")
        print(f"   - Std: {vel_magnitude.std():.4f} m/s")
        
        # Drug delivery analysis
        print(f"\n💊 Drug Delivery Analysis:")
        
        # Classify flow regions for drug delivery
        slow_flow = vel_magnitude < 0.05  # Deposition zones
        medium_flow = (vel_magnitude >= 0.05) & (vel_magnitude < 0.2)  # Transport zones
        fast_flow = vel_magnitude >= 0.2  # Main airways
        
        print(f"   - Deposition zones (<0.05 m/s): {np.sum(slow_flow)} cells ({np.sum(slow_flow)/len(velocities)*100:.1f}%)")
        print(f"   - Transport zones (0.05-0.2 m/s): {np.sum(medium_flow)} cells ({np.sum(medium_flow)/len(velocities)*100:.1f}%)")
        print(f"   - Main airways (>0.2 m/s): {np.sum(fast_flow)} cells ({np.sum(fast_flow)/len(velocities)*100:.1f}%)")
        
        # Create visualizations
        create_velocity_plots(velocities, vel_magnitude)
        
    # Parse pressure data
    if os.path.exists(pressure_file):
        print(f"\n📊 Loading pressure field...")
        pressures = parse_openfoam_field(pressure_file)
        
        if pressures is not None and len(pressures) > 0:
            print(f"✅ Loaded {len(pressures)} pressure values")
            print(f"📈 Pressure Statistics:")
            print(f"   - Min: {pressures.min():.6f} Pa")
            print(f"   - Max: {pressures.max():.6f} Pa")
            print(f"   - Mean: {pressures.mean():.6f} Pa")
            print(f"   - Range: {pressures.max() - pressures.min():.6f} Pa")
            
            create_pressure_plots(pressures)

def create_velocity_plots(velocities, vel_magnitude):
    """Create velocity visualization plots"""
    
    # Velocity magnitude histogram
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Velocity magnitude distribution
    plt.subplot(2, 3, 1)
    plt.hist(vel_magnitude, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Velocity Magnitude (m/s)')
    plt.ylabel('Frequency')
    plt.title('🌬️ Drug Delivery Velocity Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for flow regime boundaries
    plt.axvline(0.05, color='orange', linestyle='--', label='Deposition threshold')
    plt.axvline(0.2, color='red', linestyle='--', label='Transport threshold')
    plt.legend()
    
    # Subplot 2: Velocity components
    plt.subplot(2, 3, 2)
    plt.scatter(velocities[:, 0], velocities[:, 1], c=vel_magnitude, 
               cmap='viridis', alpha=0.6, s=1)
    plt.colorbar(label='Velocity Magnitude (m/s)')
    plt.xlabel('Vx (m/s)')
    plt.ylabel('Vy (m/s)')
    plt.title('🔄 Velocity Components (X-Y)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Z-component vs magnitude
    plt.subplot(2, 3, 3)
    plt.scatter(velocities[:, 2], vel_magnitude, alpha=0.6, s=1, color='green')
    plt.xlabel('Vz (m/s)')
    plt.ylabel('Velocity Magnitude (m/s)')
    plt.title('📐 Axial Flow vs Total Velocity')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Flow regime pie chart
    plt.subplot(2, 3, 4)
    slow_count = np.sum(vel_magnitude < 0.05)
    medium_count = np.sum((vel_magnitude >= 0.05) & (vel_magnitude < 0.2))
    fast_count = np.sum(vel_magnitude >= 0.2)
    
    labels = ['Deposition\n(<0.05 m/s)', 'Transport\n(0.05-0.2 m/s)', 'Main Airways\n(>0.2 m/s)']
    sizes = [slow_count, medium_count, fast_count]
    colors = ['lightblue', 'orange', 'red']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('💊 Drug Delivery Flow Regimes')
    
    # Subplot 5: Velocity vector sample (3D)
    ax = plt.subplot(2, 3, 5, projection='3d')
    
    # Sample every 100th vector to avoid crowding
    sample_idx = np.arange(0, len(velocities), max(1, len(velocities)//1000))
    sample_pos = sample_idx.reshape(-1, 1) * np.array([1, 1, 1])  # Approximate positions
    sample_vel = velocities[sample_idx]
    
    ax.quiver(sample_pos[:, 0], sample_pos[:, 1], sample_pos[:, 2],
              sample_vel[:, 0], sample_vel[:, 1], sample_vel[:, 2],
              length=10, normalize=True, alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title('🎯 Velocity Vectors (Sample)')
    
    # Subplot 6: Cumulative velocity distribution
    plt.subplot(2, 3, 6)
    sorted_vel = np.sort(vel_magnitude)
    cumulative = np.arange(1, len(sorted_vel) + 1) / len(sorted_vel) * 100
    plt.plot(sorted_vel, cumulative, linewidth=2, color='purple')
    plt.xlabel('Velocity Magnitude (m/s)')
    plt.ylabel('Cumulative Percentage (%)')
    plt.title('📈 Cumulative Velocity Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add percentile markers
    p50 = np.percentile(vel_magnitude, 50)
    p90 = np.percentile(vel_magnitude, 90)
    plt.axvline(p50, color='orange', linestyle=':', label=f'50th percentile: {p50:.3f}')
    plt.axvline(p90, color='red', linestyle=':', label=f'90th percentile: {p90:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('drug_delivery_velocity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Velocity plots saved as 'drug_delivery_velocity_analysis.png'")

def create_pressure_plots(pressures):
    """Create pressure visualization plots"""
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Pressure distribution
    plt.subplot(2, 2, 1)
    plt.hist(pressures, bins=50, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Pressure (Pa)')
    plt.ylabel('Frequency')
    plt.title('🌬️ Breathing Pressure Distribution')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Pressure vs cell index (spatial variation)
    plt.subplot(2, 2, 2)
    plt.plot(pressures, alpha=0.7, linewidth=0.5, color='blue')
    plt.xlabel('Cell Index')
    plt.ylabel('Pressure (Pa)')
    plt.title('📍 Spatial Pressure Variation')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Pressure statistics box plot
    plt.subplot(2, 2, 3)
    plt.boxplot([pressures], labels=['Pressure'])
    plt.ylabel('Pressure (Pa)')
    plt.title('📊 Pressure Statistics')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Pressure gradient analysis
    plt.subplot(2, 2, 4)
    pressure_gradient = np.gradient(pressures)
    plt.hist(pressure_gradient, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Pressure Gradient (Pa/cell)')
    plt.ylabel('Frequency')
    plt.title('📐 Pressure Gradient Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('drug_delivery_pressure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Pressure plots saved as 'drug_delivery_pressure_analysis.png'")

if __name__ == "__main__":
    analyze_drug_delivery_results()