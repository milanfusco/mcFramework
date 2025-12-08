"""
Neutron Transport Simulation Demonstration.

This script demonstrates the MCMC neutron transport simulation capabilities
in both 1D and 2D geometries, including:
- Neutron flux distribution
- Leakage and absorption probabilities
- k-effective (criticality) calculations

Run this script to see neutron transport simulations in action.
"""

import matplotlib.pyplot as plt
import numpy as np

from mcframework.sims.neutron_transport import (
    Geometry1DSlab,
    Geometry2DPlane,
    KEFFSimulation,
    Material,
    NeutronTransportSimulation,
)


def demo_1d_flux_distribution():
    """
    Demonstrate 1D neutron transport with flux distribution.
    
    Creates a simple 1D slab with U-235 fuel and computes the
    spatial flux distribution.
    """
    print("\n" + "="*70)
    print("DEMO 1: 1D Slab - Flux Distribution")
    print("="*70)
    
    # Create materials
    u235 = Material.create_u235()
    
    # Create 1D geometry: 10 cm slab of pure U-235
    geometry = Geometry1DSlab(
        x_min=0.0,
        x_max=10.0,
        boundaries=[0.0, 10.0],
        materials=[u235],
        boundary_condition="vacuum",
    )
    
    # Create simulation
    sim = NeutronTransportSimulation(
        name="1D Flux Distribution",
        geometry=geometry,
        source_position=(5.0,),  # Center source
        source_energy_group=2,   # Fast neutrons
        flux_bins=50,
    )
    
    sim.set_seed(42)
    
    # Run simulation
    print("\nRunning 1D neutron transport simulation...")
    print("  - Geometry: 10 cm U-235 slab")
    print("  - Source: Center (5.0 cm), fast neutrons")
    print("  - Boundary: Vacuum")
    
    result = sim.run(
        n_simulations=1000,
        parallel=False,
        return_type="flux",
    )
    
    print(f"\n  Mean flux: {result.mean:.4f}")
    print(f"  Std Dev: {result.std:.4f}")
    print(f"  Execution time: {result.execution_time:.2f} seconds")
    
    # Compute full flux distribution
    print("\nComputing spatial flux distribution...")
    flux_dist = sim.compute_flux_distribution(
        n_histories=5000,
        flux_bins=50,
    )
    
    # Plot results
    plt.figure(figsize=(10, 6))
    x_bins = np.linspace(0, 10, 50)
    plt.plot(x_bins, flux_dist, 'b-', linewidth=2, label='Flux Distribution')
    plt.axvline(x=5.0, color='r', linestyle='--', label='Source Position')
    plt.xlabel('Position (cm)', fontsize=12)
    plt.ylabel('Flux (arbitrary units)', fontsize=12)
    plt.title('1D Neutron Flux Distribution in U-235 Slab', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('neutron_1d_flux.png', dpi=150)
    print("  Saved plot: neutron_1d_flux.png")
    plt.close()


def demo_1d_heterogeneous():
    """
    Demonstrate 1D heterogeneous geometry with fuel and moderator.
    
    Creates a 1D slab with alternating fuel and water regions.
    """
    print("\n" + "="*70)
    print("DEMO 2: 1D Heterogeneous Geometry (Fuel + Water)")
    print("="*70)
    
    # Create materials
    u235 = Material.create_u235()
    h2o = Material.create_water()
    
    # Create heterogeneous geometry: Water | Fuel | Water
    # 0-2 cm: Water, 2-8 cm: U-235, 8-10 cm: Water
    geometry = Geometry1DSlab(
        x_min=0.0,
        x_max=10.0,
        boundaries=[0.0, 2.0, 8.0, 10.0],
        materials=[h2o, u235, h2o],
        boundary_condition="vacuum",
    )
    
    # Create simulation
    sim = NeutronTransportSimulation(
        name="1D Heterogeneous",
        geometry=geometry,
        source_position=(5.0,),
        source_energy_group=2,
        flux_bins=50,
    )
    
    sim.set_seed(42)
    
    print("\nRunning heterogeneous 1D simulation...")
    print("  - Geometry: Water (2cm) | U-235 (6cm) | Water (2cm)")
    print("  - Source: Center of fuel region (5.0 cm)")
    
    # Compute statistics
    print("\nComputing leakage probability...")
    result_leakage = sim.run(
        n_simulations=5000,
        parallel=False,
        return_type="leakage_prob",
    )
    
    print(f"  Leakage probability: {result_leakage.mean:.4f} ± {result_leakage.std:.4f}")
    
    print("\nComputing absorption probability...")
    result_absorption = sim.run(
        n_simulations=5000,
        parallel=False,
        return_type="absorption_prob",
    )
    
    print(f"  Absorption probability: {result_absorption.mean:.4f} ± {result_absorption.std:.4f}")
    
    # Compute flux distribution
    flux_dist = sim.compute_flux_distribution(n_histories=10000, flux_bins=50)
    
    # Plot with material regions
    plt.figure(figsize=(12, 6))
    x_bins = np.linspace(0, 10, 50)
    
    # Shade material regions
    plt.axvspan(0, 2, alpha=0.2, color='blue', label='Water')
    plt.axvspan(2, 8, alpha=0.2, color='red', label='U-235')
    plt.axvspan(8, 10, alpha=0.2, color='blue')
    
    plt.plot(x_bins, flux_dist, 'k-', linewidth=2, label='Flux')
    plt.axvline(x=5.0, color='orange', linestyle='--', label='Source')
    plt.xlabel('Position (cm)', fontsize=12)
    plt.ylabel('Flux (arbitrary units)', fontsize=12)
    plt.title('Flux Distribution in Heterogeneous 1D Geometry', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('neutron_1d_heterogeneous.png', dpi=150)
    print("\n  Saved plot: neutron_1d_heterogeneous.png")
    plt.close()


def demo_2d_geometry():
    """
    Demonstrate 2D neutron transport.
    
    Creates a 2D plane with fuel in the center surrounded by water.
    """
    print("\n" + "="*70)
    print("DEMO 3: 2D Plane - Fuel Assembly")
    print("="*70)
    
    # Create materials
    u235 = Material.create_u235()
    h2o = Material.create_water()
    
    # Create 2D geometry: 10x10 cm plane
    # Center 6x6 cm fuel region surrounded by water
    geometry = Geometry2DPlane(
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        regions=[
            (2.0, 2.0, 8.0, 8.0, u235),  # Fuel region
        ],
        default_material=h2o,  # Water everywhere else
        boundary_condition="vacuum",
    )
    
    # Create simulation
    sim = NeutronTransportSimulation(
        name="2D Fuel Assembly",
        geometry=geometry,
        source_position=(5.0, 5.0),  # Center source
        source_energy_group=2,
        flux_bins=50,
    )
    
    sim.set_seed(42)
    
    print("\nRunning 2D neutron transport simulation...")
    print("  - Geometry: 10x10 cm plane")
    print("  - Fuel: 6x6 cm U-235 in center")
    print("  - Moderator: Water surrounding fuel")
    print("  - Source: Center (5.0, 5.0 cm)")
    
    result = sim.run(
        n_simulations=2000,
        parallel=False,
        return_type="flux",
    )
    
    print(f"\n  Mean flux: {result.mean:.4f}")
    print(f"  Std Dev: {result.std:.4f}")
    print(f"  Execution time: {result.execution_time:.2f} seconds")
    
    # Compute leakage
    result_leakage = sim.run(
        n_simulations=2000,
        parallel=False,
        return_type="leakage_prob",
    )
    
    print(f"  Leakage probability: {result_leakage.mean:.4f} ± {result_leakage.std:.4f}")


def demo_keff_calculation():
    """
    Demonstrate k-effective (criticality) calculation.
    
    Computes the effective neutron multiplication factor for a
    fissile system using power iteration.
    """
    print("\n" + "="*70)
    print("DEMO 4: k-effective (Criticality) Calculation")
    print("="*70)
    
    # Create pure U-235 slab
    u235 = Material.create_u235()
    
    # 10 cm slab of pure U-235
    geometry = Geometry1DSlab(
        x_min=0.0,
        x_max=10.0,
        boundaries=[0.0, 10.0],
        materials=[u235],
        boundary_condition="reflective",  # Reflective for criticality
    )
    
    # Create k-eff simulation
    sim = KEFFSimulation(
        name="k-eff Calculation",
        geometry=geometry,
        initial_source_positions=[(5.0,)],
        n_generations=100,
        n_neutrons_per_generation=500,
    )
    
    sim.set_seed(42)
    
    print("\nRunning k-effective calculation...")
    print("  - Geometry: 10 cm U-235 slab")
    print("  - Boundary: Reflective")
    print("  - Generations: 100")
    print("  - Neutrons per generation: 500")
    print("  - Skip first 10 generations for convergence")
    
    result = sim.run(
        n_simulations=20,  # 20 independent k-eff estimates
        parallel=False,
        skip_generations=10,
    )
    
    print(f"\n  Mean k-eff: {result.mean:.5f}")
    print(f"  Std Dev: {result.std:.5f}")
    print(f"  95% CI: [{result.mean - 1.96*result.std/np.sqrt(result.n_simulations):.5f}, "
          f"{result.mean + 1.96*result.std/np.sqrt(result.n_simulations):.5f}]")
    print(f"  Execution time: {result.execution_time:.2f} seconds")
    
    if result.mean > 1.0:
        print(f"\n  System is SUPERCRITICAL (k-eff > 1)")
    elif result.mean < 1.0:
        print(f"\n  System is SUBCRITICAL (k-eff < 1)")
    else:
        print(f"\n  System is CRITICAL (k-eff = 1)")
    
    # Plot k-eff distribution
    plt.figure(figsize=(10, 6))
    plt.hist(result.results, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=result.mean, color='r', linestyle='--', linewidth=2, label=f'Mean k-eff = {result.mean:.4f}')
    plt.axvline(x=1.0, color='g', linestyle='-', linewidth=2, label='Critical (k-eff = 1.0)')
    plt.xlabel('k-effective', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of k-effective Estimates', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('neutron_keff_distribution.png', dpi=150)
    print("\n  Saved plot: neutron_keff_distribution.png")
    plt.close()


def demo_keff_vs_size():
    """
    Demonstrate how k-eff varies with system size.
    
    Shows the relationship between slab thickness and criticality.
    """
    print("\n" + "="*70)
    print("DEMO 5: k-effective vs System Size")
    print("="*70)
    
    u235 = Material.create_u235()
    
    sizes = [5.0, 7.0, 10.0, 15.0, 20.0]  # Different slab thicknesses (cm)
    k_eff_values = []
    k_eff_errors = []
    
    print("\nComputing k-eff for different system sizes...")
    
    for size in sizes:
        print(f"  Size: {size:.1f} cm...", end=" ")
        
        geometry = Geometry1DSlab(
            x_min=0.0,
            x_max=size,
            boundaries=[0.0, size],
            materials=[u235],
            boundary_condition="reflective",
        )
        
        sim = KEFFSimulation(
            geometry=geometry,
            initial_source_positions=[(size/2,)],
            n_generations=80,
            n_neutrons_per_generation=300,
        )
        
        sim.set_seed(42)
        
        result = sim.run(
            n_simulations=15,
            parallel=False,
            skip_generations=10,
        )
        
        k_eff_values.append(result.mean)
        k_eff_errors.append(result.std / np.sqrt(result.n_simulations))
        
        print(f"k-eff = {result.mean:.4f} ± {k_eff_errors[-1]:.4f}")
    
    # Plot k-eff vs size
    plt.figure(figsize=(10, 6))
    plt.errorbar(sizes, k_eff_values, yerr=k_eff_errors, 
                 fmt='o-', linewidth=2, markersize=8, capsize=5,
                 label='k-eff with uncertainty')
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Critical (k-eff = 1.0)')
    plt.xlabel('Slab Thickness (cm)', fontsize=12)
    plt.ylabel('k-effective', fontsize=12)
    plt.title('k-effective vs System Size (U-235 Slab)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('neutron_keff_vs_size.png', dpi=150)
    print("\n  Saved plot: neutron_keff_vs_size.png")
    plt.close()


def main():
    """Run all neutron transport demonstrations."""
    print("\n" + "="*70)
    print("NEUTRON TRANSPORT SIMULATION DEMONSTRATIONS")
    print("Markov Chain Monte Carlo Methods")
    print("="*70)
    
    # Run all demos
    demo_1d_flux_distribution()
    demo_1d_heterogeneous()
    demo_2d_geometry()
    demo_keff_calculation()
    demo_keff_vs_size()
    
    print("\n" + "="*70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*70)
    print("\nGenerated plots:")
    print("  - neutron_1d_flux.png")
    print("  - neutron_1d_heterogeneous.png")
    print("  - neutron_keff_distribution.png")
    print("  - neutron_keff_vs_size.png")
    print("\nThese demonstrations show:")
    print("  1. Neutron flux distribution in 1D geometries")
    print("  2. Heterogeneous materials (fuel + moderator)")
    print("  3. 2D geometry transport")
    print("  4. k-effective (criticality) calculations")
    print("  5. Critical mass estimation")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

