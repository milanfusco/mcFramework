from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from mcframework import MonteCarloFramework
from mcframework import PiEstimationSimulation, PortfolioSimulation
from mcframework.stats_engine import build_default_engine


def progress(completed: int, total: int):
    step = max(1, total // 10)
    if completed % step == 0 or completed == total:
        print(f"Progress: {completed}/{total} ({100 * completed / total:.0f}%)")


def create_pi_visualizations(pi_result):
    """Create comprehensive visualizations for Pi estimation simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Pi Estimation Monte Carlo Analysis',
                 fontsize=16,
                 fontweight='bold')
    
    # 1. Histogram with statistical annotations
    ax1 = axes[0, 0]
    n_bins = min(50, len(pi_result.results) // 20)
    counts, bins, patches = ax1.hist(pi_result.results,
                                     bins=n_bins,
                                     alpha=0.7,
                                     density=True,
                                     color='skyblue',
                                     edgecolor='black')
    
    # Add statistical lines and annotations
    ax1.axvline(np.pi,
                color='red',
                linestyle='-',
                linewidth=2,
                label=f'True π = {np.pi:.6f}')
    ax1.axvline(pi_result.mean,
                color='orange',
                linestyle='--',
                linewidth=2,
                label=f'Mean = {pi_result.mean:.6f}')
    ax1.axvline(pi_result.percentiles[50],
                color='green',
                linestyle=':',
                linewidth=2,
                label=f'Median = {pi_result.percentiles[50]:.6f}')
    
    # Add confidence interval shading
    ci = pi_result.stats.get('ci_mean', {})
    if ci:
        ax1.axvspan(ci['low'],
                    ci['high'],
                    alpha=0.2,
                    color='yellow',
                    label=f"95% CI [{ci['low']:.5f}, {ci['high']:.5f}]")
    
    ax1.set_xlabel('Estimated π Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of π Estimates')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Convergence plot
    ax2 = axes[0, 1]
    cumulative_mean = np.cumsum(pi_result.results) / np.arange(1,
                                                               len(pi_result.results) + 1)
    sample_indices = np.arange(1, len(pi_result.results) + 1)
    
    ax2.plot(sample_indices,
             cumulative_mean,
             color='blue',
             alpha=0.8,
             linewidth=1.5)
    ax2.axhline(np.pi,
                color='red',
                linestyle='-',
                linewidth=2,
                label=f'True π = {np.pi:.6f}')
    ax2.fill_between(sample_indices,
                     cumulative_mean - 1.96 * pi_result.std / np.sqrt(
                         sample_indices),
                     cumulative_mean + 1.96 * pi_result.std / np.sqrt(
                         sample_indices),
                     alpha=0.2,
                     color='blue',
                     label='±1.96 SE')
    
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Cumulative Mean')
    ax2.set_title('Convergence to True π Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    # 3. Error analysis
    ax3 = axes[1, 0]
    errors = pi_result.results - np.pi
    ax3.hist(errors,
             bins=n_bins,
             alpha=0.7,
             color='lightcoral',
             edgecolor='black')
    ax3.axvline(0,
                color='black',
                linestyle='-',
                linewidth=2,
                label='Zero Error')
    ax3.axvline(np.mean(errors),
                color='blue',
                linestyle='--',
                linewidth=2,
                label=f'Mean Error = {np.mean(errors):.6f}')
    
    ax3.set_xlabel('Estimation Error (Estimate - True π)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Estimation Errors')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Q-Q plot for normality check
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(pi_result.results, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_portfolio_visualizations(port_res):
    """Create comprehensive visualizations for Portfolio simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Portfolio Value Monte Carlo Analysis',
                 fontsize=16,
                 fontweight='bold')
    
    # 1. Histogram of final portfolio values
    ax1 = axes[0, 0]
    n_bins = min(50, len(port_res.results) // 20)
    counts, bins, patches = ax1.hist(port_res.results,
                                     bins=n_bins,
                                     alpha=0.7,
                                     color='lightgreen',
                                     edgecolor='black')
    
    # Color bars by value ranges
    initial_value = 10_000  # From simulation parameters
    for i, (count, bin_edge) in enumerate(zip(counts, bins[:-1])):
        if bin_edge < initial_value:
            patches[i].set_facecolor('red')
        elif bin_edge < initial_value * 1.5:
            patches[i].set_facecolor('orange')
        else:
            patches[i].set_facecolor('green')
    
    ax1.axvline(initial_value,
                color='black',
                linestyle='-',
                linewidth=2,
                label=f'Initial Value = ${initial_value:,}')
    ax1.axvline(port_res.mean,
                color='blue',
                linestyle='--',
                linewidth=2,
                label=f'Mean = ${port_res.mean:,.0f}')
    ax1.axvline(port_res.percentiles[50],
                color='purple',
                linestyle=':',
                linewidth=2,
                label=f'Median = ${port_res.percentiles[50]:,.0f}')
    
    ax1.set_xlabel('Final Portfolio Value ($)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Final Portfolio Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis as currency
    ax1.ticklabel_format(style='plain', axis='x')
    
    # 2. Risk metrics visualization
    ax2 = axes[0, 1]
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_values = [
        port_res.percentiles.get(p, np.percentile(port_res.results, p))
        for p in percentiles]
    
    colors = ['red' if p <= 25 else
              'orange' if p <= 75 else
              'green'
              for p in percentiles]
    
    bars = ax2.bar(percentiles,
                   pct_values,
                   color=colors,
                   alpha=0.7,
                   edgecolor='black')
    ax2.axhline(initial_value,
                color='black',
                linestyle='-',
                linewidth=2,
                label=f'Initial Value = ${initial_value:,}')
    
    # Add value labels on bars
    for bar, value in zip(bars, pct_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.,
                 height + height * 0.01,
                 f'${value:,.0f}',
                 ha='center',
                 va='bottom',
                 fontsize=8)
    
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Risk Profile: Value at Percentiles')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns analysis
    ax3 = axes[1, 0]
    returns = (port_res.results - initial_value) / initial_value * 100
    ax3.hist(returns, bins=n_bins, alpha=0.7, color='gold', edgecolor='black')
    ax3.axvline(0,
                color='black',
                linestyle='-',
                linewidth=2,
                label='Break-even')
    ax3.axvline(np.mean(returns),
                color='blue',
                linestyle='--',
                linewidth=2,
                label=f'Mean Return = {np.mean(returns):.1f}%')
    
    # Calculate and show probability of loss
    prob_loss = np.mean(returns < 0) * 100
    ax3.text(0.05,
             0.95,
             f'P(Loss) = {prob_loss:.1f}%',
             transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
             verticalalignment='top')
    
    ax3.set_xlabel('Total Return (%)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Total Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot with outliers
    ax4 = axes[1, 1]
    box_data = [port_res.results]
    bp = ax4.boxplot(box_data, patch_artist=True, labels=['Portfolio Value'])
    bp['boxes'][0].set_facecolor('lightblue')
    
    # Add statistical annotations
    stats_text = f"""Mean: ${port_res.mean:,.0f}
        Std Dev: ${port_res.std:,.0f}
        Skewness: {port_res.stats.get('skew', 'N/A')}
        Kurtosis: {port_res.stats.get('kurtosis', 'N/A')}"""
    
    ax4.text(0.02,
             0.98,
             stats_text,
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             verticalalignment='top',
             fontfamily='monospace')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.set_title('Box Plot with Statistical Summary')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_comparison_chart(pi_result, port_res):
    """Create a comparison chart between the two simulations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Simulation Comparison: Convergence Analysis',
                 fontsize=16,
                 fontweight='bold')
    
    # Pi estimation convergence
    pi_cumulative = np.cumsum(pi_result.results) / np.arange(1,
                                                             len(pi_result.results) + 1)
    pi_indices = np.arange(1, len(pi_result.results) + 1)
    
    ax1.plot(pi_indices,
             pi_cumulative,
             color='blue',
             alpha=0.8,
             linewidth=1.5,
             label='Running Mean')
    ax1.axhline(np.pi,
                color='red',
                linestyle='-',
                linewidth=2,
                label=f'True π = {np.pi:.6f}')
    ax1.fill_between(pi_indices,
                     pi_cumulative - 1.96 * pi_result.std / np.sqrt(pi_indices),
                     pi_cumulative + 1.96 * pi_result.std / np.sqrt(pi_indices),
                     alpha=0.2,
                     color='blue',
                     label='±1.96 SE')
    ax1.set_xlabel('Number of Simulations')
    ax1.set_ylabel('Estimated π Value')
    ax1.set_title('Pi Estimation Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Portfolio value convergence
    port_cumulative = np.cumsum(port_res.results) / np.arange(1,
                                                              len(port_res.results) + 1)
    port_indices = np.arange(1, len(port_res.results) + 1)
    
    ax2.plot(port_indices,
             port_cumulative,
             color='green',
             alpha=0.8,
             linewidth=1.5,
             label='Running Mean')
    ax2.fill_between(port_indices,
                     port_cumulative - 1.96 * port_res.std / np.sqrt(
                         port_indices),
                     port_cumulative + 1.96 * port_res.std / np.sqrt(
                         port_indices),
                     alpha=0.2,
                     color='green',
                     label='±1.96 SE')
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Portfolio Value Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    return fig


def main():
    fw = MonteCarloFramework()
    pi_sim = PiEstimationSimulation()
    port_sim = PortfolioSimulation()
    
    # Reproducible across processes
    pi_sim.set_seed(43)
    port_sim.set_seed(43)
    
    fw.register_simulation(pi_sim)
    fw.register_simulation(port_sim)
    
    print("Running Pi Estimation…")
    pi_engine = build_default_engine()
    pi_result = fw.run_simulation("Pi Estimation",
                                  55_000,
                                  n_points=30_000,
                                  parallel=True,
                                  progress_callback=progress,
                                  compute_stats=True,
                                  confidence=0.95,
                                  ci_method="auto",
                                  extra_context={"target": float(np.pi),
                                                 # used by bias/mse/markov
                                                 "eps"   : 0.001,
                                                 # tolerance for Markov + req_n
                                                 }, )
    
    print("Running Portfolio Simulation…")
    port_res = fw.run_simulation("Portfolio Simulation",
                                 55_000,
                                 initial_value=10_000,
                                 annual_return=0.07,
                                 volatility=0.20,
                                 years=10,
                                 parallel=True,
                                 progress_callback=progress,
                                 compute_stats=True,
                                 confidence=0.95,
                                 ci_method="auto",
                                 extra_context={"target": float(19_671.51),
                                                # used by bias/mse/markov
                                                "eps"   : 1,
                                                # tolerance for Markov + req_n
                                                }, )
    
    print("\n" + "*" * 50)
    print("COMPARISON METRICS:")
    comparison = fw.compare_results(["Pi Estimation", "Portfolio Simulation"])
    for name, value in comparison.items():
        print(f"  {name}: {value:.5f}")
    print("*" * 50 + "\n")
    
    print(pi_result.result_to_string())
    print("\n")
    print(port_res.result_to_string())
    
    # Create comprehensive visualizations
    print("\nGenerating visualizations...")
    
    # Set matplotlib style for better-looking plots
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Create Pi estimation visualizations
    pi_fig = create_pi_visualizations(pi_result)
    
    # Create Portfolio visualizations
    port_fig = create_portfolio_visualizations(port_res)
    
    # Create comparison chart
    comp_fig = create_comparison_chart(pi_result, port_res)
    
    # Show all plots
    plt.show()
    
    # Optionally save plots
    save_plots = input("\nSave plots to files? (y/N): ").lower().strip() == 'y'
    if save_plots:
        pi_fig.savefig('pi_estimation_analysis.png',
                       bbox_inches='tight',
                       dpi=300)
        port_fig.savefig('portfolio_analysis.png',
                         bbox_inches='tight',
                         dpi=300)
        comp_fig.savefig('simulation_comparison.png',
                         bbox_inches='tight',
                         dpi=300)
        print("Plots saved as PNG files!")


if __name__ == "__main__":
    
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
