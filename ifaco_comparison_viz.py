"""
Enhanced Visualization Module for IFACO Comparison Study
Specialized visualizations for comparing Standalone AFSA, Standalone ACO, and Hybrid IFACO
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, Any, List
import json

class ComparisonVisualizer:
    """Advanced visualizations for comparison study"""

    def __init__(self, results: Dict[str, Any]):
        """
        Initialize comparison visualizer

        Args:
            results: Complete comparison study results dictionary
        """
        self.results = results
        self.colors = {
            'standalone_afsa': '#3498db',  # Blue
            'standalone_aco': '#e74c3c',   # Red
            'hybrid_ifaco': '#2ecc71'      # Green
        }
        self.labels = {
            'standalone_afsa': 'Standalone AFSA',
            'standalone_aco': 'Standalone ACO',
            'hybrid_ifaco': 'Hybrid IFACO'
        }

        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_performance_comparison(self, save_path: str = 'performance_comparison.png',
                                   show_plot: bool = True):
        """
        Create comprehensive performance comparison chart
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        variants = ['standalone_afsa', 'standalone_aco', 'hybrid_ifaco']

        # Extract data
        fitness_data = []
        distance_data = []
        emission_data = []
        route_data = []
        time_data = []
        labels_list = []
        colors_list = []

        for variant in variants:
            if 'error' not in self.results[variant]:
                fitness_data.append(self.results[variant]['best_fitness'])
                distance_data.append(self.results[variant]['best_distance'])
                emission_data.append(self.results[variant]['best_emission'])
                route_data.append(self.results[variant]['num_routes'])
                time_data.append(self.results[variant]['execution_time'])
                labels_list.append(self.labels[variant])
                colors_list.append(self.colors[variant])

        x = np.arange(len(labels_list))

        # Plot 1: Fitness Comparison
        bars1 = ax1.bar(x, fitness_data, color=colors_list, edgecolor='black',
                       linewidth=2, alpha=0.8)
        ax1.set_ylabel('Multi-Objective Fitness', fontsize=12, fontweight='bold')
        ax1.set_title('Overall Fitness Comparison\n(Lower is Better)',
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels_list, rotation=15, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels and mark best
        best_idx = fitness_data.index(min(fitness_data))
        for i, (bar, val) in enumerate(zip(bars1, fitness_data)):
            height = bar.get_height()
            label = f'{val:.2f}'
            if i == best_idx:
                label += ' ⭐'
            ax1.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

        # Plot 2: Distance vs Emissions (Scatter)
        scatter = ax2.scatter(distance_data, emission_data, s=400, c=colors_list,
                            edgecolors='black', linewidths=2, alpha=0.7)

        for i, label in enumerate(labels_list):
            ax2.annotate(label.replace('Standalone ', '').replace('Hybrid ', ''),
                        (distance_data[i], emission_data[i]),
                        fontsize=9, fontweight='bold', ha='center', va='center',
                        color='white')

        ax2.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('CO₂ Emissions (kg)', fontsize=12, fontweight='bold')
        ax2.set_title('Distance vs Emissions Trade-off',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Component Breakdown (Stacked Bar)
        width = 0.6
        p1 = ax3.bar(x, distance_data, width, label='Distance',
                    color='#3498db', edgecolor='black', linewidth=1.5)
        p2 = ax3.bar(x, emission_data, width, bottom=distance_data,
                    label='Emissions', color='#e67e22', edgecolor='black', linewidth=1.5)

        ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax3.set_title('Fitness Components\n(Distance + Emissions)',
                     fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels_list, rotation=15, ha='right')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Execution Time vs Routes
        colors_by_time = [colors_list[i] for i in range(len(time_data))]
        bars4 = ax4.bar(x, time_data, color=colors_by_time, edgecolor='black',
                       linewidth=2, alpha=0.8)

        # Add route count as text on bars
        for i, (bar, routes) in enumerate(zip(bars4, route_data)):
            height = bar.get_height()
            ax4.annotate(f'{routes} routes',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            ax4.annotate(f'{time_data[i]:.1f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height/2),
                        ha='center', va='center',
                        fontsize=10, fontweight='bold', color='white')

        ax4.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax4.set_title('Computational Efficiency',
                     fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels_list, rotation=15, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved performance comparison to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_improvement_analysis(self, save_path: str = 'improvement_analysis.png',
                                  show_plot: bool = True):
        """
        Visualize improvement percentages between variants
        """
        if 'comparison' not in self.results or not self.results['comparison']:
            print("No comparison data available")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        comp = self.results['comparison']

        # Prepare improvement data
        improvements = []
        labels = []
        colors = []

        if 'standalone_aco_vs_afsa' in comp:
            imp = comp['standalone_aco_vs_afsa']['fitness_improvement_percent']
            improvements.append(imp)
            labels.append('ACO vs\nAFSA')
            colors.append('#2ecc71' if imp > 0 else '#e74c3c')

        if 'hybrid_ifaco_vs_afsa' in comp:
            imp = comp['hybrid_ifaco_vs_afsa']['fitness_improvement_percent']
            improvements.append(imp)
            labels.append('Hybrid vs\nAFSA')
            colors.append('#2ecc71' if imp > 0 else '#e74c3c')

        if 'hybrid_vs_standalone_aco' in comp:
            imp = comp['hybrid_vs_standalone_aco']['fitness_improvement_percent']
            improvements.append(imp)
            labels.append('Hybrid vs\nStandalone ACO')
            colors.append('#2ecc71' if imp > 0 else '#e74c3c')

        # Plot 1: Improvement bars
        x = np.arange(len(labels))
        bars = ax1.bar(x, improvements, color=colors, edgecolor='black',
                      linewidth=2, alpha=0.8, width=0.6)

        ax1.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax1.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Fitness Improvement Analysis\n(Positive = Better)',
                     fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            label = f'{imp:+.2f}%'
            y_pos = height + (5 if height > 0 else -15)
            va = 'bottom' if height > 0 else 'top'

            ax1.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, y_pos),
                        textcoords="offset points",
                        ha='center', va=va,
                        fontsize=11, fontweight='bold')

        # Plot 2: Fitness values comparison
        variants = ['standalone_afsa', 'standalone_aco', 'hybrid_ifaco']
        fitness_vals = []
        distance_vals = []
        emission_vals = []
        variant_labels = []

        for variant in variants:
            if 'error' not in self.results[variant]:
                fitness_vals.append(self.results[variant]['best_fitness'])
                distance_vals.append(self.results[variant]['best_distance'])
                emission_vals.append(self.results[variant]['best_emission'])
                variant_labels.append(self.labels[variant].replace('Standalone ', '').replace('Hybrid ', ''))

        x2 = np.arange(len(variant_labels))
        width = 0.25

        bars1 = ax2.bar(x2 - width, distance_vals, width, label='Distance',
                       color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x2, emission_vals, width, label='Emissions',
                       color='#e67e22', edgecolor='black', linewidth=1.5)
        bars3 = ax2.bar(x2 + width, fitness_vals, width, label='Total Fitness',
                       color='#9b59b6', edgecolor='black', linewidth=1.5)

        ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax2.set_title('Detailed Metric Comparison',
                     fontsize=13, fontweight='bold')
        ax2.set_xticks(x2)
        ax2.set_xticklabels(variant_labels, fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved improvement analysis to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_convergence_comparison(self, save_path: str = 'convergence_comparison.png',
                                   show_plot: bool = True):
        """
        Compare convergence behavior of all three variants
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        variants = ['standalone_afsa', 'standalone_aco', 'hybrid_ifaco']

        for variant in variants:
            if 'error' in self.results[variant]:
                continue

            # Get convergence history
            if 'fitness_history' in self.results[variant]:
                history = self.results[variant]['fitness_history']
                iterations = range(len(history))
                ax.plot(iterations, history, 
                       label=self.labels[variant],
                       color=self.colors[variant],
                       linewidth=2.5, marker='o', markersize=4,
                       markevery=max(1, len(history)//15))

            elif 'iteration_history' in self.results[variant]:
                history = self.results[variant]['iteration_history']
                # Filter out inf values
                history_filtered = [f if f != float('inf') else None for f in history]

                valid_points = [(i, f) for i, f in enumerate(history_filtered) if f is not None]
                if valid_points:
                    iter_valid, fitness_valid = zip(*valid_points)
                    ax.plot(iter_valid, fitness_valid,
                           label=self.labels[variant],
                           color=self.colors[variant],
                           linewidth=2.5, marker='s', markersize=4,
                           markevery=max(1, len(valid_points)//15))

        ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Multi-Objective Fitness', fontsize=13, fontweight='bold')
        ax.set_title('Convergence Behavior Comparison\n(Lower is Better)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved convergence comparison to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

        return fig

    def create_comparison_dashboard(self, output_dir: str = '.', show_plots: bool = False):
        """
        Create complete comparison dashboard with all visualizations
        """
        print("\n" + "="*60)
        print("GENERATING COMPARISON DASHBOARD")
        print("="*60)

        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("\n1. Creating performance comparison...")
        self.plot_performance_comparison(f'{output_dir}/performance_comparison.png', show_plots)

        print("\n2. Creating improvement analysis...")
        self.plot_improvement_analysis(f'{output_dir}/improvement_analysis.png', show_plots)

        print("\n3. Creating convergence comparison...")
        self.plot_convergence_comparison(f'{output_dir}/convergence_comparison.png', show_plots)

        print("\n" + "="*60)
        print("✅ COMPARISON DASHBOARD COMPLETE")
        print("="*60)
        print(f"\nFiles saved in: {output_dir}/")
        print("  - performance_comparison.png")
        print("  - improvement_analysis.png")
        print("  - convergence_comparison.png")

def visualize_comparison_from_json(json_file: str, output_dir: str = 'comparison_dashboard',
                                   show_plots: bool = False):
    """
    Generate comparison visualizations from saved JSON results

    Args:
        json_file: Path to comparison results JSON file
        output_dir: Directory to save visualization files
        show_plots: Whether to display plots interactively
    """
    with open(json_file, 'r') as f:
        results = json.load(f)

    visualizer = ComparisonVisualizer(results)
    visualizer.create_comparison_dashboard(output_dir, show_plots)

if __name__ == "__main__":
    print("IFACO Comparison Visualization Module")
    print("Usage:")
    print("  from ifaco_comparison_viz import visualize_comparison_from_json")
    print("  visualize_comparison_from_json('ifaco_comparison_results.json')")
