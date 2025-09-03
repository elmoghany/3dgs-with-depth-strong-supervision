#!/usr/bin/env python3
"""
Comprehensive training comparison tool for weak vs strong depth supervision.
Generates detailed loss curves and statistical analysis.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_tensorboard_data(logdir):
    """Extract training metrics from tensorboard logs"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Please install tensorboard: pip install tensorboard")
        return None
        
    logdir = Path(logdir)
    
    # Find tensorboard log files
    tb_files = list(logdir.glob("**/events.out.tfevents.*"))
    if not tb_files:
        print(f"No tensorboard files found in {logdir}")
        return None
        
    # Load the most recent log file
    latest_log = max(tb_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading tensorboard data from: {latest_log}")
    
    ea = EventAccumulator(str(latest_log.parent))
    ea.Reload()
    
    # Extract available scalar tags
    scalar_tags = ea.Tags()['scalars']
    
    data = {}
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data

def plot_comparison(weak_data, strong_data, output_dir):
    """Generate comparison plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Define key metrics to compare
    key_metrics = [
        ('train_loss_patches/total_loss', 'Total Training Loss', 'Loss'),
        ('train_loss_patches/l1_loss', 'L1 Training Loss', 'L1 Loss'),
        ('train/loss_viewpoint - l1_loss', 'Validation L1 Loss', 'L1 Loss'),
        ('train/loss_viewpoint - psnr', 'Validation PSNR', 'PSNR (dB)'),
        ('total_points', 'Total Gaussians', 'Count'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (metric, title, ylabel) in enumerate(key_metrics):
        ax = axes[idx]
        
        # Plot weak supervision
        if weak_data and metric in weak_data:
            steps_w = weak_data[metric]['steps']
            values_w = weak_data[metric]['values']
            ax.plot(steps_w, values_w, 'b-', label='Weak Supervision', linewidth=2)
        
        # Plot strong supervision
        if strong_data and metric in strong_data:
            steps_s = strong_data[metric]['steps']
            values_s = strong_data[metric]['values']
            ax.plot(steps_s, values_s, 'r-', label='Strong Supervision', linewidth=2)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(key_metrics) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_dir / 'training_comparison.png'}")
    
    return fig

def compute_statistics(weak_data, strong_data, output_dir):
    """Compute detailed statistical comparison"""
    output_dir = Path(output_dir)
    
    stats = {
        'weak_supervision': {},
        'strong_supervision': {},
        'comparison': {}
    }
    
    # Metrics to analyze
    metrics_to_analyze = [
        'train_loss_patches/total_loss',
        'train_loss_patches/l1_loss', 
        'train/loss_viewpoint - l1_loss',
        'train/loss_viewpoint - psnr',
        'total_points'
    ]
    
    for metric in metrics_to_analyze:
        stats['weak_supervision'][metric] = {}
        stats['strong_supervision'][metric] = {}
        stats['comparison'][metric] = {}
        
        # Weak supervision stats
        if weak_data and metric in weak_data:
            values_w = weak_data[metric]['values']
            if values_w:
                stats['weak_supervision'][metric] = {
                    'final_value': values_w[-1],
                    'min_value': min(values_w),
                    'max_value': max(values_w),
                    'mean_value': np.mean(values_w),
                    'std_value': np.std(values_w),
                    'total_iterations': len(values_w)
                }
        
        # Strong supervision stats  
        if strong_data and metric in strong_data:
            values_s = strong_data[metric]['values']
            if values_s:
                stats['strong_supervision'][metric] = {
                    'final_value': values_s[-1],
                    'min_value': min(values_s),
                    'max_value': max(values_s),
                    'mean_value': np.mean(values_s),
                    'std_value': np.std(values_s),
                    'total_iterations': len(values_s)
                }
        
        # Comparison stats
        if (weak_data and metric in weak_data and strong_data and metric in strong_data):
            values_w = weak_data[metric]['values']
            values_s = strong_data[metric]['values']
            
            if values_w and values_s:
                final_improvement = values_s[-1] - values_w[-1]
                relative_improvement = (final_improvement / values_w[-1]) * 100 if values_w[-1] != 0 else 0
                
                stats['comparison'][metric] = {
                    'absolute_improvement': final_improvement,
                    'relative_improvement_percent': relative_improvement,
                    'strong_better': final_improvement > 0 if 'psnr' in metric else final_improvement < 0,
                }
    
    # Save statistics
    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPARISON SUMMARY")
    print("="*80)
    
    for metric in metrics_to_analyze:
        if metric in stats['comparison'] and stats['comparison'][metric]:
            comp = stats['comparison'][metric]
            print(f"\n{metric}:")
            print(f"  Final improvement: {comp['absolute_improvement']:.4f}")
            print(f"  Relative improvement: {comp['relative_improvement_percent']:.2f}%")
            print(f"  Strong supervision better: {comp['strong_better']}")
    
    print(f"\nDetailed statistics saved to: {output_dir / 'comparison_summary.json'}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Compare weak vs strong supervision training")
    parser.add_argument("--weak", required=True, help="Path to weak supervision model directory")
    parser.add_argument("--strong", required=True, help="Path to strong supervision model directory")  
    parser.add_argument("--output", default="comparison_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("Extracting tensorboard data...")
    weak_data = extract_tensorboard_data(args.weak)
    strong_data = extract_tensorboard_data(args.strong)
    
    if not weak_data and not strong_data:
        print("No tensorboard data found in either directory!")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    print("Generating comparison plots...")
    plot_comparison(weak_data, strong_data, output_dir)
    
    print("Computing statistics...")
    compute_statistics(weak_data, strong_data, output_dir)
    
    print("\nComparison complete! Check the output directory for results.")

if __name__ == "__main__":
    main()