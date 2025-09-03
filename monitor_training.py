#!/usr/bin/env python3
"""
Real-time training monitor for 3DGS training progress.
Shows live updates of loss curves and training statistics.
"""

import time
import json
import os
import argparse
from pathlib import Path

def get_latest_metrics(model_dir):
    """Extract the latest training metrics"""
    model_path = Path(model_dir)
    
    # Try to find tensorboard data
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        tb_files = list(model_path.glob("**/events.out.tfevents.*"))
        if tb_files:
            latest_log = max(tb_files, key=lambda f: f.stat().st_mtime)
            ea = EventAccumulator(str(latest_log.parent))
            ea.Reload()
            
            metrics = {}
            scalar_tags = ea.Tags()['scalars']
            
            for tag in ['train_loss_patches/total_loss', 'train_loss_patches/l1_loss', 
                       'test/loss_viewpoint - psnr', 'total_points']:
                if tag in scalar_tags:
                    scalar_events = ea.Scalars(tag)
                    if scalar_events:
                        metrics[tag] = {
                            'latest_step': scalar_events[-1].step,
                            'latest_value': scalar_events[-1].value,
                            'total_points': len(scalar_events)
                        }
            
            return metrics
            
    except ImportError:
        pass
    
    return None

def monitor_training(model_dir, refresh_interval=30):
    """Monitor training progress in real-time"""
    print(f"Monitoring training in: {model_dir}")
    print(f"Refresh interval: {refresh_interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            metrics = get_latest_metrics(model_dir)
            
            if metrics:
                print(f"\n{'='*60}")
                print(f"Training Progress - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")
                
                for metric_name, data in metrics.items():
                    display_name = metric_name.replace('train_loss_patches/', '').replace('test/loss_viewpoint - ', 'Test ')
                    print(f"{display_name:20s}: {data['latest_value']:.6f} (iter {data['latest_step']})")
                
                print(f"{'='*60}")
            else:
                print(f"No metrics found at {time.strftime('%H:%M:%S')} - training may not have started yet...")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

def main():
    parser = argparse.ArgumentParser(description="Monitor 3DGS training progress")
    parser.add_argument("model_dir", help="Path to model directory to monitor")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Model directory does not exist: {args.model_dir}")
        return
    
    monitor_training(args.model_dir, args.refresh)

if __name__ == "__main__":
    main()