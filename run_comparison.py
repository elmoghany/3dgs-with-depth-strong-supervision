#!/usr/bin/env python3
"""
Run comparative training between weak and strong depth supervision.
This script helps set up both training runs with proper logging.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_training_comparison(scene_path, depth_images_path, gt_depth_path, output_base):
    """Run both weak and strong supervision training"""
    
    output_base = Path(output_base)
    output_base.mkdir(exist_ok=True)
    
    weak_output = output_base / "weak_supervision"
    strong_output = output_base / "strong_supervision"
    
    # Test iterations for frequent evaluation
    test_iters = "500 1000 2000 3000 5000 7000 10000 15000 20000 30000"
    
    print("="*80)
    print("STARTING COMPARATIVE TRAINING")
    print("="*80)
    print(f"Scene path: {scene_path}")
    print(f"Depth images (weak): {depth_images_path}")
    print(f"GT depth (strong): {gt_depth_path}")
    print(f"Output base: {output_base}")
    print("="*80)
    
    # Weak supervision command
    weak_cmd = [
        "python", "train.py",
        "-s", str(scene_path),
        "-d", str(depth_images_path),
        "-m", str(weak_output),
        "--test_iterations", *test_iters.split(),
        "--iterations", "30000"
    ]
    
    # Strong supervision command  
    strong_cmd = [
        "python", "train.py", 
        "-s", str(scene_path),
        "--depth_dir", str(gt_depth_path),
        "--depth_format", "png16",
        "--depth_units", "meters", 
        "--depth_weight", "3.5",
        "--depth_loss", "huber",
        "--depth_grad_weight", "0.2",
        "--depth_warmup", "1500",
        "-m", str(strong_output),
        "--test_iterations", *test_iters.split(),
        "--iterations", "30000"
    ]
    
    print("\n1. WEAK SUPERVISION TRAINING")
    print("Command:", " ".join(weak_cmd))
    print("\nStarting weak supervision training...")
    weak_process = subprocess.Popen(weak_cmd)
    
    print("\n2. STRONG SUPERVISION TRAINING")  
    print("Command:", " ".join(strong_cmd))
    print("\nStarting strong supervision training...")
    strong_process = subprocess.Popen(strong_cmd)
    
    print("\n" + "="*80)
    print("BOTH TRAINING PROCESSES STARTED")
    print("="*80)
    print(f"Weak supervision PID: {weak_process.pid}")
    print(f"Strong supervision PID: {strong_process.pid}")
    print("\nMonitoring commands:")
    print(f"python monitor_training.py {weak_output}")
    print(f"python monitor_training.py {strong_output}")
    print(f"\nTensorboard: tensorboard --logdir {output_base} --port 6006")
    print(f"\nComparison: python compare_training.py --weak {weak_output} --strong {strong_output}")
    print("="*80)
    
    try:
        # Wait for both processes
        weak_result = weak_process.wait()
        strong_result = strong_process.wait()
        
        print(f"\nTraining completed!")
        print(f"Weak supervision exit code: {weak_result}")
        print(f"Strong supervision exit code: {strong_result}")
        
        # Automatically run comparison
        print("\nRunning automatic comparison...")
        comparison_cmd = [
            "python", "compare_training.py",
            "--weak", str(weak_output),
            "--strong", str(strong_output), 
            "--output", str(output_base / "comparison_results")
        ]
        subprocess.run(comparison_cmd)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        weak_process.terminate()
        strong_process.terminate()

def main():
    parser = argparse.ArgumentParser(description="Run comparative depth supervision training")
    parser.add_argument("-s", "--scene", required=True, help="Scene path")
    parser.add_argument("-d", "--depth_images", required=True, help="Weak supervision depth images path")  
    parser.add_argument("--gt_depth", required=True, help="Strong supervision GT depth path")
    parser.add_argument("-o", "--output", default="comparison_training", help="Output base directory")
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.scene):
        print(f"Scene path does not exist: {args.scene}")
        return
    
    if not os.path.exists(args.depth_images):
        print(f"Depth images path does not exist: {args.depth_images}")
        return
        
    if not os.path.exists(args.gt_depth):
        print(f"GT depth path does not exist: {args.gt_depth}")
        return
    
    run_training_comparison(args.scene, args.depth_images, args.gt_depth, args.output)

if __name__ == "__main__":
    main()