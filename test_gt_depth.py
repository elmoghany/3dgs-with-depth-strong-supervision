#!/usr/bin/env python3
"""
Test script to validate GT depth supervision implementation.
This tests the argument parsing and basic functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def test_gt_depth_args():
    """Test that GT depth arguments are properly added and parsed."""
    parser = ArgumentParser(description="Test GT depth arguments")
    
    # Add parameter groups
    lp = ModelParams(parser)
    op = OptimizationParams(parser) 
    pp = PipelineParams(parser)
    
    # Test default values
    test_args = []
    args = parser.parse_args(test_args)
    
    opt = op.extract(args)
    
    # Check that new GT depth arguments exist and have correct defaults
    assert hasattr(opt, 'depth_dir'), "depth_dir argument missing"
    assert hasattr(opt, 'depth_format'), "depth_format argument missing"
    assert hasattr(opt, 'depth_units'), "depth_units argument missing"
    assert hasattr(opt, 'depth_weight'), "depth_weight argument missing"
    assert hasattr(opt, 'depth_loss'), "depth_loss argument missing"
    assert hasattr(opt, 'depth_grad_weight'), "depth_grad_weight argument missing"
    assert hasattr(opt, 'depth_warmup'), "depth_warmup argument missing"
    assert hasattr(opt, 'depth_valid_min'), "depth_valid_min argument missing"
    assert hasattr(opt, 'depth_valid_max'), "depth_valid_max argument missing"
    
    # Check default values
    assert opt.depth_dir == "", f"Expected empty string, got {opt.depth_dir}"
    assert opt.depth_format == "png16", f"Expected 'png16', got {opt.depth_format}"
    assert opt.depth_units == "meters", f"Expected 'meters', got {opt.depth_units}"
    assert opt.depth_weight == 2.0, f"Expected 2.0, got {opt.depth_weight}"
    assert opt.depth_loss == "huber", f"Expected 'huber', got {opt.depth_loss}"
    assert opt.depth_grad_weight == 0.1, f"Expected 0.1, got {opt.depth_grad_weight}"
    assert opt.depth_warmup == 2000, f"Expected 2000, got {opt.depth_warmup}"
    assert opt.depth_valid_min == 1e-4, f"Expected 1e-4, got {opt.depth_valid_min}"
    assert opt.depth_valid_max == 80.0, f"Expected 80.0, got {opt.depth_valid_max}"
    
    print("✓ All GT depth arguments are correctly configured with expected defaults")
    
    # Test custom values
    test_args_custom = [
        "--depth_dir", "/path/to/depths",
        "--depth_format", "exr", 
        "--depth_units", "millimeters",
        "--depth_weight", "3.0",
        "--depth_loss", "l1",
        "--depth_grad_weight", "0.2",
        "--depth_warmup", "1000",
        "--depth_valid_min", "0.1",
        "--depth_valid_max", "100.0"
    ]
    
    args_custom = parser.parse_args(test_args_custom)
    opt_custom = op.extract(args_custom)
    
    assert opt_custom.depth_dir == "/path/to/depths"
    assert opt_custom.depth_format == "exr"
    assert opt_custom.depth_units == "millimeters"
    assert opt_custom.depth_weight == 3.0
    assert opt_custom.depth_loss == "l1"
    assert opt_custom.depth_grad_weight == 0.2
    assert opt_custom.depth_warmup == 1000
    assert opt_custom.depth_valid_min == 0.1
    assert opt_custom.depth_valid_max == 100.0
    
    print("✓ Custom GT depth argument values are correctly parsed")

def test_import_updated_modules():
    """Test that all modified modules can be imported without errors."""
    try:
        from scene import Scene
        from scene.dataset_readers import _read_gt_depth, readColmapSceneInfo
        from scene.cameras import Camera
        from utils.camera_utils import loadCam, cameraList_from_camInfos
        from gaussian_renderer import render
        print("✓ All modified modules import successfully")
    except Exception as e:
        print(f"✗ Import error: {e}")
        raise

if __name__ == "__main__":
    print("Testing GT depth supervision implementation...")
    print()
    
    test_import_updated_modules()
    test_gt_depth_args()
    
    print()
    print("🎉 All tests passed! GT depth supervision is ready to use.")
    print()
    print("Usage example:")
    print("python train.py -s <scene> -m <output> \\")
    print("  --depth_dir /path/to/gt_depths \\")
    print("  --depth_format png16 --depth_units meters \\")
    print("  --depth_weight 2.0 --depth_loss huber \\")
    print("  --depth_warmup 2000 --depth_grad_weight 0.1")