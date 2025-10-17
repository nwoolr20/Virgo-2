#!/usr/bin/env python3
"""
Run All Training - Complete Progressive Training Pipeline

This script runs all training scales from demo to medium, trains models,
saves all neural fields, and documents results comprehensively.

Following the empirical validation strategy:
1. Demo (100 samples) - Infrastructure validation
2. Small (1K samples) - Quick validation  
3. Medium (50K samples) - Initial comparison

All trained models are saved for analysis and use.
"""

import argparse
import subprocess
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        return False
    
    print(f"\n✅ Completed: {description}")
    return True


def run_single_scale(scale, device, base_dir):
    """Run training for a single scale"""
    
    scales_config = {
        "demo": {"samples": 100, "epochs": 5, "batch_size": 2},
        "small": {"samples": 1000, "epochs": 10, "batch_size": 4},
        "medium": {"samples": 10000, "epochs": 15, "batch_size": 8},
    }
    
    if scale not in scales_config:
        print(f"Error: Unknown scale '{scale}'")
        return False
    
    config = scales_config[scale]
    exp_dir = Path(base_dir) / f"{scale}_training"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {scale.upper()} Scale")
    print(f"{'='*70}")
    print(f"Samples: {config['samples']:,}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Output: {exp_dir}")
    print(f"{'='*70}\n")
    
    # Run the key experiment (trains both neural field and baseline)
    cmd = [
        sys.executable,
        "scripts/run_key_experiment.py",
        "--num-samples", str(config['samples']),
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--device", device,
        "--save-dir", str(exp_dir)
    ]
    
    success = run_command(cmd, f"{scale.upper()} scale training")
    
    if success:
        # List saved models
        print(f"\n{'='*70}")
        print(f"SAVED MODELS for {scale.upper()} scale:")
        print(f"{'='*70}")
        
        for model_dir in exp_dir.glob("*"):
            if model_dir.is_dir():
                print(f"\n📁 {model_dir.name}/")
                for model_file in model_dir.glob("*.pt"):
                    size_mb = model_file.stat().st_size / (1024 * 1024)
                    print(f"   ✓ {model_file.name} ({size_mb:.1f} MB)")
                
                # Check for training history
                history_file = model_dir / "training_history.json"
                if history_file.exists():
                    with open(history_file) as f:
                        history = json.load(f)
                    print(f"   ✓ training_history.json")
                    if "train_losses" in history and history["train_losses"]:
                        initial_loss = history["train_losses"][0]
                        final_loss = history["train_losses"][-1]
                        improvement = ((initial_loss - final_loss) / initial_loss) * 100
                        print(f"      Loss: {initial_loss:.4f} → {final_loss:.4f} ({improvement:.1f}% improvement)")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Run all progressive training scales and save neural fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scales on CPU (demo -> small -> medium)
  python scripts/run_all_training.py
  
  # Run specific scale only
  python scripts/run_all_training.py --scale demo
  
  # Run all scales on GPU
  python scripts/run_all_training.py --device cuda
  
This will train neural field models at increasing scales, save all trained
models, and document results. Models are saved in trained_models/ directory.
        """
    )
    
    parser.add_argument(
        "--scale",
        choices=["demo", "small", "medium", "all"],
        default="all",
        help="Training scale to run (default: all)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for training (default: cpu)"
    )
    parser.add_argument(
        "--base-dir",
        default="./trained_models",
        help="Base directory for trained models (default: ./trained_models)"
    )
    
    args = parser.parse_args()
    
    # Create base directory
    base_dir = Path(args.base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PROGRESSIVE NEURAL FIELD TRAINING")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Output Directory: {base_dir}")
    print(f"Strategy: Empirical validation at each scale")
    print("="*70)
    
    # Determine which scales to run
    if args.scale == "all":
        scales_to_run = ["demo", "small", "medium"]
    else:
        scales_to_run = [args.scale]
    
    print(f"\nScales to train: {', '.join(s.upper() for s in scales_to_run)}")
    
    # Run training for each scale
    results = {}
    for scale in scales_to_run:
        print(f"\n\n{'#'*70}")
        print(f"# STARTING: {scale.upper()} SCALE TRAINING")
        print(f"{'#'*70}\n")
        
        success = run_single_scale(scale, args.device, args.base_dir)
        results[scale] = success
        
        if not success:
            print(f"\n❌ Training failed at {scale.upper()} scale. Stopping.")
            break
        
        print(f"\n✅ {scale.upper()} scale training completed successfully!")
    
    # Summary
    print(f"\n\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")
    
    for scale, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{scale.upper():10} : {status}")
    
    # Document where models are saved
    print(f"\n{'='*70}")
    print("TRAINED MODELS LOCATION")
    print(f"{'='*70}")
    print(f"\nAll trained neural fields saved in: {base_dir.absolute()}/")
    print("\nDirectory structure:")
    print(f"  {base_dir.name}/")
    for scale in scales_to_run:
        if results.get(scale):
            print(f"    ├── {scale}_training/")
            print(f"    │     ├── neural_field/")
            print(f"    │     │     ├── checkpoint_best.pt")
            print(f"    │     │     ├── final_model.pt")
            print(f"    │     │     └── training_history.json")
            print(f"    │     └── baseline/")
            print(f"    │           ├── checkpoint_best.pt")
            print(f"    │           ├── final_model.pt")
            print(f"    │           └── training_history.json")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("\n1. Analyze results:")
    print(f"   python scripts/analyze_results.py --dir {base_dir}")
    print("\n2. Test trained models:")
    print(f"   python scripts/test_trained_model.py {base_dir}/demo_training/neural_field/final_model.pt")
    print("\n3. Compare models:")
    print("   See WIN_LOSE_ANALYSIS.md for detailed comparison framework")
    print("\n4. Continue to larger scales if validated:")
    print("   python scripts/run_all_training.py --scale large --device cuda")
    
    # Return success if all completed
    all_success = all(results.values())
    
    if all_success:
        print(f"\n✅ All training completed successfully!")
        print(f"✅ All neural fields saved to {base_dir.absolute()}/")
        return 0
    else:
        print(f"\n❌ Some training runs failed. Check logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
