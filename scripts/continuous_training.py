#!/usr/bin/env python3
"""
Continuous Training Script - Progressive Scale Testing

Runs experiments at increasing scales following the validated methodology:
1. Demo (100 samples) - Validate infrastructure
2. Small (1K samples) - Quick validation
3. Medium (50K samples) - Initial comparison
4. Large (500K samples) - Scale test
5. Production (5M+ samples) - Full training

Automatically updates WIN_LOSE_ANALYSIS.md with results.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


SCALES = {
    "demo": {"samples": 100, "epochs": 5, "batch_size": 2, "desc": "Infrastructure validation"},
    "small": {"samples": 1000, "epochs": 10, "batch_size": 4, "desc": "Quick validation"},
    "medium": {"samples": 50000, "epochs": 20, "batch_size": 8, "desc": "Initial comparison"},
    "large": {"samples": 500000, "epochs": 30, "batch_size": 16, "desc": "Scale test"},
    "production": {"samples": 5000000, "epochs": 50, "batch_size": 32, "desc": "Full training"}
}


def run_experiment(scale_name, device="cpu", base_dir="./experiments"):
    """Run experiment at specified scale"""
    
    if scale_name not in SCALES:
        print(f"Error: Unknown scale '{scale_name}'. Available: {list(SCALES.keys())}")
        return False
    
    config = SCALES[scale_name]
    exp_dir = Path(base_dir) / f"{scale_name}_comparison"
    
    print("=" * 70)
    print(f"Starting {scale_name.upper()} Scale Experiment")
    print("=" * 70)
    print(f"Description: {config['desc']}")
    print(f"Samples: {config['samples']:,}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Device: {device}")
    print(f"Output: {exp_dir}")
    print("=" * 70)
    
    # Build command
    cmd = [
        sys.executable,
        "scripts/run_key_experiment.py",
        "--num-samples", str(config['samples']),
        "--epochs", str(config['epochs']),
        "--batch-size", str(config['batch_size']),
        "--device", device,
        "--save-dir", str(exp_dir)
    ]
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {scale_name.upper()} experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {scale_name.upper()} experiment failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n✗ {scale_name.upper()} experiment interrupted by user")
        return False


def update_analysis_doc(scale_name, results_path):
    """Update WIN_LOSE_ANALYSIS.md with experiment results"""
    
    analysis_file = Path("WIN_LOSE_ANALYSIS.md")
    if not analysis_file.exists():
        print("Warning: WIN_LOSE_ANALYSIS.md not found, skipping update")
        return
    
    results_file = Path(results_path) / "comparison_results.json"
    if not results_file.exists():
        print(f"Warning: Results file {results_file} not found, skipping update")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Extract key metrics
        nf_metrics = results['models']['neural_field']
        baseline_metrics = results['models']['baseline_transformer']
        comparison = results['comparison']
        
        # Create summary
        summary = f"""
### {scale_name.capitalize()} Scale ({results['config']['num_samples']:,} Samples)

**Status**: Complete

**Configuration:**
- Samples: {results['config']['num_samples']:,}
- Epochs: [From training]
- Batch Size: [From training]
- Device: CPU/GPU

**Results:**

**Neural Field:**
- Parameters: {nf_metrics['parameters']:,}
- Best Val Loss: {nf_metrics['best_val_loss']:.4f}
- Best Val Perplexity: {nf_metrics['best_val_perplexity']:.2f}
- Final Val Loss: {nf_metrics['final_val_loss']:.4f}
- Final Val Perplexity: {nf_metrics['final_val_perplexity']:.2f}

**Baseline Transformer:**
- Parameters: {baseline_metrics['parameters']:,}
- Best Val Loss: {baseline_metrics['best_val_loss']:.4f}
- Best Val Perplexity: {baseline_metrics['best_val_perplexity']:.2f}
- Final Val Loss: {baseline_metrics['final_val_loss']:.4f}
- Final Val Perplexity: {baseline_metrics['final_val_perplexity']:.2f}

**Comparison:**
- Winner: {comparison['winner'].replace('_', ' ').title()}
- NF Better: {'Yes' if comparison['nf_better_loss'] else 'No'}
- Perplexity Improvement: {comparison['perplexity_improvement_percent']:+.2f}%

**Analysis:**
- [Add interpretation here based on results]

**Timestamp**: {comparison['timestamp']}

---
"""
        
        print(f"\n{scale_name.capitalize()} Scale Results Summary:")
        print(summary)
        
        print(f"\nResults saved to: {results_file}")
        print("Update WIN_LOSE_ANALYSIS.md manually with the above summary")
        
    except Exception as e:
        print(f"Error updating analysis: {e}")


def run_progressive_training(start_scale="demo", device="cpu", base_dir="./experiments"):
    """
    Run progressive training starting from specified scale.
    Continues to next scale only if current scale completes successfully.
    """
    
    scale_order = ["demo", "small", "medium", "large", "production"]
    
    if start_scale not in scale_order:
        print(f"Error: Invalid start scale '{start_scale}'")
        return
    
    start_idx = scale_order.index(start_scale)
    
    print("\n" + "=" * 70)
    print("PROGRESSIVE TRAINING PROTOCOL")
    print("=" * 70)
    print(f"Starting from: {start_scale}")
    print(f"Remaining scales: {scale_order[start_idx:]}")
    print(f"Device: {device}")
    print("=" * 70)
    print("\nNote: Each scale proceeds only if previous completes successfully")
    print("This ensures we don't waste compute on unproven approaches")
    print("=" * 70)
    
    input("\nPress Enter to begin, or Ctrl+C to cancel...")
    
    for scale in scale_order[start_idx:]:
        success = run_experiment(scale, device=device, base_dir=base_dir)
        
        if success:
            exp_dir = Path(base_dir) / f"{scale}_comparison"
            update_analysis_doc(scale, exp_dir)
            
            # Ask if should continue (except for last scale)
            if scale != scale_order[-1]:
                next_scale = scale_order[scale_order.index(scale) + 1]
                next_config = SCALES[next_scale]
                
                print(f"\n{'=' * 70}")
                print(f"Next: {next_scale.upper()} scale")
                print(f"  Samples: {next_config['samples']:,}")
                print(f"  Estimated time: [Depends on hardware]")
                print(f"{'=' * 70}")
                
                response = input("\nContinue to next scale? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Stopping progressive training")
                    break
        else:
            print(f"\n{'=' * 70}")
            print(f"✗ {scale.upper()} scale failed or was interrupted")
            print("Stopping progressive training")
            print(f"{'=' * 70}")
            break
    
    print("\n" + "=" * 70)
    print("PROGRESSIVE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results saved in: {base_dir}")
    print("Review WIN_LOSE_ANALYSIS.md for detailed findings")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Run progressive neural field training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python scripts/continuous_training.py --scale demo
  
  # Run progressive training starting from small
  python scripts/continuous_training.py --progressive --start small --device cuda
  
  # Run specific scale on GPU
  python scripts/continuous_training.py --scale medium --device cuda

Available scales: demo, small, medium, large, production
"""
    )
    
    parser.add_argument("--scale", type=str, choices=list(SCALES.keys()),
                       help="Run single experiment at specified scale")
    parser.add_argument("--progressive", action="store_true",
                       help="Run progressive training (multiple scales)")
    parser.add_argument("--start", type=str, default="demo",
                       choices=list(SCALES.keys()),
                       help="Starting scale for progressive training")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--base-dir", type=str, default="./experiments",
                       help="Base directory for experiments")
    
    args = parser.parse_args()
    
    if args.progressive:
        run_progressive_training(
            start_scale=args.start,
            device=args.device,
            base_dir=args.base_dir
        )
    elif args.scale:
        success = run_experiment(
            args.scale,
            device=args.device,
            base_dir=args.base_dir
        )
        if success:
            exp_dir = Path(args.base_dir) / f"{args.scale}_comparison"
            update_analysis_doc(args.scale, exp_dir)
    else:
        parser.print_help()
        print("\nError: Must specify either --scale or --progressive")
        sys.exit(1)


if __name__ == "__main__":
    main()
