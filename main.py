import argparse
import os
import sys
import torch

from src.inference.quantize import (
    get_model,
    get_dataloaders,
    quantize_static,
    load_quantized_model,
    run_benchmarking_and_report
)


def main():
    parser = argparse.ArgumentParser(
        description="Fruits Ripeness Classification - Quantization & Inference Benchmarking (CPU)"
    )
    
    # Execution modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--quantize",
        type=str,
        metavar="FLOAT_CKPT",
        help="Perform static quantization on the provided float model checkpoint path."
    )
    group.add_argument(
        "--benchmark",
        type=str,
        nargs=2,
        metavar=("FLOAT_CKPT", "QUANTIZED_CKPT"),
        help="Run CPU benchmark comparing the Float32 model and the Quantized INT8 model."
    )
    group.add_argument(
        "--run-all",
        type=str,
        metavar="FLOAT_CKPT",
        help="Quantize the float model checkpoint AND benchmark the resulting INT8 model."
    )
    
    # Options
    parser.add_argument(
        "--out-quantized",
        type=str,
        default="models/fruit_model_quantized.pth",
        help="Output path for the quantized checkpoint. (Default: models/fruit_model_quantized.pth)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to the dataset directory containing Train/ and Test/ folders. (Default: data)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for calibration and evaluation. (Default: 32)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["fbgemm", "qnnpack"],
        default="fbgemm",
        help="Quantization backend engine. Choose 'fbgemm' (x86) or 'qnnpack' (ARM/mobile). (Default: fbgemm)"
    )
    
    args = parser.parse_args()
    
    # 0. Ensure target output directory exists
    os.makedirs(os.path.dirname(args.out_quantized), exist_ok=True)
    
    # 1. Load Data
    print(f"Loading data from: {args.data_dir}...")
    train_loader, test_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=2
    )
    
    # 2. Execution logic
    if args.quantize:
        # Static Quantization Mode
        float_ckpt = args.quantize
        # We need a calibration dataset. Use train_loader if available, otherwise test_loader.
        calib_loader = train_loader if train_loader is not None else test_loader
        if calib_loader is None:
            print("Error: No data available for calibration. Please make sure the dataset is in data/.", file=sys.stderr)
            sys.exit(1)
            
        # Load the Float model architecture + checkpoints weights
        float_model = get_model(checkpoint_path=float_ckpt, num_classes=3, pretrained=False)
        
        # Quantize the model
        quantized_model = quantize_static(float_model, calib_loader, backend=args.backend)
        
        # Save quantized model
        torch.save(quantized_model.state_dict(), args.out_quantized)
        print(f"Quantized INT8 model state dictionary saved to: {args.out_quantized}")
        
    elif args.benchmark:
        # Benchmark Mode
        float_ckpt, quantized_ckpt = args.benchmark
        if test_loader is None:
            print("Error: Test dataset is required for benchmarking. Check data/Test/ folder.", file=sys.stderr)
            sys.exit(1)
            
        # Load both models
        float_model = get_model(checkpoint_path=float_ckpt, num_classes=3, pretrained=False)
        quantized_model = load_quantized_model(quantized_ckpt, num_classes=3, backend=args.backend)
        
        # Run benchmark
        run_benchmarking_and_report(
            float_model=float_model,
            quantized_model=quantized_model,
            float_path=float_ckpt,
            quantized_path=quantized_ckpt,
            test_loader=test_loader
        )
        
    elif args.run_all:
        # Quantize + Benchmark Mode
        float_ckpt = args.run_all
        calib_loader = train_loader if train_loader is not None else test_loader
        if calib_loader is None or test_loader is None:
            print("Error: Calibration and test datasets are both required for this pipeline.", file=sys.stderr)
            sys.exit(1)
            
        # A. Quantize
        float_model = get_model(checkpoint_path=float_ckpt, num_classes=3, pretrained=False)
        quantized_model = quantize_static(float_model, calib_loader, backend=args.backend)
        torch.save(quantized_model.state_dict(), args.out_quantized)
        print(f"Quantized model saved to: {args.out_quantized}")
        
        # B. Reload models clean to avoid state contamination
        float_model_eval = get_model(checkpoint_path=float_ckpt, num_classes=3, pretrained=False)
        quantized_model_eval = load_quantized_model(args.out_quantized, num_classes=3, backend=args.backend)
        
        # C. Benchmark
        run_benchmarking_and_report(
            float_model=float_model_eval,
            quantized_model=quantized_model_eval,
            float_path=float_ckpt,
            quantized_path=args.out_quantized,
            test_loader=test_loader
        )


if __name__ == "__main__":
    main()
