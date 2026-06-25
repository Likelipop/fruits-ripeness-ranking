import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models.quantization as qmodels
from tqdm import tqdm


class FruitRipenessModel(nn.Module):
    """
    Fruit Ripeness Classification Model based on a quantizable ResNet18.
    This model wraps torchvision's QuantizableResNet, which is structurally
    prepared for static and dynamic quantization on CPU.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Load the quantizable resnet18 structure
        if pretrained:
            from torchvision.models import ResNet18_Weights
            self.model = qmodels.resnet18(weights=ResNet18_Weights.DEFAULT, quantize=False)
        else:
            self.model = qmodels.resnet18(weights=None, quantize=False)
        
        # Replace the final fully connected classification head
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)


def get_model(checkpoint_path=None, num_classes=3, pretrained=True):
    """
    Instantiates the model and optionally loads pre-trained state dict weights.
    """
    model = FruitRipenessModel(num_classes=num_classes, pretrained=pretrained)
    
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        print(f"Loading float model checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
    return model


def get_dataloaders(data_dir, batch_size=32, num_workers=2):
    """
    Creates standard ImageFolder datasets and PyTorch DataLoaders for train and test sets.
    The folders should contain subdirectories for classes: Overripe, Ripe, Unripe.
    """
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")

    # Define standard transforms for ResNet18
    # CenterCrop to 224 and normalization to ImageNet standards
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = None
    train_loader = None
    if os.path.exists(train_dir):
        train_dataset = datasets.ImageFolder(train_dir, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        print(f"Loaded training set from {train_dir} ({len(train_dataset)} images, classes: {train_dataset.classes})")
    else:
        print(f"Warning: Training directory not found at {train_dir}")

    test_dataset = None
    test_loader = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        print(f"Loaded test set from {test_dir} ({len(test_dataset)} images, classes: {test_dataset.classes})")
    else:
        print(f"Warning: Test directory not found at {test_dir}")

    return train_loader, test_loader


def quantize_static(model, calibration_loader, backend="fbgemm"):
    """
    Performs Post-Training Static Quantization (PTSQ) on a float PyTorch model.
    """
    print(f"\n--- Starting Static Quantization ({backend} backend) ---")
    
    # 1. Set model to evaluation mode
    model.eval()
    
    # 2. Fuse Modules
    print("Fusing modules (Conv + BatchNorm + ReLU)...")
    model.model.fuse_model()
    
    # 3. Configure the quantization settings
    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    print(f"Set qconfig to default for {backend}.")
    
    # 4. Prepare the model for static quantization calibration
    print("Preparing model for calibration...")
    torch.ao.quantization.prepare(model, inplace=True)
    
    # 5. Run Calibration
    print("Calibrating model on representative dataset subset...")
    num_calibration_batches = 6  # ~192 images at batch_size=32
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(calibration_loader, desc="Calibration")):
            model(images)
            if i >= num_calibration_batches - 1:
                break
                
    # 6. Convert the model to quantized INT8 weights and activations
    print("Converting model to quantized INT8 precision...")
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    print("Quantization complete.")
    
    return quantized_model


def load_quantized_model(checkpoint_path, num_classes=3, backend="fbgemm"):
    """
    Loads a saved quantized INT8 model state dictionary.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Quantized checkpoint not found at: {checkpoint_path}")
        
    print(f"Instantiating and preparing model architecture to load quantized weights from {checkpoint_path}...")
    # 1. Instantiate the float model
    model = get_model(num_classes=num_classes, pretrained=False)
    
    # 2. Fuse model
    model.model.fuse_model()
    
    # 3. Configure backend and prepare
    torch.backends.quantized.engine = backend
    model.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(model, inplace=True)
    
    # 4. Convert model to quantized structure
    quantized_model = torch.ao.quantization.convert(model, inplace=False)
    
    # 5. Load quantized state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    quantized_model.load_state_dict(state_dict)
    
    print("Loaded quantized model successfully.")
    return quantized_model


def evaluate_model(model, dataloader, desc="Evaluating"):
    """
    Evaluates model accuracy and average CPU inference latency per image.
    """
    model.eval()
    correct = 0
    total = 0
    inference_times = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            start_time = time.perf_counter()
            outputs = model(images)
            end_time = time.perf_counter()
            
            batch_time_ms = (end_time - start_time) * 1000
            time_per_image = batch_time_ms / images.size(0)
            inference_times.append(time_per_image)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    avg_latency_ms = sum(inference_times) / len(inference_times)
    
    return accuracy, avg_latency_ms


def get_file_size_mb(file_path):
    """Returns file size in Megabytes."""
    if not os.path.exists(file_path):
        return 0.0
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def run_benchmarking_and_report(float_model, quantized_model, float_path, quantized_path, test_loader):
    """
    Benchmarks and compares model size, accuracy, and latency,
    and automatically generates a `quantization_report.md` file.
    """
    print("\n--- Running Evaluation Benchmark on Local CPU ---")
    
    # Measure sizes
    float_size = get_file_size_mb(float_path)
    quantized_size = get_file_size_mb(quantized_path)
    
    # Evaluate Float Model
    print("Benchmarking Float32 Model...")
    float_accuracy, float_latency = evaluate_model(float_model, test_loader, desc="Float32 Model")
    
    # Evaluate Quantized Model
    print("Benchmarking Quantized INT8 Model...")
    quantized_accuracy, quantized_latency = evaluate_model(quantized_model, test_loader, desc="Quantized INT8 Model")
    
    # Calculate performance improvements
    size_reduction_ratio = float_size / quantized_size if quantized_size > 0 else 0.0
    speedup_ratio = float_latency / quantized_latency if quantized_latency > 0 else 0.0
    
    # Print clean table to console
    print("\n" + "="*60)
    print(f"{'Metric':<25} | {'Float32 Model':<15} | {'Quantized INT8':<15}")
    print("-"*60)
    print(f"{'Model File Size (MB)':<25} | {float_size:<15.2f} | {quantized_size:<15.2f} ({size_reduction_ratio:.1f}x smaller)")
    print(f"{'Accuracy on Test Set (%)':<25} | {float_accuracy:<15.2f} | {quantized_accuracy:<15.2f}")
    print(f"{'Inference Latency/Image':<25} | {float_latency:<15.2f} ms | {quantized_latency:<15.2f} ms ({speedup_ratio:.1f}x faster)")
    print("="*60 + "\n")
    
    # Generate markdown report
    report_content = f"""# Quantization Experiment & CPU Benchmarking Report

This report summarizes the performance comparison between the **Float32 PyTorch model** (trained via transfer learning on Google Colab) and the **Quantized INT8 model** (post-training static quantization calibrated locally on CPU).

## Experiment Configuration
- **Model Architecture**: Quantizable ResNet18 (`torchvision.models.quantization.resnet18`)
- **Dataset**: Fruits Ripeness Dataset (Classes: `Overripe`, `Ripe`, `Unripe`)
- **Quantization Technique**: Post-Training Static Quantization (PTSQ)
- **Calibration Engine / Backend**: `fbgemm` (x86 CPU optimized)

## Benchmarking Results

| Evaluation Metric | Float32 Model | Quantized INT8 Model | Comparison / Optimization Ratio |
| :--- | :---: | :---: | :---: |
| **Model Size (MB)** | {float_size:.2f} MB | {quantized_size:.2f} MB | **{size_reduction_ratio:.2f}x** reduction in storage |
| **Test Accuracy (%)** | {float_accuracy:.2f}% | {quantized_accuracy:.2f}% | **{quantized_accuracy - float_accuracy:+.2f}%** change in accuracy |
| **CPU Latency per Image** | {float_latency:.2f} ms | {quantized_latency:.2f} ms | **{speedup_ratio:.2f}x** speedup on local inference |

## Key Insights
1. **Size Optimization**: Quantization compressed the model parameters from 32-bit floats to 8-bit integers, yielding a **{size_reduction_ratio:.1f}x** decrease in binary footprint. This makes the model highly deployable on CPU-bound edge devices.
2. **Latency Speedup**: The INT8 matrix multiplication execution on CPU led to a **{speedup_ratio:.1f}x** inference speedup per image.
3. **Accuracy Retention**: Quantization introduces minor numerical precision errors. The accuracy shifted by **{quantized_accuracy - float_accuracy:+.2f}%**, demonstrating that Post-Training Static Quantization preserves performance very well.

*Report auto-generated on {time.strftime('%Y-%m-%d %H:%M:%S')} local time.*
"""
    
    report_path = "quantization_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Successfully generated markdown report at: {os.path.abspath(report_path)}")
