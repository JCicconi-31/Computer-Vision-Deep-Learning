import os
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

SAMPLE_IMAGE_PATH = os.path.join(ASSETS_DIR, "sample_image.png")
OUTPUT_PLOT_PATH = os.path.join(ASSETS_DIR, "milestone_0_output.png")


def check_hardware():
    cuda_available = torch.cuda.is_available()
    print("=== Environment & Hardware ===")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {cuda_available}")

    if cuda_available:
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
        total_vram_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        allocated_mb = torch.cuda.memory_allocated(device_id) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(device_id) / (1024**2)

        print(f"GPU device      : {gpu_name}")
        print(f"Total VRAM      : {total_vram_gb:.2f} GB")
        print(f"Allocated VRAM  : {allocated_mb:.2f} MB")
        print(f"Reserved VRAM   : {reserved_mb:.2f} MB")
    else:
        print("Warning: Running on CPU.")
    print()


def get_or_create_sample_image(path=SAMPLE_IMAGE_PATH):
    if not os.path.exists(path):
        img = Image.new("RGB", (320, 240), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle([30, 30, 130, 130], fill=(220, 50, 50), outline=(0, 0, 0), width=2)
        draw.ellipse([170, 40, 270, 140], fill=(40, 180, 70), outline=(0, 0, 0), width=2)
        draw.polygon([(160, 160), (90, 220), (230, 220)], fill=(30, 100, 230), outline=(0, 0, 0))
        img.save(path)
    return path


def test_tensor_pipeline(image_path=SAMPLE_IMAGE_PATH):
    print("=== Visual Data & Tensor Exploration ===")
    
    # 1. Raw image
    pil_img = Image.open(image_path).convert("RGB")
    np_img = np.array(pil_img)
    print(f"NumPy shape (HWC) : {np_img.shape}, dtype: {np_img.dtype}, range: [{np_img.min()}, {np_img.max()}]")

    # 2. PyTorch tensor conversion
    to_tensor = T.ToTensor()
    tensor_img = to_tensor(pil_img)
    print(f"Tensor shape (CHW): {tensor_img.shape}, dtype: {tensor_img.dtype}, range: [{tensor_img.min():.4f}, {tensor_img.max():.4f}]")

    # 3. Normalization (ImageNet stats)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalized_tensor = normalize(tensor_img)
    print(f"Normalized range  : [{normalized_tensor.min():.4f}, {normalized_tensor.max():.4f}]")

    for idx, channel in enumerate(["R", "G", "B"]):
        mean_val = normalized_tensor[idx].mean().item()
        std_val = normalized_tensor[idx].std().item()
        print(f"  Channel {channel}: mean = {mean_val:+.3f}, std = {std_val:.3f}")

    # 4. Batch dimension & GPU transfer
    batch_tensor = tensor_img.unsqueeze(0)
    print(f"Batched shape     : {batch_tensor.shape} on {batch_tensor.device}")

    if torch.cuda.is_available():
        batch_tensor_gpu = batch_tensor.to("cuda")
        print(f"GPU tensor device : {batch_tensor_gpu.device}")
        print(f"VRAM in use       : {torch.cuda.memory_allocated() / (1024**2):.4f} MB")
    print()

    # 5. Data augmentation
    augmentation = T.Compose([
        T.RandomHorizontalFlip(p=0.7),
        T.RandomRotation(degrees=25),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        T.Resize((224, 224), antialias=True)
    ])

    augmented_samples = [augmentation(pil_img) for _ in range(3)]
    save_comparison_plot(pil_img, augmented_samples, OUTPUT_PLOT_PATH)


def save_comparison_plot(original, augmented_list, output_file):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    
    axes[0].imshow(original)
    axes[0].set_title("Original (320x240)")
    axes[0].axis("off")

    for i, img in enumerate(augmented_list):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"Augmented {i + 1} (224x224)")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {output_file}")


if __name__ == "__main__":
    check_hardware()
    img_path = get_or_create_sample_image()
    test_tensor_pipeline(img_path)
