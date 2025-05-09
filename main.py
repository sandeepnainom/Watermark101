import os
import numpy as np
import pickle
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# CIFAR-10 label names
cifar10_labels = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

def load_cifar10_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def convert_cifar10_data_to_image(data):
    return data.reshape(3, 32, 32).transpose(1, 2, 0)

def add_text_watermark(image, text="© YourCompany", font_size=24):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    width, height = pil_image.size
    x = width - text_width - 20
    y = height - text_height - 20

    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([x-10, y-10, x+text_width+10, y+text_height+10], fill=(0, 0, 0, 100))
    overlay_draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    combined = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
    return np.array(combined.convert('RGB'))

def show_comparison(original, watermarked, label, idx, comparison_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Image Comparison - {label}", fontsize=16, fontweight='bold')

    ax[0].imshow(original)
    ax[0].set_title("Original", fontsize=14)
    ax[0].axis('off')

    ax[1].imshow(watermarked)
    ax[1].set_title("Watermarked", fontsize=14)
    ax[1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


    if comparison_path:
        fig.savefig(comparison_path, dpi=300)
        print(f"💾 Saved comparison image to: {comparison_path}")

def calculate_quality_metrics(original, watermarked):
    psnr_value = psnr(original, watermarked, data_range=255)
    ssim_value = ssim(original, watermarked, channel_axis=2, data_range=255)
    return psnr_value, ssim_value

# ==== MAIN ====
original_images_path = 'data/original_images/cifar-10-batches-py'
output_path = 'output'
os.makedirs(output_path, exist_ok=True)

original_files = [os.path.join(original_images_path, f) for f in os.listdir(original_images_path) if f.startswith('data_batch')]
num_images = 5

all_images = []
all_labels = []

for file in original_files:
    batch = load_cifar10_batch(file)
    all_images.extend(batch[b'data'])
    all_labels.extend(batch[b'labels'])

total = len(all_images)
used_indices = set()

for i in range(num_images):
    while True:
        idx = random.randint(0, total - 1)
        if idx not in used_indices:
            used_indices.add(idx)
            break

    image_data = convert_cifar10_data_to_image(np.array(all_images[idx]))
    label_text = cifar10_labels[all_labels[idx]]

    # Resize to 256x256 with high quality
    pil_img = Image.fromarray(image_data).resize((256, 256), resample=Image.LANCZOS)
    resized_image = np.array(pil_img)

    # Add watermark
    watermarked_image = add_text_watermark(resized_image, text="© YourCompany", font_size=24)

    # Show comparison and save image
    comparison_path = os.path.join(output_path, f"comparison_{idx}_{label_text.lower()}.png")
    show_comparison(resized_image, watermarked_image, label_text, i, comparison_path)

    # Calculate PSNR and SSIM
    psnr_val, ssim_val = calculate_quality_metrics(resized_image, watermarked_image)
    print(f"✅ Saved image {i+1}: {label_text}")
    print(f"🔍 PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")
