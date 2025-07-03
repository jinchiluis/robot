import time
import torch
from patchcore_exth import SimpleImageDataset
from PIL import Image
from torchvision import transforms
import torchvision.models as models

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
model = model.cuda()  # Move model to GPU!
model.eval()

# Test each component separately
dataset = SimpleImageDataset("temp_patchcore_training", transform=transform)

# 1. Raw image loading
start = time.time()
for img_path in dataset.images[:16]:
    img = Image.open(img_path).convert('RGB')
print(f"Raw loading: {(time.time() - start)*1000:.1f}ms total, {(time.time() - start)*1000/16:.1f}ms per image")

# 2. Background masking (if used)
if dataset.masker and dataset.mask_method:
    img = Image.open(dataset.images[0]).convert('RGB')
    start = time.time()
    for _ in range(16):
        if dataset.mask_method == 'center_crop':
            masked = dataset.masker.center_crop_percent(img, **dataset.mask_params)
        elif dataset.mask_method == 'edge_crop':
            masked = dataset.masker.edge_based_crop(img, **dataset.mask_params)
    print(f"Masking: {(time.time() - start)*1000:.1f}ms total, {(time.time() - start)*1000/16:.1f}ms per image")

# 3. Transforms
img = Image.open(dataset.images[0]).convert('RGB')
if dataset.masker and dataset.mask_method:
    img = dataset.masker.center_crop_percent(img, **dataset.mask_params)  # or edge_crop
start = time.time()
for _ in range(16):
    transformed = transform(img)
print(f"Transforms: {(time.time() - start)*1000:.1f}ms total, {(time.time() - start)*1000/16:.1f}ms per image")

# 4. Model forward pass
model.eval()
with torch.no_grad():
    dummy_batch = torch.randn(16, 3, 512, 512).cuda()  # Assuming your transform outputs 512x512
    torch.cuda.synchronize()
    start = time.time()
    _ = model(dummy_batch)
    torch.cuda.synchronize()
print(f"Model forward: {(time.time() - start)*1000:.1f}ms total, {(time.time() - start)*1000/16:.1f}ms per image")

# 5. Check if you're using batch processing
print(f"\nAre you processing images in batches or one at a time?")
print(f"If using DataLoader, what's your batch_size?")