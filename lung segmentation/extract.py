import argparse
import os
import sys
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms.functional as TF

# 1. SETUP COMMAND LINE ARGUMENTS
parser = argparse.ArgumentParser(description="Run inference using PretrainedUNet for lung segmentation.")
parser.add_argument(
    "--input", "-i",
    type=Path,
    default=Path(r"c:\Gojii\xray lung\lung-segmentation\input\dataset\images"),
    help="Path to the input images directory containing PNG/JPG files."
)
parser.add_argument(
    "--output", "-o",
    type=Path,
    default=Path(r"c:\Gojii\xray lung\forked\CheXNet-Inference\lung segmentation\output_masks"),
    help="Path to the output directory where predicted masks will be saved."
)
parser.add_argument(
    "--checkpoint", "-c",
    type=Path,
    default=Path(r"c:\Gojii\xray lung\lung-segmentation\models\unet-6v.pt"),
    help="Path to the model checkpoint weight file (.pt)."
)
parser.add_argument(
    "--project-path", "-p",
    type=Path,
    default=Path(r"c:\Gojii\xray lung\lung-segmentation"),
    help="Path to the original project directory containing 'src' modules."
)

args = parser.parse_args()

# 2. Add the original project path to sys.path so we can import 'src' modules from anywhere
original_project_path = str(args.project_path)
if original_project_path not in sys.path:
    sys.path.append(original_project_path)

# pyrefly: ignore [missing-import]
from src.models import PretrainedUNet
# pyrefly: ignore [missing-import]
from src.data import blend

# 3. INITIALIZE DEVICE & LOAD MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running inference on device: {device}")

# Validate checkpoint path
if not args.checkpoint.exists():
    print(f"Error: Model checkpoint file not found at: {args.checkpoint}")
    sys.exit(1)

print(f"Loading model weights from: {args.checkpoint}")
model = PretrainedUNet(
    in_channels=1,
    out_channels=2,
    batch_norm=True,
    upscale_mode="bilinear"
)
model.load_state_dict(torch.load(args.checkpoint, map_location=device))
model.to(device)
model.eval()

# 4. RUN INFERENCE ON THE TARGET IMAGES AND SPLITS
# Validate input directory
if not args.input.exists():
    print(f"Error: Input directory does not exist: {args.input}")
    sys.exit(1)

# Auto-detect data splits (train, valid, test)
splits = ["train", "valid", "test"]
detected_splits = [s for s in splits if (args.input / s).is_dir()]

# Also check if there is an 'images' folder inside (e.g., Splitted dataset/images/test)
if not detected_splits and (args.input / "images").is_dir():
    images_dir = args.input / "images"
    detected_splits = [s for s in splits if (images_dir / s).is_dir()]
    if detected_splits:
        args.input = images_dir

if detected_splits:
    print(f"Detected data splits in input directory: {', '.join(detected_splits)}")
    job_list = [(args.input / split, args.output / split, split) for split in detected_splits]
else:
    job_list = [(args.input, args.output, None)]

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

for input_dir, output_dir, split_name in job_list:
    split_info = f"[{split_name.upper()} split] " if split_name else ""
    print(f"\nProcessing {split_info}images from: {input_dir}")
    
    # Find all matching files (case-insensitive extensions)
    image_paths = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]
    
    if not image_paths:
        print(f"No matching images found in: {input_dir}")
        print(f"Supported formats: {', '.join(sorted(image_extensions))}")
        continue
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(image_paths)} images. Starting inference...")
    
    for idx, img_path in enumerate(image_paths):
        # Load image and convert to Palette mode (grayscale input)
        origin_img = Image.open(img_path).convert("P")
        
        # Preprocess (Resize to 512x512, convert to tensor, and normalize by subtracting 0.5)
        resized_img = TF.resize(origin_img, (512, 512))
        img_tensor = TF.to_tensor(resized_img) - 0.5  # Shape: [1, 512, 512]
        
        # Add batch dimension and send to device
        img_batch = img_tensor.unsqueeze(0).to(device)  # Shape: [1, 1, 512, 512]
        
        # Predict mask
        with torch.no_grad():
            output = model(img_batch)
            softmax = torch.nn.functional.log_softmax(output, dim=1)
            pred_mask = torch.argmax(softmax, dim=1)  # Shape: [1, 512, 512]
            
            # Remove batch dimension and move to CPU
            pred_mask_cpu = pred_mask[0].to("cpu").float()
            
        # Save predicted mask as image
        pred_pil = TF.to_pil_image(pred_mask_cpu)
        pred_pil.save(output_dir / img_path.name)
        
        # Optional: Save blended visualization (CXR image overlaid with red mask)
        # blended_img = blend(img_tensor, mask2=pred_mask_cpu)
        # blended_img.save(output_dir / f"blend_{img_path.name}")
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_paths):
            print(f"{split_info}Processed {idx + 1}/{len(image_paths)} images...")
            
    print(f"{split_info}Inference completed! Predicted masks saved to: {output_dir}")
