import argparse
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter YOLO bounding boxes. Delete a bbox if it does not intersect with the lung segmentation mask."
    )
    parser.add_argument(
        "--dataset-dir", "-d",
        type=Path,
        default=Path(r"c:\Gojii\xray lung\forked\CheXNet-Inference\Splitted dataset"),
        help="Path to the YOLO dataset folder (contains images/ and labels/ subfolders)."
    )
    parser.add_argument(
        "--masks-dir", "-m",
        type=Path,
        default=Path(r"c:\Gojii\xray lung\forked\CheXNet-Inference\lung segmentation\output_masks"),
        help="Path to the lung segmentation masks folder (contains train/, valid/, test/ subfolders)."
    )
    parser.add_argument(
        "--vis-dir", "-v",
        type=Path,
        default=Path(r"c:\Gojii\xray lung\forked\CheXNet-Inference\lung segmentation\deleted_bbox_visualizations"),
        help="Path to save visualizations of deleted bounding boxes."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=127,
        help="Binarization threshold for the masks (0-255)."
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not backup the original labels folder before modifying."
    )
    return parser.parse_args()

def save_visualization(image_path: Path, mask_path: Path, kept_boxes: list, deleted_boxes: list, output_path: Path, threshold: int):
    try:
        # Load original image and convert to RGB
        img = Image.open(image_path).convert("RGB")
        img_w, img_h = img.size
        
        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask_resized = mask.resize((img_w, img_h), Image.Resampling.NEAREST)
        
        # Overlay lung mask as translucent green
        mask_arr = np.array(mask_resized)
        alpha = np.where(mask_arr > threshold, 60, 0).astype(np.uint8)
        alpha_img = Image.fromarray(alpha, mode="L")
        
        green_overlay = Image.new("RGB", (img_w, img_h), (0, 255, 0))
        visual_img = Image.composite(green_overlay, img, alpha_img)
        
        draw = ImageDraw.Draw(visual_img)
        
        # Helper to draw boxes
        def draw_box(box_data, color, label):
            class_id, x_center, y_center, w, h = box_data
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int((x_center - w / 2) * img_w)
            x_max = int((x_center + w / 2) * img_w)
            y_min = int((y_center - h / 2) * img_h)
            y_max = int((y_center + h / 2) * img_h)
            
            # Clip to image boundaries
            x_min = max(0, min(x_min, img_w - 1))
            x_max = max(0, min(x_max, img_w - 1))
            y_min = max(0, min(y_min, img_h - 1))
            y_max = max(0, min(y_max, img_h - 1))
            
            # Draw rectangle
            for thickness in range(3):
                draw.rectangle(
                    [x_min - thickness, y_min - thickness, x_max + thickness, y_max + thickness],
                    outline=color
                )
            
            # Draw label text
            draw.text((x_min + 5, y_min + 5), f"{label} (Class {class_id})", fill=color)
            
        # Draw kept boxes in green
        for box in kept_boxes:
            draw_box(box, (0, 255, 0), "KEPT")
            
        # Draw deleted boxes in red
        for box in deleted_boxes:
            draw_box(box, (255, 0, 0), "DELETED")
            
        # Save visualization
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visual_img.save(output_path)
    except Exception as e:
        print(f"  Error creating visualization for {image_path.name}: {e}")

def process_split(label_dir: Path, mask_dir: Path, threshold: int, vis_dir: Path, backup_dir: Path, dataset_dir: Path):
    if not label_dir.exists():
        print(f"Label directory does not exist: {label_dir}")
        return 0, 0, 0, 0
        
    print(f"\nProcessing labels in: {label_dir}")
    print(f"Using masks from: {mask_dir}")
    
    total_files = 0
    total_bboxes_before = 0
    total_bboxes_after = 0
    files_modified = 0
    files_become_empty = 0
    missing_masks = 0
    
    label_paths = list(label_dir.glob("*.txt"))
    if not label_paths:
        print("No label .txt files found.")
        return 0, 0, 0, 0
        
    for label_path in label_paths:
        total_files += 1
        stem = label_path.stem
        
        # Check if corresponding mask image exists
        mask_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
                
        # Determine source label path (read from backup if available to allow idempotency)
        read_path = label_path
        if backup_dir:
            backup_label_path = backup_dir / label_dir.name / label_path.name
            if backup_label_path.exists():
                read_path = backup_label_path
                
        if not mask_path:
            missing_masks += 1
            with open(read_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            total_bboxes_before += len(lines)
            total_bboxes_after += len(lines)
            continue
            
        # Load mask and convert to grayscale numpy array
        try:
            mask_img = Image.open(mask_path).convert("L")
            mask_arr = np.array(mask_img)
            mask_w, mask_h = mask_img.size
        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}")
            with open(read_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            total_bboxes_before += len(lines)
            total_bboxes_after += len(lines)
            continue
            
        # Read current bboxes from the source path
        with open(read_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        bboxes_before = len(lines)
        total_bboxes_before += bboxes_before
        
        kept_lines = []
        kept_boxes = []
        deleted_boxes = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                kept_lines.append(line)
                continue
                
            box_data = (class_id, x_center, y_center, w, h)
            
            # Convert normalized coordinates to pixel coordinates
            x_min = int((x_center - w / 2) * mask_w)
            x_max = int((x_center + w / 2) * mask_w)
            y_min = int((y_center - h / 2) * mask_h)
            y_max = int((y_center + h / 2) * mask_h)
            
            # Clip to mask dimensions
            x_min = max(0, min(x_min, mask_w - 1))
            x_max = max(0, min(x_max, mask_w - 1))
            y_min = max(0, min(y_min, mask_h - 1))
            y_max = max(0, min(y_max, mask_h - 1))
            
            # Extract region of interest from mask
            roi = mask_arr[y_min : y_max + 1, x_min : x_max + 1]
            
            # Check if any pixel in the ROI is above the threshold (i.e. belongs to lung)
            if np.any(roi > threshold):
                kept_lines.append(line)
                kept_boxes.append(box_data)
            else:
                deleted_boxes.append(box_data)
                
        bboxes_after = len(kept_lines)
        total_bboxes_after += bboxes_after
        
        # Write modified bboxes back to label file
        if bboxes_before != bboxes_after:
            files_modified += 1
            if bboxes_after == 0:
                files_become_empty += 1
            
            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
        else:
            # For idempotency, ensure local matches if it was changed
            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(kept_lines)
                
        # Generate visualization if boxes were deleted
        if deleted_boxes and vis_dir:
            image_dir = dataset_dir / "images" / label_dir.name
            image_path = None
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                candidate = image_dir / f"{stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break
            
            if image_path:
                vis_output_path = vis_dir / label_dir.name / f"{stem}_deleted_bbox.jpg"
                save_visualization(image_path, mask_path, kept_boxes, deleted_boxes, vis_output_path, threshold)
                print(f"  [Visualization] Saved to: {vis_output_path}")
                
    print(f"Results for {label_dir.name} split:")
    print(f"  Processed files: {total_files} (missing masks for {missing_masks} files)")
    print(f"  Total bounding boxes before filtering: {total_bboxes_before}")
    print(f"  Total bounding boxes after filtering: {total_bboxes_after}")
    print(f"  Bboxes deleted: {total_bboxes_before - total_bboxes_after}")
    print(f"  Modified label files: {files_modified} (of which {files_become_empty} became empty)")
    
    return total_bboxes_before, total_bboxes_after, files_modified, missing_masks

def main():
    args = parse_args()
    
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory does not exist at: {args.dataset_dir}")
        sys.exit(1)
        
    labels_root = args.dataset_dir / "labels"
    if not labels_root.exists():
        print(f"Error: Labels directory not found in dataset: {labels_root}")
        sys.exit(1)
        
    if not args.masks_dir.exists():
        print(f"Error: Masks directory does not exist at: {args.masks_dir}")
        sys.exit(1)
        
    # Backup labels directory if needed
    backup_dir = args.dataset_dir / "labels_backup"
    if not args.no_backup:
        # Only create a backup if it doesn't already exist.
        if not backup_dir.exists():
            print(f"Creating original labels backup at: {backup_dir}")
            shutil.copytree(labels_root, backup_dir)
        else:
            print(f"Using existing labels backup at: {backup_dir}")
    else:
        backup_dir = None
        
    splits = ["train", "valid", "test"]
    
    grand_before = 0
    grand_after = 0
    grand_modified = 0
    grand_missing = 0
    
    for split in splits:
        label_split_dir = labels_root / split
        mask_split_dir = args.masks_dir / split
        
        # Check if this split exists in labels
        if label_split_dir.is_dir():
            before, after, modified, missing = process_split(
                label_split_dir, mask_split_dir, args.threshold, args.vis_dir, backup_dir, args.dataset_dir
            )
            grand_before += before
            grand_after += after
            grand_modified += modified
            grand_missing += missing
            
    print("\n" + "="*40)
    print("GRAND TOTALS:")
    print(f"  Total bboxes before: {grand_before}")
    print(f"  Total bboxes after: {grand_after}")
    print(f"  Total bboxes deleted: {grand_before - grand_after}")
    print(f"  Total label files modified: {grand_modified}")
    if grand_missing > 0:
        print(f"  WARNING: Missing masks for {grand_missing} label files (these files were not filtered).")
    print("="*40)
    print("Done!")

if __name__ == "__main__":
    main()
