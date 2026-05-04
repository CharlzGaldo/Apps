import os
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tiff', '.tif'}
LABEL_EXTS = {'.txt'}
SKIP_EXTS = {'.xml', '.json', '.csv', '.yaml', '.yml'}

def organize_folder(split_dir):
    split_path = Path(split_dir)
    if not split_path.exists():
        print(f"⚠️  Skipping {split_dir} (not found)")
        return
    
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    moved_img, moved_lbl, skipped = 0, 0, 0
    
    for file in split_path.iterdir():
        if not file.is_file():
            continue
            
        ext = file.suffix.lower()
        
        if ext in IMAGE_EXTS:
            shutil.move(str(file), str(images_dir / file.name))
            moved_img += 1
        elif ext in LABEL_EXTS:
            shutil.move(str(file), str(labels_dir / file.name))
            moved_lbl += 1
        elif ext in SKIP_EXTS:
            skipped += 1
        else:
            print(f"❓ Unknown file type: {file.name}")
    
    print(f"✅ {split_path.name}: {moved_img} images, {moved_lbl} labels, {skipped} skipped")

for split in ['train', 'val', 'test']:
    organize_folder(f"dataset/{split}")

print("\n🎉 Done!")