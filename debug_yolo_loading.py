"""
Debug script to check YOLO loading issues
Run this to see what's happening with your YOLO model file
"""
import os
from pathlib import Path

# Find the _MODELLING directory
SCRIPT_DIR = Path(__file__).parent.absolute()
MODELLING_ROOT = str(SCRIPT_DIR / "_MODELLING")

print("="*60)
print("YOLO Loading Debug Information")
print("="*60)
print(f"\nScript directory: {SCRIPT_DIR}")
print(f"MODELLING_ROOT: {MODELLING_ROOT}")
print(f"MODELLING_ROOT exists: {os.path.exists(MODELLING_ROOT)}")

# List all files in _MODELLING
print(f"\n{'='*60}")
print("Files in _MODELLING directory:")
print("="*60)
if os.path.exists(MODELLING_ROOT):
    all_files = list(Path(MODELLING_ROOT).glob('*'))
    if all_files:
        for f in sorted(all_files):
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            print(f"  {'[FILE]' if f.is_file() else '[DIR] '} {f.name:40s} {size_mb:>8.2f} MB")
    else:
        print("  (empty directory)")
else:
    print("  Directory does not exist!")

# Check for YOLO files specifically
print(f"\n{'='*60}")
print("YOLO-related files:")
print("="*60)
yolo_candidates = [
    'yolov12x-cls.pt',
    'yolov12-cls.pt',
    'yolo12x-cls.pt',
    'yolo.pt',
    'best.pt',
]

for candidate in yolo_candidates:
    full_path = os.path.join(MODELLING_ROOT, candidate)
    exists = os.path.exists(full_path)
    print(f"  {candidate:30s} {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    if exists:
        size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"    → Size: {size_mb:.2f} MB")
        print(f"    → Full path: {full_path}")

# Check for any .pt files
print(f"\n{'='*60}")
print("All .pt files in _MODELLING:")
print("="*60)
if os.path.exists(MODELLING_ROOT):
    pt_files = list(Path(MODELLING_ROOT).glob('*.pt'))
    if pt_files:
        for pt in sorted(pt_files):
            size_mb = pt.stat().st_size / (1024 * 1024)
            print(f"  {pt.name:40s} {size_mb:>8.2f} MB")
    else:
        print("  No .pt files found")

# Try to import ultralytics
print(f"\n{'='*60}")
print("Checking ultralytics installation:")
print("="*60)
try:
    from ultralytics import YOLO
    print("  ✅ ultralytics is installed")
    print(f"  Version: {YOLO.__module__}")
    
    # Try to load if file exists
    yolo_path = os.path.join(MODELLING_ROOT, 'yolov12x-cls.pt')
    if os.path.exists(yolo_path):
        print(f"\n  Attempting to load: {yolo_path}")
        try:
            model = YOLO(yolo_path)
            print("  ✅ YOLO model loaded successfully!")
            print(f"  Model type: {type(model)}")
        except Exception as e:
            print(f"  ❌ Failed to load YOLO model")
            print(f"  Error: {e}")
            import traceback
            print("\n  Full traceback:")
            traceback.print_exc()
    else:
        print(f"\n  File not found: {yolo_path}")
        
except ImportError as e:
    print(f"  ❌ ultralytics not installed: {e}")
    print("\n  Install with: pip install ultralytics")

print(f"\n{'='*60}")
print("Recommendations:")
print("="*60)
if not os.path.exists(MODELLING_ROOT):
    print("  1. Create the _MODELLING directory")
    print("  2. Place your YOLO model file there")
elif not list(Path(MODELLING_ROOT).glob('*.pt')):
    print("  1. No .pt files found in _MODELLING")
    print("  2. Copy your YOLO weights file to _MODELLING/")
    print("  3. Rename it to 'yolov12x-cls.pt' or update the code")
else:
    print("  1. Verify the YOLO file is not corrupted")
    print("  2. Check the file size (should be ~50-200 MB)")
    print("  3. Try loading it manually with: YOLO('path/to/file.pt')")

print("="*60)