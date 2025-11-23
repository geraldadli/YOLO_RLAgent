"""
Manual YOLO loading test
This will help us see the exact error and fix it
"""
import os
from pathlib import Path

# Setup path
SCRIPT_DIR = Path(__file__).parent.absolute()
MODELLING_ROOT = SCRIPT_DIR / "_MODELLING"
YOLO_PATH = MODELLING_ROOT / "yolov12x-cls.pt"

print("="*60)
print("Manual YOLO Loading Test")
print("="*60)
print(f"Looking for YOLO at: {YOLO_PATH}")
print(f"File exists: {YOLO_PATH.exists()}")

if YOLO_PATH.exists():
    size_mb = YOLO_PATH.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")
else:
    print("ERROR: YOLO file not found!")
    exit(1)

print("\n" + "="*60)
print("Step 1: Import custom modules")
print("="*60)

try:
    import yolo_custom_modules
    print("✅ Custom modules imported successfully")
except ImportError as e:
    print(f"⚠️ Could not import yolo_custom_modules: {e}")
    print("   Continuing without custom modules...")

print("\n" + "="*60)
print("Step 2: Import ultralytics")
print("="*60)

try:
    from ultralytics import YOLO
    import ultralytics
    print(f"✅ ultralytics imported successfully")
    print(f"   Version: {ultralytics.__version__}")
except ImportError as e:
    print(f"❌ Could not import ultralytics: {e}")
    print("\nInstall with: pip install ultralytics")
    exit(1)

print("\n" + "="*60)
print("Step 3: Check for A2C2f module")
print("="*60)

try:
    import ultralytics.nn.modules.block as block_module
    has_a2c2f = hasattr(block_module, 'A2C2f')
    print(f"A2C2f available: {has_a2c2f}")
    
    if has_a2c2f:
        print(f"A2C2f type: {type(block_module.A2C2f)}")
    else:
        print("⚠️ A2C2f not found in ultralytics.nn.modules.block")
        print("   Available modules:")
        for name in dir(block_module):
            if not name.startswith('_') and name[0].isupper():
                print(f"     - {name}")
except Exception as e:
    print(f"Error checking modules: {e}")

print("\n" + "="*60)
print("Step 4: Load YOLO model")
print("="*60)

try:
    print(f"Loading: {YOLO_PATH}")
    model = YOLO(str(YOLO_PATH))
    print("✅ YOLO model loaded successfully!")
    print(f"   Model type: {type(model)}")
    print(f"   Model names: {model.names if hasattr(model, 'names') else 'N/A'}")
    
    # Try a dummy prediction
    print("\nStep 5: Test prediction on dummy image")
    import numpy as np
    from PIL import Image
    
    dummy_img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
    print("Created dummy 640x640 image")
    
    results = model.predict(dummy_img, verbose=False)
    print(f"✅ Prediction successful!")
    print(f"   Results type: {type(results)}")
    print(f"   Number of results: {len(results)}")
    
except Exception as e:
    print(f"❌ Failed to load YOLO model")
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {e}")
    
    print("\n" + "="*60)
    print("Full Traceback:")
    print("="*60)
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("Suggested fixes:")
    print("="*60)
    
    if "A2C2f" in str(e):
        print("1. Make sure yolo_custom_modules.py is in the same directory")
        print("2. Run: pip install --upgrade ultralytics>=8.1.0")
        print("3. Try downloading a standard YOLOv8 model instead:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('yolov8n-cls.pt')  # Auto-downloads")
    elif "module" in str(e).lower():
        print("1. Install missing dependencies:")
        print("   pip install torch torchvision ultralytics")
    else:
        print("1. Verify the .pt file is not corrupted")
        print("2. Try re-downloading the model")
        print("3. Check if the model is compatible with your ultralytics version")

print("\n" + "="*60)