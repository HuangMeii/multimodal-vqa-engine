"""Quick test for Florence-2 inside container"""
import sys
sys.path.insert(0, "/app")

from services.t1_vision.florence2_captioner import Florence2Captioner
from PIL import Image
import io

print("=== Initializing Florence2Captioner ===")
c = Florence2Captioner()
print("=== Done ===")

# Test 1: CAPTION
print("\n=== CAPTION ===")
img = Image.new('RGB', (224, 224), color='red')
cap = c.generate_caption(img)
print(f"Result: {repr(cap)}")

# Test 2: DETAILED CAPTION
print("\n=== DETAILED CAPTION ===")
dc = c.generate_detailed_caption(img)
print(f"Result: {repr(dc)}")

# Test 3: OD
print("\n=== OD ===")
od = c.generate_od(img)
print(f"Result: {repr(od)}")

print("\n=== ALL TESTS DONE ===")
