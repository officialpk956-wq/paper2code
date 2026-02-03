"""
Run all visual comparison tests in sequence.
"""

import sys

print("="*70)
print("RUNNING ALL VISUAL COMPARISON TESTS")
print("="*70)

tests = [
    ("test_resnet_vs_vit.py", "ResNet vs ViT comparison"),
    ("test_visual_comparison.py", "Visual highlighting determinism"),
    ("test_single_arch_mode.py", "Backward compatibility"),
    ("test_visual_features_complete.py", "Comprehensive feature validation")
]

failed = []

for test_file, description in tests:
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print('='*70)
    
    try:
        with open(test_file) as f:
            code = f.read()
        exec(code)
        print(f"\n✅ {test_file} PASSED")
    except Exception as e:
        print(f"\n❌ {test_file} FAILED: {e}")
        failed.append((test_file, str(e)))

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

if not failed:
    print(f"✅ ALL {len(tests)} TESTS PASSED")
    print("\n🎉 Visual comparison features are production-ready!")
else:
    print(f"❌ {len(failed)} of {len(tests)} tests failed:")
    for test_file, error in failed:
        print(f"  - {test_file}: {error}")
    sys.exit(1)
