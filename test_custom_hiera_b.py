"""
Test script for Custom Hiera-B Model
Tests all components and verifies functionality
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports"""
    logger.info("Testing imports...")
    errors = []
    
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
    
    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
    
    try:
        import sam2
        logger.info("✓ SAM2")
    except ImportError as e:
        errors.append(f"SAM2: {e}")
    
    try:
        from models.custom_hiera_b import CustomHieraB, build_custom_hiera_b
        logger.info("✓ Custom Hiera-B model")
    except ImportError as e:
        errors.append(f"Custom Hiera-B: {e}")
    
    try:
        from backend.custom_hiera_encoder import DomainSpecificPreprocessor
        logger.info("✓ Custom preprocessing")
    except ImportError as e:
        errors.append(f"Custom preprocessing: {e}")
    
    try:
        from backend.sam2_model import SAM2Model
        logger.info("✓ SAM2Model wrapper")
    except ImportError as e:
        errors.append(f"SAM2Model: {e}")
    
    if errors:
        logger.error("Import errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✓ All imports successful")
    return True


def test_custom_hiera_b():
    """Test Custom Hiera-B model"""
    logger.info("Testing Custom Hiera-B model...")
    
    try:
        import torch
        from models.custom_hiera_b import build_custom_hiera_b
        
        # Create model
        encoder = build_custom_hiera_b(d_model=256)
        logger.info("✓ Custom Hiera-B encoder created")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output = encoder(dummy_input)
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"  Output keys: {output.keys()}")
        logger.info(f"  Vision features shape: {output['vision_features'].shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Custom Hiera-B test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_preprocessing():
    """Test preprocessing pipeline"""
    logger.info("Testing preprocessing...")
    
    try:
        import numpy as np
        from backend.custom_hiera_encoder import DomainSpecificPreprocessor
        
        preprocessor = DomainSpecificPreprocessor(
            target_size=1024,
            fps=10,
            normalize=True,
            interpolate_corrupted=True
        )
        
        # Test frame preprocessing
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor, metadata = preprocessor.preprocess_frame(dummy_frame)
        
        logger.info(f"✓ Preprocessing successful")
        logger.info(f"  Output shape: {tensor.shape}")
        logger.info(f"  Metadata: {metadata.keys()}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Preprocessing test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_augmentation():
    """Test augmentation"""
    logger.info("Testing augmentation...")
    
    try:
        import numpy as np
        from backend.custom_hiera_encoder import DomainSpecificAugmentation
        
        augmenter = DomainSpecificAugmentation()
        
        dummy_frame = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        dummy_mask = np.random.randint(0, 1, (1024, 1024), dtype=np.uint8)
        
        aug_frame, aug_mask = augmenter.augment(dummy_frame, dummy_mask)
        
        logger.info(f"✓ Augmentation successful")
        logger.info(f"  Augmented frame shape: {aug_frame.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Augmentation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_dependencies():
    """Test dependency checker"""
    logger.info("Testing dependency checker...")
    
    try:
        from utils.dependency_checker import check_all_dependencies
        
        results = check_all_dependencies(install_missing=False)
        
        if results["all_installed"]:
            logger.info("✓ All dependencies installed")
        else:
            logger.warning(f"⚠ Missing dependencies: {results['missing']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Dependency check failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("CUSTOM HIERA-B MODEL TEST SUITE")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Dependencies", test_dependencies),
        ("Preprocessing", test_preprocessing),
        ("Augmentation", test_augmentation),
        ("Custom Hiera-B Model", test_custom_hiera_b),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

