"""
Test script to verify all three models (Hiera-B, L, S) can be loaded
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test loading all three models"""
    from backend.sam2_model import SAM2Model
    
    models_to_test = [
        "SAM 2 Hiera-B",
        "SAM 2 Hiera-S", 
        "SAM 2 Hiera-L"
    ]
    
    results = {}
    
    for model_type in models_to_test:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_type}")
        logger.info(f"{'='*60}")
        
        try:
            model = SAM2Model(
                model_type=model_type,
                device="CPU",
                use_custom_preprocessing=False  # Disable for faster loading
            )
            results[model_type] = "SUCCESS"
            logger.info(f"✓ {model_type} loaded successfully!")
        except Exception as e:
            results[model_type] = f"FAILED: {str(e)[:200]}"
            logger.error(f"✗ {model_type} failed: {str(e)[:200]}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for model_type, result in results.items():
        status = "✓" if result == "SUCCESS" else "✗"
        logger.info(f"{status} {model_type}: {result}")
    
    success_count = sum(1 for r in results.values() if r == "SUCCESS")
    total_count = len(results)
    
    logger.info(f"\nResults: {success_count}/{total_count} models loaded successfully")
    
    return success_count == total_count

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)

