"""
Setup script for Custom Hiera-B SAM2 Project
Installs dependencies and verifies installation
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a shell command"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{description}")
    logger.info(f"{'='*60}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    """Main setup function"""
    logger.info("="*60)
    logger.info("CUSTOM HIERA-B SAM2 PROJECT SETUP")
    logger.info("="*60)
    
    # Step 1: Install SAM2 from local directory
    logger.info("\nStep 1: Installing SAM2 from local directory...")
    sam2_dir = Path(__file__).parent / "segment-anything-2"
    if sam2_dir.exists():
        if not run_command(
            [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
            "Installing SAM2"
        ):
            logger.error("Failed to install SAM2")
            return False
    else:
        logger.error(f"SAM2 directory not found: {sam2_dir}")
        return False
    
    # Step 2: Install other dependencies
    logger.info("\nStep 2: Installing other dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        if not run_command(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            "Installing requirements"
        ):
            logger.warning("Some packages may have failed to install")
    else:
        logger.warning(f"Requirements file not found: {requirements_file}")
    
    # Step 3: Check dependencies
    logger.info("\nStep 3: Checking dependencies...")
    from utils.dependency_checker import check_all_dependencies, print_dependency_report
    
    results = check_all_dependencies(install_missing=False)
    print_dependency_report(results)
    
    # Step 4: Run tests
    logger.info("\nStep 4: Running tests...")
    test_script = Path(__file__).parent / "test_custom_hiera_b.py"
    if test_script.exists():
        run_command(
            [sys.executable, str(test_script)],
            "Running test suite"
        )
    else:
        logger.warning(f"Test script not found: {test_script}")
    
    logger.info("\n" + "="*60)
    logger.info("SETUP COMPLETE!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Verify installation: python test_custom_hiera_b.py")
    logger.info("2. Run the app: streamlit run app/main.py")
    logger.info("3. Check dependencies: python utils/dependency_checker.py")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

