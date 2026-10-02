"""
Dependency Checker and Installer
Ensures all required packages are installed
"""

import sys
import subprocess
import importlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_PACKAGES = {
    "torch": "torch>=2.2.0",
    "torchvision": "torchvision>=0.17.0",
    "torchaudio": "torchaudio>=2.2.0",
    "numpy": "numpy>=1.24.0",
    "cv2": "opencv-python>=4.8.0",
    "PIL": "pillow>=10.0.0",
    "streamlit": "streamlit>=1.29.0",
    "sam2": None,  # Installed from local directory
    "hydra": "hydra-core>=1.3.2",
    "iopath": "iopath>=0.1.10",
    "huggingface_hub": "huggingface-hub>=0.19.0",
}


def check_package(package_name: str, import_name: str = None) -> bool:
    """
    Check if a package is installed
    
    Args:
        package_name: Package name for pip
        import_name: Import name (if different from package_name)
    
    Returns:
        True if package is available
    """
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def check_sam2_installation() -> bool:
    """Check if SAM2 is properly installed"""
    try:
        import sam2
        from sam2.build_sam import build_sam2_video_predictor
        return True
    except ImportError:
        return False


def install_package(package_spec: str) -> bool:
    """
    Install a package using pip
    
    Args:
        package_spec: Package specification (e.g., "torch>=2.2.0")
    
    Returns:
        True if installation successful
    """
    try:
        logger.info(f"Installing {package_spec}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package_spec],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Successfully installed {package_spec}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_spec}: {e}")
        return False


def install_sam2() -> bool:
    """Install SAM2 from local directory"""
    sam2_dir = Path(__file__).parent.parent / "segment-anything-2"
    if not sam2_dir.exists():
        logger.error(f"SAM2 directory not found: {sam2_dir}")
        return False
    
    try:
        logger.info("Installing SAM2 from local directory...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", str(sam2_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Successfully installed SAM2")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install SAM2: {e}")
        return False


def check_all_dependencies(install_missing: bool = False) -> dict:
    """
    Check all required dependencies
    
    Args:
        install_missing: Whether to automatically install missing packages
    
    Returns:
        Dictionary with check results
    """
    results = {
        "all_installed": True,
        "missing": [],
        "installed": [],
        "errors": [],
    }
    
    # Check SAM2 separately
    if not check_sam2_installation():
        results["missing"].append("sam2")
        results["all_installed"] = False
        if install_missing:
            if install_sam2():
                results["installed"].append("sam2")
            else:
                results["errors"].append("sam2")
    
    # Check other packages
    for package_name, package_spec in REQUIRED_PACKAGES.items():
        if package_name == "sam2":
            continue  # Already checked
        
        import_name = package_name
        if package_name == "cv2":
            import_name = "cv2"
        elif package_name == "PIL":
            import_name = "PIL"
        elif package_name == "hydra":
            import_name = "hydra"
        
        if check_package(package_name, import_name):
            results["installed"].append(package_name)
        else:
            results["missing"].append(package_name)
            results["all_installed"] = False
            
            if install_missing and package_spec:
                if install_package(package_spec):
                    results["installed"].append(package_name)
                    results["missing"].remove(package_name)
                else:
                    results["errors"].append(package_name)
    
    return results


def print_dependency_report(results: dict):
    """Print a formatted dependency report"""
    print("\n" + "="*60)
    print("DEPENDENCY CHECK REPORT")
    print("="*60)
    
    if results["installed"]:
        print(f"\n✅ Installed ({len(results['installed'])}):")
        for pkg in sorted(results["installed"]):
            print(f"   - {pkg}")
    
    if results["missing"]:
        print(f"\n❌ Missing ({len(results['missing'])}):")
        for pkg in sorted(results["missing"]):
            print(f"   - {pkg}")
    
    if results["errors"]:
        print(f"\n⚠️  Installation Errors ({len(results['errors'])}):")
        for pkg in sorted(results["errors"]):
            print(f"   - {pkg}")
    
    print("\n" + "="*60)
    
    if results["all_installed"]:
        print("✅ All dependencies are installed!")
    else:
        print("❌ Some dependencies are missing.")
        print("\nTo install missing packages, run:")
        print("   python utils/dependency_checker.py --install")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check and install dependencies")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Automatically install missing packages"
    )
    args = parser.parse_args()
    
    results = check_all_dependencies(install_missing=args.install)
    print_dependency_report(results)
    
    sys.exit(0 if results["all_installed"] else 1)

