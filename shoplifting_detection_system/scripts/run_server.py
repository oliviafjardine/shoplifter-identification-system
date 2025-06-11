#!/usr/bin/env python3
"""
Professional Shoplifting Detection System Server
Runs the reorganized professional structure
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if system is ready to run"""
    issues = []

    # Check if required files exist in new structure
    required_files = [
        'src/shoplifting_detection/api/main.py',
        'config/requirements.txt',
        'assets/web/templates/index.html'
    ]
    for file in required_files:
        if not Path(file).exists():
            issues.append(f"Missing file: {file}")

    # Check if core modules exist
    core_modules = [
        'src/shoplifting_detection/core/detector.py',
        'src/shoplifting_detection/services/camera_service.py',
        'ml/training/data_generators.py'
    ]
    for module in core_modules:
        if not Path(module).exists():
            issues.append(f"Missing core module: {module}")

    return issues

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    try:
        # Run tests from the tests directory
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'],
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_application():
    """Start the main application"""
    print("🚀 Starting Shoplifting Detection System...")
    print("=" * 50)

    # Add src to Python path
    src_path = Path("src").absolute()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        # Import and run the FastAPI application
        from shoplifting_detection.api.main import app
        import uvicorn

        print("🌐 Starting server on http://localhost:8000")
        print("📊 Dashboard available at http://localhost:8000")
        print("🔍 API docs at http://localhost:8000/docs")
        print("🛑 Press Ctrl+C to stop")

        uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Trying alternative startup method...")

        # Fallback: run the main.py directly
        try:
            subprocess.run([sys.executable, 'src/shoplifting_detection/api/main.py'])
        except Exception as e2:
            print(f"❌ Alternative startup failed: {e2}")

    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Application error: {e}")

def main():
    """Main function"""
    print("🛡️ Professional Shoplifting Detection System")
    print("=" * 50)

    # Check requirements
    issues = check_requirements()
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"   - {issue}")

        print(f"\n❌ {len(issues)} issues found. Please fix them before running.")
        print("\n💡 Quick fixes:")
        print("   1. Make sure you're in the shoplifting_detection_system directory")
        print("   2. Install dependencies: pip install fastapi uvicorn pytest")
        print("   3. Check that all files are in the correct locations")
        return

    print("✅ System checks passed!")

    # Ask if user wants to run tests
    try:
        run_tests_input = input("\nRun system tests first? (recommended) (y/n): ").lower()
        if run_tests_input == 'y':
            if not run_tests():
                continue_input = input("\nTests failed. Continue anyway? (y/n): ").lower()
                if continue_input != 'y':
                    return
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return

    # Start the application
    print("\n🎯 Starting application...")
    print("📱 Dashboard will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)

    start_application()

if __name__ == "__main__":
    main()
