#!/usr/bin/env python3
"""
Quick Start Script for Shoplifting Detection System
Run this file to start the main application
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start the shoplifting detection system"""
    print("🛡️ Shoplifting Detection System - Quick Start")
    print("=" * 50)
    
    # Change to the system directory
    system_dir = Path("shoplifting_detection_system")
    if not system_dir.exists():
        print("❌ System directory not found!")
        print("   Make sure you're in the correct directory")
        return
    
    # Check if main.py exists
    main_file = system_dir / "main.py"
    if not main_file.exists():
        print("❌ Main application file not found!")
        return
    
    print("🚀 Starting the main application...")
    print("🌐 Dashboard will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Run the main application
        subprocess.run([sys.executable, str(main_file)], cwd=str(system_dir))
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
