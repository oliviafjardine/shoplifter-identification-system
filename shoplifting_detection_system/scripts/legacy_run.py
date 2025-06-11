#!/usr/bin/env python3
"""
Simple run script for Shoplifting Detection System
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_requirements():
    """Check if system is ready to run"""
    issues = []
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        issues.append("Virtual environment not activated")
    
    # Check if required files exist
    required_files = ['main.py', 'requirements.txt', '.env', 'static/index.html']
    for file in required_files:
        if not Path(file).exists():
            issues.append(f"Missing file: {file}")
    
    # Check if database is running (if using Docker)
    try:
        result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
        if 'postgres' not in result.stdout:
            issues.append("PostgreSQL database not running (try: docker-compose up -d postgres)")
    except FileNotFoundError:
        issues.append("Docker not found - you may need to start PostgreSQL manually")
    
    return issues

def start_database():
    """Start PostgreSQL database using Docker"""
    print("🐳 Starting PostgreSQL database...")
    try:
        result = subprocess.run(['docker-compose', 'up', '-d', 'postgres'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Database started successfully")
            time.sleep(3)  # Wait for database to be ready
            return True
        else:
            print(f"❌ Failed to start database: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker not found. Please start PostgreSQL manually.")
        return False

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    try:
        result = subprocess.run([sys.executable, 'test_system.py'], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def start_application():
    """Start the main application"""
    print("🚀 Starting Shoplifting Detection System...")
    print("=" * 50)
    
    try:
        # Start the FastAPI application
        subprocess.run([sys.executable, 'main.py'])
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Application error: {e}")

def main():
    """Main function"""
    print("🛡️ Shoplifting Detection System Launcher")
    print("=" * 50)
    
    # Check requirements
    issues = check_requirements()
    if issues:
        print("⚠️ Issues found:")
        for issue in issues:
            print(f"   - {issue}")
        
        # Try to fix some issues automatically
        if "PostgreSQL database not running" in str(issues):
            if input("\nWould you like to start the database? (y/n): ").lower() == 'y':
                if not start_database():
                    print("❌ Could not start database. Please start it manually.")
                    return
        
        if "Virtual environment not activated" in str(issues):
            print("\n💡 To activate virtual environment:")
            if os.name == 'nt':  # Windows
                print("   venv\\Scripts\\activate")
            else:  # Unix/Linux/macOS
                print("   source venv/bin/activate")
            return
        
        # Check again after fixes
        remaining_issues = check_requirements()
        if remaining_issues:
            print(f"\n❌ {len(remaining_issues)} issues remain. Please fix them before running.")
            return
    
    print("✅ System checks passed!")
    
    # Ask if user wants to run tests
    if input("\nRun system tests first? (recommended) (y/n): ").lower() == 'y':
        if not run_tests():
            if input("\nTests failed. Continue anyway? (y/n): ").lower() != 'y':
                return
    
    # Start the application
    print("\n🎯 Starting application...")
    print("📱 Dashboard will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 50)
    
    start_application()

if __name__ == "__main__":
    main()
