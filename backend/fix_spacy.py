#!/usr/bin/env python3
"""
Comprehensive spaCy fix script for version compatibility issues
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def fix_spacy():
    """Fix spaCy version compatibility issues"""
    print("🔧 Fixing spaCy version compatibility issues...")
    
    # Step 1: Uninstall current spaCy
    print("\n📦 Step 1: Uninstalling current spaCy...")
    run_command(f"{sys.executable} -m pip uninstall spacy -y", "Uninstalling spaCy")
    
    # Step 2: Install compatible spaCy version
    print("\n📦 Step 2: Installing compatible spaCy version...")
    if not run_command(f"{sys.executable} -m pip install spacy==3.8.0", "Installing spaCy 3.8.0"):
        return False
    
    # Step 3: Remove old model if exists
    print("\n🗑️ Step 3: Removing old spaCy model...")
    run_command(f"{sys.executable} -m spacy uninstall en_core_web_sm", "Removing old model")
    
    # Step 4: Download compatible model
    print("\n📥 Step 4: Downloading compatible spaCy model...")
    if not run_command(f"{sys.executable} -m spacy download en_core_web_sm", "Downloading model"):
        return False
    
    # Step 5: Validate installation
    print("\n✅ Step 5: Validating installation...")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy model loaded successfully!")
        print("✅ All spaCy issues resolved!")
        return True
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 spaCy Fix Script")
    print("=" * 50)
    
    if fix_spacy():
        print("\n🎉 spaCy has been successfully fixed!")
        print("You can now run: python app.py")
    else:
        print("\n❌ spaCy fix failed. Please try manual steps:")
        print("1. pip uninstall spacy")
        print("2. pip install spacy==3.8.0")
        print("3. python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main() 