#!/usr/bin/env python3
"""
Setup script for spacy model installation
"""

import subprocess
import sys

def install_spacy_model():
    """Install the required spacy model"""
    try:
        print("Installing spacy model: en_core_web_sm")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", "en_core_web_sm"
        ])
        print("✅ spacy model installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing spacy model: {e}")
        print("Please run manually: python -m spacy download en_core_web_sm")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    install_spacy_model() 