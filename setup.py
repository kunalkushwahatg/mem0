#!/usr/bin/env python3
"""
Setup script for the Memory-Aware Chatbot
"""

import subprocess
import sys
import os

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running")
            return True
    except:
        pass
    
    print("❌ Ollama is not running or not accessible")
    print("Please start Ollama with: ollama serve")
    return False

def check_model():
    """Check if the required model is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Check for llama3.2 or similar models
            available_models = [name for name in model_names if 'llama' in name.lower()]
            
            if available_models:
                print(f"✅ Available models: {', '.join(available_models)}")
                return True
            else:
                print("❌ No Llama models found")
                print("Please install a model with: ollama pull llama3.2")
                return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def install_requirements():
    """Install Python requirements"""
    try:
        print("📦 Installing Python requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def main():
    print("🚀 Memory-Aware Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check Ollama
    if not check_ollama():
        print("\n📋 To start Ollama:")
        print("1. Install Ollama from https://ollama.ai")
        print("2. Run: ollama serve")
        print("3. In another terminal, run: ollama pull llama3.2")
        sys.exit(1)
    
    # Check model
    if not check_model():
        print("\n📋 To install a model:")
        print("Run: ollama pull llama3.2")
        sys.exit(1)
    
    print("\n🎉 Setup complete! You can now run the chatbot with:")
    print("python chat.py")

if __name__ == "__main__":
    main()
