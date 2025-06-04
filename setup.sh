#!/bin/bash

# Setup script for RNN Sine Wave Generator project
# This script automates the installation of all necessary Python packages
# Optimized for remote SSH connections on VSCode

set -e  # Exit on any error

echo "=========================================="
echo "RNN Sine Wave Generator - Setup Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python3 first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
print_success "Found $PYTHON_VERSION"

# Check if pip is installed
print_status "Checking pip installation..."
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3 first."
    exit 1
fi

print_success "pip3 is available"

# Upgrade pip to latest version
print_status "Upgrading pip to latest version..."
python3 -m pip install --upgrade pip --user

# Create virtual environment (optional but recommended)
VENV_DIR="venv_rnn_project"
if [ ! -d "$VENV_DIR" ]; then
    print_status "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip in virtual environment
print_status "Upgrading pip in virtual environment..."
pip install --upgrade pip

# Define required packages based on the imports in Main.py
print_status "Installing required Python packages..."

# Core scientific computing packages
PACKAGES=(
    "numpy>=1.21.0"
    "matplotlib>=3.5.0"
    "scipy>=1.7.0"
    "scikit-learn>=1.0.0"
    "tqdm>=4.62.0"
)

# Install core packages
for package in "${PACKAGES[@]}"; do
    print_status "Installing $package..."
    pip install "$package"
    if [ $? -eq 0 ]; then
        print_success "Successfully installed $package"
    else
        print_error "Failed to install $package"
        exit 1
    fi
done

# Function to detect GPU capabilities
detect_gpu() {
    local gpu_available=false
    local cuda_available=false
    local mps_available=false
    
    # Check for NVIDIA GPU and CUDA
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            print_status "NVIDIA GPU detected"
            cuda_available=true
            gpu_available=true
        fi
    fi
    
    # Check for Apple Silicon GPU (Metal Performance Shaders)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Check if running on Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            print_status "Apple Silicon detected - MPS acceleration available"
            mps_available=true
            gpu_available=true
        fi
    fi
    
    # Check for AMD GPU (ROCm) on Linux
    if [[ "$OSTYPE" == "linux-gnu"* ]] && command -v rocm-smi &> /dev/null; then
        if rocm-smi &> /dev/null; then
            print_status "AMD GPU detected"
            gpu_available=true
        fi
    fi
    
    echo "$gpu_available:$cuda_available:$mps_available"
}

# Detect GPU capabilities
print_status "Detecting GPU capabilities..."
GPU_INFO=$(detect_gpu)
IFS=':' read -r GPU_AVAILABLE CUDA_AVAILABLE MPS_AVAILABLE <<< "$GPU_INFO"

# Install PyTorch based on detected hardware
print_status "Installing PyTorch..."

if [[ "$CUDA_AVAILABLE" == "true" ]]; then
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    TORCH_TYPE="CUDA"
elif [[ "$MPS_AVAILABLE" == "true" ]]; then
    print_status "Installing PyTorch with MPS support for Apple Silicon..."
    pip install torch torchvision torchaudio
    TORCH_TYPE="MPS (Apple Silicon)"
else
    print_status "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    TORCH_TYPE="CPU"
fi

# Verify PyTorch installation
if python3 -c "import torch; print('PyTorch version:', torch.__version__)" &> /dev/null; then
    print_success "Successfully installed PyTorch ($TORCH_TYPE)"
    
    # Test GPU availability
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

# Check for MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Metal Performance Shaders) available: True')
else:
    print('MPS available: False')
"
else
    print_error "Failed to install PyTorch, trying fallback installation..."
    pip install torch torchvision torchaudio
    if python3 -c "import torch" &> /dev/null; then
        print_success "Successfully installed PyTorch (fallback)"
    else
        print_error "Failed to install PyTorch"
        exit 1
    fi
fi

# Additional useful packages for data science and visualization
OPTIONAL_PACKAGES=(
    "jupyter>=1.0.0"
    "ipython>=7.0.0"
    "seaborn>=0.11.0"
    "pandas>=1.3.0"
)

print_status "Installing optional packages for enhanced development experience..."
for package in "${OPTIONAL_PACKAGES[@]}"; do
    print_status "Installing $package..."
    pip install "$package"
    if [ $? -eq 0 ]; then
        print_success "Successfully installed $package"
    else
        print_warning "Failed to install $package (optional)"
    fi
done

# Create requirements.txt file for future reference
print_status "Generating requirements.txt file..."
pip freeze > requirements.txt
print_success "requirements.txt file created"

# Verify installation by checking imports and GPU functionality
print_status "Verifying package installations and GPU functionality..."
python3 -c "
import sys
packages_to_test = [
    ('numpy', 'np'),
    ('matplotlib.pyplot', 'plt'),
    ('torch', 'torch'),
    ('sklearn.decomposition', 'PCA'),
    ('scipy.optimize', 'root, least_squares'),
    ('scipy.linalg', 'eig'),
    ('tqdm', 'tqdm')
]

failed_imports = []
for package_info in packages_to_test:
    try:
        if len(package_info) == 2:
            package, alias = package_info
            exec(f'import {package} as {alias}')
        else:
            package = package_info[0]
            exec(f'import {package}')
        print(f'✓ {package}')
    except ImportError as e:
        print(f'✗ {package}: {e}')
        failed_imports.append(package)

# Test PyTorch GPU functionality
print('\n--- PyTorch GPU Compatibility Test ---')
import torch
print(f'PyTorch version: {torch.__version__}')

# Test CUDA
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.device_count()} device(s)')
    device = torch.device('cuda')
    # Test basic tensor operations on GPU
    try:
        test_tensor = torch.randn(10, 10).to(device)
        test_result = torch.matmul(test_tensor, test_tensor.t())
        print('✓ CUDA tensor operations working')
    except Exception as e:
        print(f'✗ CUDA tensor operations failed: {e}')
else:
    print('○ CUDA not available')

# Test MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ MPS (Apple Silicon GPU) available')
    device = torch.device('mps')
    # Test basic tensor operations on MPS
    try:
        test_tensor = torch.randn(10, 10).to(device)
        test_result = torch.matmul(test_tensor, test_tensor.t())
        print('✓ MPS tensor operations working')
    except Exception as e:
        print(f'✗ MPS tensor operations failed: {e}')
else:
    print('○ MPS not available')

# Always test CPU as fallback
print('✓ CPU available (fallback)')
device = torch.device('cpu')
try:
    test_tensor = torch.randn(10, 10).to(device)
    test_result = torch.matmul(test_tensor, test_tensor.t())
    print('✓ CPU tensor operations working')
except Exception as e:
    print(f'✗ CPU tensor operations failed: {e}')
    failed_imports.append('torch_cpu_ops')

# Determine recommended device for the project
if torch.cuda.is_available():
    recommended_device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    recommended_device = 'mps'
else:
    recommended_device = 'cpu'

print(f'\\nRecommended device for your project: {recommended_device}')

if failed_imports:
    print(f'\\nFailed imports: {failed_imports}')
    sys.exit(1)
else:
    print('\\nAll required packages imported successfully!')
"

if [ $? -eq 0 ]; then
    print_success "All package installations verified successfully!"
else
    print_error "Some packages failed verification"
    exit 1
fi

# Create a device detection helper script
print_status "Creating device detection helper script..."
cat > check_device.py << 'EOF'
#!/usr/bin/env python3
"""
Device Detection Helper for RNN Project
Run this script to check available compute devices
"""

import torch

def check_devices():
    print("=== PyTorch Device Detection ===")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Check CUDA
    if torch.cuda.is_available():
        print("✓ CUDA GPU(s) available:")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        recommended = "cuda"
    else:
        print("○ CUDA not available")
        recommended = None
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("✓ MPS (Apple Silicon GPU) available")
        if recommended is None:
            recommended = "mps"
    else:
        print("○ MPS not available")
    
    # CPU is always available
    print("✓ CPU available")
    if recommended is None:
        recommended = "cpu"
    
    print(f"\nRecommended device: {recommended}")
    
    # Test tensor operations
    print("\n=== Testing Tensor Operations ===")
    for device_name in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []) + (['mps'] if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else []):
        try:
            device = torch.device(device_name)
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.matmul(test_tensor, test_tensor.t())
            print(f"✓ {device_name.upper()}: Tensor operations working")
        except Exception as e:
            print(f"✗ {device_name.upper()}: {e}")
    
    return recommended

if __name__ == "__main__":
    recommended_device = check_devices()
    print(f"\nTo use in your code:")
    print(f"device = torch.device('{recommended_device}')")
EOF

chmod +x check_device.py
print_success "Created check_device.py - run 'python3 check_device.py' to test your setup"

# Instructions for VSCode remote SSH and GPU usage
print_status "Setup complete! Here are the next steps:"
echo ""
echo "1. To use this environment in VSCode:"
echo "   - Open the Command Palette (Cmd+Shift+P on macOS, Ctrl+Shift+P on Linux/Windows)"
echo "   - Select 'Python: Select Interpreter'"
echo "   - Choose the interpreter from: $(pwd)/$VENV_DIR/bin/python"
echo ""
echo "2. To activate this environment manually:"
echo "   source $(pwd)/$VENV_DIR/bin/activate"
echo ""
echo "3. To deactivate the environment:"
echo "   deactivate"
echo ""
echo "4. GPU Usage in your Python code:"
if [[ "$CUDA_AVAILABLE" == "true" ]]; then
    echo "   - CUDA GPU detected! Use: device = torch.device('cuda')"
    echo "   - For multiple GPUs: device = torch.device('cuda:0')  # Use GPU 0"
elif [[ "$MPS_AVAILABLE" == "true" ]]; then
    echo "   - Apple Silicon GPU detected! Use: device = torch.device('mps')"
else
    echo "   - No GPU detected. Use: device = torch.device('cpu')"
fi
echo ""
echo "5. Example device selection code for your Main.py:"
echo "   # Add this after your imports in Main.py:"
echo "   if torch.cuda.is_available():"
echo "       device = torch.device('cuda')"
echo "       print(f'Using CUDA GPU: {torch.cuda.get_device_name(0)}')"
echo "   elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():"
echo "       device = torch.device('mps')"
echo "       print('Using Apple Silicon GPU (MPS)')"
echo "   else:"
echo "       device = torch.device('cpu')"
echo "       print('Using CPU')"
echo ""
echo "6. To run the main script:"
echo "   python3 Main.py"
echo ""

print_success "Setup completed successfully!"
print_status "You can now run your RNN Sine Wave Generator project!"

echo "=========================================="