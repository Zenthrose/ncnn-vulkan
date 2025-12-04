# NCNN LLM Platform - Installation Guide

## System Requirements

### Minimum Hardware
- **CPU**: x86_64, ARM64, or compatible architecture
- **RAM**: 8GB+ recommended (16GB+ for larger models)
- **Storage**: 10GB+ free space (for source code and build artifacts)
- **GPU**: Vulkan-compatible GPU recommended (NVIDIA, AMD, Intel integrated graphics)

### Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+, Fedora 30+, Arch Linux
- **Windows**: Windows 10+ with MSVC 2019+
- **macOS**: macOS 10.15+ (Intel/Apple Silicon)

## Build Tools and Dependencies

### Core Build Tools
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build

# CentOS/RHEL/Fedora
sudo dnf install -y gcc gcc-c++ cmake git ninja-build  # Fedora
sudo yum install -y gcc gcc-c++ cmake git ninja-build  # CentOS/RHEL

# Arch Linux
sudo pacman -S base-devel cmake git ninja

# macOS
xcode-select --install  # Installs clang and build tools
brew install cmake ninja

# Windows (using Chocolatey)
choco install cmake ninja git visualstudio2019community
```

### Vulkan SDK
```bash
# Ubuntu/Debian
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.4.313-jammy.list https://packages.lunarg.com/vulkan/1.4.313/lunarg-vulkan-1.4.313-jammy.list
sudo apt-get update
sudo apt-get install -y vulkan-sdk

# Or using LunarG installer (all platforms)
# Download from: https://vulkan.lunarg.com/sdk/home

# Arch Linux
sudo pacman -S vulkan-devel

# macOS
brew install molten-vk vulkan-headers

# Windows
# Download Vulkan SDK installer from https://vulkan.lunarg.com/sdk/home
```

### Qt Libraries (for GUI)
```bash
# Ubuntu/Debian
sudo apt-get install -y qtbase5-dev qt5-default qtbase5-dev-tools

# Or Qt6
sudo apt-get install -y qt6-base-dev qt6-tools-dev

# CentOS/RHEL/Fedora
sudo dnf install -y qt5-qtbase-devel  # Fedora

# Arch Linux
sudo pacman -S qt5-base qt6-base

# macOS
brew install qt5
# Or
brew install qt6

# Windows
# Download Qt installer from https://www.qt.io/download
# Or using vcpkg
vcpkg install qt5-base
```

### Additional Libraries
```bash
# Ubuntu/Debian
sudo apt-get install -y libcurl4-openssl-dev libssl-dev libprotobuf-dev protobuf-compiler

# CentOS/RHEL/Fedora
sudo dnf install -y libcurl-devel openssl-devel protobuf-devel protobuf-compiler

# Arch Linux
sudo pacman -S curl openssl protobuf

# macOS
brew install curl openssl protobuf

# Windows (vcpkg)
vcpkg install curl openssl protobuf
```

### Python Dependencies (for some tools)
```bash
# Install Python 3.8+
python3 --version

# Install required packages
pip3 install huggingface-hub  # For model downloading
```

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/Zenthrose/ncnn-vulkan.git
cd ncnn-vulkan
```

### 2. Initialize Submodules
```bash
git submodule update --init --recursive
```

### 3. Create Build Directory
```bash
mkdir build
cd build
```

### 4. Configure Build
```bash
# Basic build
cmake .. -DNCNN_VULKAN=ON

# With GUI support
cmake .. -DNCNN_VULKAN=ON -DBUILD_GUI=ON

# With additional options
cmake .. \
  -DNCNN_VULKAN=ON \
  -DBUILD_GUI=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_BUILD_TESTS=ON
```

### 5. Build
```bash
# Using make
make -j$(nproc)

# Or using ninja (if available)
ninja
```

### 6. Install (Optional)
```bash
sudo make install
```

## Model Setup

### Download Models
**Important**: You must download your own AI models. The repository does not include model files.

```bash
# Install huggingface CLI
pip3 install huggingface-hub

# Download example models
huggingface-cli download microsoft/Phi-3-mini-4k-instruct Phi-3-mini-4k-instruct-Q4_K_M.gguf --local-dir models/
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 Mistral-7B-Instruct-v0.1.Q4_0.gguf --local-dir models/
huggingface-cli download Qwen/Qwen2-1.5B-Instruct Qwen2-1.5B-Instruct.Q4_0.gguf --local-dir models/

# Or download manually from Hugging Face
# Visit: https://huggingface.co/models?pipeline_tag=text-generation
# Search for GGUF format models
```

## Testing Installation

### Basic Functionality Test
```bash
cd build

# Test GGUF loading
./load_gguf ../models/Phi-3-mini-4k-instruct-Q4_K_M.gguf

# Test LLM demo (if built)
./llm_platform_demo

# Test CLI
./llm_cli --help
```

### Tool System Test
```bash
# Test tools
./tool_cli calculate expression="2+3*4"
./tool_cli web_search query="AI news"
```

### GUI Test (if built)
```bash
./llm_gui
```

## Troubleshooting

### Common Issues

#### Vulkan Not Found
```bash
# Check Vulkan installation
vulkaninfo

# Install Vulkan SDK if missing
# Follow Vulkan SDK installation steps above
```

#### Qt GUI Not Building
```bash
# Check Qt installation
qmake --version

# Install Qt if missing
# Follow Qt installation steps above
```

#### CMake Errors
```bash
# Clean build
cd build
rm -rf *
cmake .. -DNCNN_VULKAN=ON

# Check CMake version
cmake --version
# Should be 3.12+
```

#### Submodule Issues
```bash
# Reinitialize submodules
git submodule deinit -f .
git submodule update --init --recursive
```

#### Memory Issues During Build
```bash
# Reduce parallel jobs
make -j2  # Instead of -j$(nproc)
```

### Performance Optimization

#### Vulkan Settings
```bash
# Set environment variables
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json  # Intel
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json  # AMD
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json  # NVIDIA
```

#### CPU Optimization
```bash
# Enable AVX optimizations
export NCNN_AVX=ON
export NCNN_AVX2=ON
```

## Platform-Specific Notes

### Ubuntu/Debian
- Use `apt-cache search` to find package names
- Some packages may require `universe` repository

### CentOS/RHEL
- Enable EPEL repository for additional packages
- Use `dnf` on newer versions, `yum` on older

### Arch Linux
- Most packages available in official repositories
- Use AUR for additional tools if needed

### macOS
- Use Homebrew for package management
- Ensure Xcode command line tools are installed

### Windows
- Use vcpkg for dependency management
- Visual Studio 2019+ required
- PowerShell may be needed for some commands

## Support

If you encounter issues:
1. Check this guide for your platform
2. Verify all dependencies are installed
3. Test with minimal build options first
4. Check GitHub issues for similar problems
5. Create an issue with your system info and error logs

## Contributing

When contributing:
1. Test builds on multiple platforms
2. Update this installation guide if new dependencies are added
3. Document any platform-specific issues
4. Provide clear installation instructions for new features
