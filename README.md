# 🚀 Getting Started with Harmony

## 📌 Project Overview

Harmony is a high-performance C++ project for real-time **audio classification**. It transforms audio signals into rich feature sets and applies machine learning models to recognize audio classes such as speech, music genres, or environmental sounds. Designed with modularity and efficiency in mind, Harmony is optimized for both speed and flexibility.

## ✨ Features

* 🎤 **Audio Processing**: Efficiently parses and processes WAV and other supported audio formats.
* 📊 **Feature Extraction**: Implements MFCC, Spectrogram, and Chroma features.
* 🤖 **Machine Learning Models**: Integrates models using **mlpack**, **dlib**, and more (e.g., SVM, Random Forest, MLP).
* ⚡ **Optimized C++ Backend**: Built for speed and performance.
* 🛠️ **Modular & Configurable**: Easily adapt feature extractors or classifiers to your needs.

## 📦 Dependencies

Harmony relies on a powerful stack of libraries for processing, feature extraction, and machine learning. These are automatically handled via the Dockerfile, but for reference:

* **OpenCV** – for visual feature processing and spectrogram handling
* **FFTW** – for fast Fourier transforms
* **Eigen** – for linear algebra and matrix operations
* **Boost** – for various utilities and system handling
* **mlpack** – for high-performance machine learning algorithms
* **dlib** – for machine learning models and numerical optimization

All dependencies are installed in the Docker environment, so you don’t need to worry about setting them up manually.

## 📂 Project Structure

```
📁 Harmony
├── 📁 core            # Main source code and classifiers
├── 📁 deploy          # Dockerfile and deployment configs
├── 📁 docs            # Project documentation and LaTeX reports
├── 📁 include         # Header files
├── 📁 models          # Pretrained models and configs
├── 📁 tools           # Helper scripts and utilities
├── 📁 utils           # Utility functions and components
├── 📄 CMakeLists.txt    # Build system config
├── 📄 infer.cpp         # Inference entry point
├── 📄 Document.pdf       # Project report
└── 📄 README.md          # Project documentation
```

## 🚀 Getting Started with Docker + CMake

This guide walks you through setting up and running the Harmony audio classification project using Docker and CMake in a streamlined development environment like Visual Studio Code.

### 🐳 1. Build the Docker Image

Leverage the provided Dockerfile in the `deploy/` folder to create your development container. Visual Studio Code will typically detect it and prompt you to build the image automatically.

```bash
# From the deploy directory
docker build -t harmony-audio-classifier -f deploy/Dockerfile .
```

### 🧱 2. Open the Project in the Docker Container

Once your Docker image is ready, use Visual Studio Code's **Remote - Containers** extension to open the project inside the container. This ensures a consistent environment across different machines.

> 💡 VS Code should automatically prompt you to "Reopen in Container" if the `.devcontainer` directory is present.

### ⚙️ 3. Build the Project with CMake

With the project running inside the container:

* Visual Studio Code may auto-detect the `CMakeLists.txt` and prompt you to configure the project.
* Or, build it manually in the terminal:

```bash
mkdir build && cd build
cmake ..
make
```

### 📁 4. Locate the Output (bin) Folder

After a successful build, you’ll find a `bin/` directory in the root or `build/` folder, containing the compiled executables.

### 🧪 5. Run the Inference Tool

Run the `infer` executable from the terminal to classify your audio samples:

```bash
./bin/infer path_to_test_data
```

Make sure the input file exists and is a supported audio format.

### ⚠️ 6. Important Notes on Audio Integrity

Please note:

* If an audio file is **corrupted** or **fails to load**, we remove it from the dataset.
* This can affect downstream results if the dataset expects that file to be present.

> ❗ If you're running test cases and need placeholder results for missing files, you will have to adjust the logic accordingly.

---

🎷 Enjoy working with Harmony! For any issues, open a GitHub issue or reach out directly. Happy coding!
