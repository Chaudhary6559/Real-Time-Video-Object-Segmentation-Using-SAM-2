# Real-Time Video Object Segmentation Using SAM 2

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.2.0+](https://img.shields.io/badge/pytorch-2.2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Windows Support](https://img.shields.io/badge/windows-10%2F11-0078d4)](https://www.microsoft.com/windows)

## 📋 Project Overview

This project develops a **real-time video object segmentation system** using **Segment Anything Model 2 (SAM 2)**. The main goal is to achieve robust object tracking and segmentation across video frames while handling challenges like rapid motion, occlusion, appearance changes, and restricted hardware resources.

### 🎯 Key Features

- **Real-time Video Segmentation**: Process video frames at high speed with minimal latency
- **Multi-model Support**: Choice between SAM 2 Hiera-S, Hiera-B, and Hiera-L models
- **GPU & CPU Support**: Optimized for both NVIDIA CUDA and CPU inference
- **Web Interface**: User-friendly Streamlit-based web application
- **Live Streaming**: Support for real-time camera/webcam input
- **Batch Processing**: Process multiple videos efficiently
- **Advanced Metrics**: Comprehensive evaluation metrics and visualization
- **Edge Device Support**: Optimized for resource-constrained devices

### 🔬 Scientific Foundation

Video object segmentation (VOS) is a fundamental computer vision task with applications in:
- **Autonomous Driving**: Identifying and tracking pedestrians and vehicles
- **Medical Imaging**: Surgical video guidance and organ tracking
- **Video Editing**: Object removal, background replacement, and motion tracking
- **Robotics**: Real-time object tracking and manipulation

SAM 2 builds upon the original Segment Anything Model with a **streaming memory mechanism** for real-time video processing, achieving state-of-the-art performance across diverse video segmentation benchmarks.

---

## 🏗️ Project Structure

```
Real-Time-Video-Object-Segmentation-Using-SAM-2/
│
├── 📄 README.md                          # Project documentation (this file)
├── 📄 SETUP_GUIDE.md                     # Detailed setup instructions for Windows
├── 📄 VIDEO_PROCESSING_FIX.md            # Video processing troubleshooting guide
├── 📝 requirements.txt                    # Python dependencies
│
├── 🚀 run.bat                            # Windows batch script to start application
├── 🚀 run.ps1                            # PowerShell script to start application
│
├── 🐍 setup_custom_hiera_b.py            # Custom Hiera-B model configuration
├── 🐍 test_custom_hiera_b.py             # Tests for custom Hiera-B model
├── 🐍 test_model_loading.py              # Model loading verification tests
│
├── 📁 app/                               # Main application directory
│   ├── main.py                           # Streamlit web application entry point
│   ├── pages/                            # Multi-page Streamlit application
│   │   ├── video_processing.py           # Video upload and processing
│   │   ├── live_streaming.py             # Live camera streaming
│   │   ├── model_management.py           # Model loading and configuration
│   │   ├── metrics_results.py            # Results visualization and metrics
│   │   └── settings.py                   # Application settings
│   └── utils/                            # Application utilities
│       ├── ui_components.py              # Streamlit UI components
│       ├── session_manager.py            # Session state management
│       └── constants.py                  # Configuration constants
│
├── 📁 backend/                           # Backend processing services
│   ├── sam2_model.py                     # SAM 2 model wrapper and inference
│   ├── video_loader.py                   # Video loading with OpenCV fallback
│   ├── preprocessing.py                  # Video preprocessing pipeline
│   ├── postprocessing.py                 # Post-processing and visualization
│   ├── metrics.py                        # Evaluation metrics calculation
│   ├── memory_optimization.py            # Memory management strategies
│   ├── model_trainer.py                  # Model fine-tuning and training
│   └── inference_engine.py               # High-performance inference engine
│
├── 📁 models/                            # Model weights and checkpoints
│   ├── checkpoints/                      # Downloaded model weights
│   │   ├── sam2_hiera_small.pt          # Hiera-S model weights
│   │   ├── sam2_hiera_base.pt           # Hiera-B model weights
│   │   └── sam2_hiera_large.pt          # Hiera-L model weights
│   └── custom/                           # Custom trained models
│       └── fine_tuned_models/            # User-trained model checkpoints
│
├── 📁 segment-anything-2/                # Official SAM 2 repository (submodule)
│   ├── sam2/                             # Core SAM 2 implementation
│   │   ├── modeling/                     # Model architecture
│   │   │   ├── backbones/               # Vision transformers
│   │   │   ├── memory_attention.py      # Memory attention mechanism
│   │   │   └── image_encoder.py         # Image encoding module
│   │   ├── modeling/sam2_base.py        # Base SAM 2 model class
│   │   ├── predictor.py                 # Predictor interface
│   │   ├── build_sam.py                 # Model building utilities
│   │   └── utils/                       # Helper utilities
│   ├── sav_dataset/                      # Segment Anything Video dataset tools
│   │   ├── sa_v_dataset.py              # Dataset implementation
│   │   └── requirements.txt              # SAV dataset dependencies
│   └── README.md                         # SAM 2 documentation
│
├── 📁 utils/                             # Utility modules
│   ├── config.py                         # Configuration management
│   ├── logger.py                         # Logging setup
│   ├── path_manager.py                   # Path utilities
│   ├── file_handler.py                   # File I/O operations
│   ├── data_loader.py                    # Data loading utilities
│   └── validators.py                     # Input validation
│
├── 📁 data/                              # Data directory (not in repo)
│   ├── videos/                           # Input video files
│   │   ├── sample_videos/                # Example videos
│   │   └── user_uploads/                 # User-uploaded videos
│   ├── outputs/                          # Processing results
│   │   ├── segmentation_masks/           # Segmentation output masks
│   │   ├── tracked_frames/               # Annotated video frames
│   │   └── metrics/                      # Evaluation results
│   └── training/                         # Training data
│       ├── images/                       # Training images
│       └── annotations/                  # Training annotations
│
├── 📁 logs/                              # Application logs
│   ├── inference.log                     # Inference logs
│   ├── training.log                      # Training logs
│   └── errors.log                        # Error logs
│
├── 📁 tests/                             # Unit and integration tests
│   ├── test_sam2_model.py                # SAM 2 model tests
│   ├── test_video_loader.py              # Video loader tests
│   ├── test_preprocessing.py             # Preprocessing tests
│   ├── test_postprocessing.py            # Post-processing tests
│   ├── test_metrics.py                   # Metrics calculation tests
│   └── test_integration.py               # End-to-end integration tests
│
└── 📁 docs/                              # Documentation
    ├── API.md                            # API documentation
    ├── ARCHITECTURE.md                   # System architecture
    ├── PERFORMANCE.md                    # Performance benchmarks
    ├── TROUBLESHOOTING.md                # Common issues and solutions
    └── CONTRIBUTING.md                   # Contribution guidelines

```

---

## 💻 Technology Stack

### Core Framework
- **PyTorch 2.2.0+** - Deep learning framework
- **Torchvision 0.17.0+** - Computer vision utilities
- **SAM 2** - Segment Anything Model 2 (from Meta)

### Web & API
- **FastAPI 0.104.1** - High-performance web framework
- **Uvicorn 0.24.0** - ASGI server
- **Streamlit 1.29.0** - Interactive web application framework

### Computer Vision
- **OpenCV 4.8.1** - Image and video processing
- **Scikit-image 0.22.0** - Image processing algorithms
- **Pillow 10.1.0** - Image library

### Model & Vision
- **Timm 0.9.12** - PyTorch Image Models
- **Transformers 4.36.2** - Hugging Face transformers
- **Hydra-Core 1.3.2** - Configuration management

### Data Processing
- **NumPy 1.24+** - Numerical computing
- **SciPy 1.11.4** - Scientific computing
- **Pandas 2.1.3** - Data manipulation
- **Scikit-learn 1.3.2** - Machine learning utilities

### Evaluation & Metrics
- **PyTorch Metrics 1.2.1** - Torchmetrics
- **COCO Tools 2.0.7** - COCO dataset evaluation
- **Albumentations 1.3.1** - Image augmentation

### Utilities
- **Loguru 0.7.2** - Logging library
- **Pydantic 2.5.0** - Data validation
- **Tqdm 4.66.1** - Progress bars
- **Python-dotenv 1.0.0** - Environment configuration

### Development Tools
- **Pytest 7.4.3** - Testing framework
- **Black 23.12.0** - Code formatter
- **Flake8 6.1.0** - Code linter
- **Mypy 1.7.1** - Static type checking

---

## 🚀 Quick Start

### Prerequisites
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.9 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Disk Space**: 5GB minimum
- **GPU** (Optional): NVIDIA GPU with CUDA 11.8+ for faster inference

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Chaudhary6559/Real-Time-Video-Object-Segmentation-Using-SAM-2.git
   cd Real-Time-Video-Object-Segmentation-Using-SAM-2
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install SAM 2 (Local)**
   ```bash
   pip install -e segment-anything-2/
   ```

4. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Run the Application**
   
   **Option A: Streamlit Web Interface**
   ```bash
   streamlit run app/main.py
   ```
   
   **Option B: Windows Scripts** (Automatic setup)
   ```bash
   # Command Prompt
   run.bat
   
   # Or PowerShell
   ./run.ps1
   ```

6. **Access Application**
   - Open browser to: `http://localhost:8501`
   - Load a model from the sidebar
   - Upload a video and start processing

---

## 📖 Usage Guide

### Loading a Model

1. Go to **Model Management** tab in sidebar
2. Select model size:
   - **SAM 2 Hiera-S** - Fastest (2-3 FPS on CPU, 15-20 FPS on GPU)
   - **SAM 2 Hiera-B** - Balanced (1-2 FPS on CPU, 8-12 FPS on GPU) ⭐ Recommended
   - **SAM 2 Hiera-L** - Most Accurate (0.5-1 FPS on CPU, 4-6 FPS on GPU)
3. Select device (CPU or CUDA GPU)
4. Click "Load Model"

### Video Processing

1. Go to **Video Upload & Processing** tab
2. Upload a video file (MP4, AVI, MOV, WebM)
3. Configure processing:
   - Target resolution (512×512 to 1024×1024)
   - Frame sampling rate
   - Confidence threshold
4. Click "Process Video"
5. View results in **Metrics & Results** tab

### Live Streaming

1. Go to **Live Streaming** tab
2. Select camera index (0 for default)
3. Set confidence threshold
4. Click "Start Streaming"
5. Objects are tracked in real-time

### Metrics & Visualization

- **Segmentation Masks**: Binary masks for each frame
- **Tracked Objects**: Bounding boxes and masks over time
- **Frame Metrics**: Per-frame accuracy and performance
- **Video Metrics**: Overall video statistics
- **Download Results**: Export segmentation masks and annotated videos

---

## 🎓 Model Information

### SAM 2 Architecture

SAM 2 extends the original SAM with:

1. **Streaming Memory Mechanism**
   - Maintains temporal context across frames
   - Handles occlusions and reappearances
   - Reduces error accumulation

2. **Memory Attention Blocks**
   - Computes attention over frame memories
   - Enables efficient long-range dependencies
   - Optimizable for edge devices

3. **Image Encoder**
   - Hierarchical Vision Transformer (Hiera)
   - Three sizes: Small (S), Base (B), Large (L)
   - Efficient feature extraction

### Performance Benchmarks

| Model | CPU (FPS) | GPU RTX 3080 (FPS) | Memory Usage | Accuracy |
|-------|-----------|-------------------|--------------|----------|
| Hiera-S | 2-3 | 15-20 | 2-3 GB | Good |
| Hiera-B | 1-2 | 8-12 | 3-4 GB | Very Good ⭐ |
| Hiera-L | 0.5-1 | 4-6 | 4-6 GB | Excellent |

---

## 🔧 Advanced Configuration

### Custom Model Path
```python
# Edit app/main.py
model = SAM2Model(
    model_type="SAM 2 Hiera-B",
    checkpoint_dir="custom/path/to/checkpoints"
)
```

### Processing Parameters
```python
# Edit backend/preprocessing.py
processor = VideoProcessor(
    fps=10,                    # Target FPS
    target_size=1024,          # Resolution
    normalize=True             # Normalize input
)
```

### Training Configuration
```python
# Edit app/main.py
trainer = ModelTrainer(
    model=model,
    epochs=50,
    batch_size=16,
    learning_rate=1e-4
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "Python not found"**
   - Reinstall Python from [python.org](https://python.org)
   - Ensure "Add Python to PATH" is checked

**2. "Module not found"**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

**3. "Port 8501 already in use"**
   ```bash
   # Windows
   netstat -ano | findstr :8501
   taskkill /PID <PID> /F
   ```

**4. "Out of memory"**
   - Close other applications
   - Reduce video resolution (512×512)
   - Lower FPS (5-10)
   - Use smaller video files

**5. "CUDA not available" (GPU users)**
   - Install NVIDIA drivers
   - Install CUDA 11.8+ and cuDNN
   - Restart computer

See **SETUP_GUIDE.md** and **VIDEO_PROCESSING_FIX.md** for detailed troubleshooting.

---

## 📊 Related Research

This project builds upon recent advances in video segmentation:

### Key Papers

1. **SAM 2: Segment Anything in Images and Videos** (2024)
   - Ravi et al. - Foundation model for unified segmentation
   - [arXiv:2408.00714](https://arxiv.org/abs/2408.00714)

2. **SAM2Long: Enhancing SAM 2 for Long Video Segmentation** (2024)
   - Training-free memory tree for long videos
   - [arXiv:2410.16268](https://arxiv.org/abs/2410.16268)

3. **Surgical SAM 2: Real-time Segment Anything in Surgical Video** (2024)
   - Efficient frame pruning for medical videos
   - [arXiv:2408.07931](https://arxiv.org/abs/2408.07931)

4. **EdgeTAM: On-Device Track Anything Model** (2025)
   - Mobile deployment optimization
   - [arXiv:2501.07256](https://arxiv.org/abs/2501.07256)

---

## 🤝 Contributing

Contributions are welcome! Areas for contribution:

- **Performance Optimization**: Improve speed and memory efficiency
- **Feature Addition**: New segmentation modes or tracking methods
- **Bug Fixes**: Report and fix issues
- **Documentation**: Improve guides and examples
- **Testing**: Add comprehensive test coverage

Please see **CONTRIBUTING.md** for guidelines.

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

The SAM 2 model weights are under the following licenses:
- Model weights: CC-BY-NC-4.0 or Apache 2.0
- Code: Apache 2.0

---

## 🙏 Acknowledgments

- **Meta AI** for the Segment Anything Model 2
- **PyTorch** team for the deep learning framework
- **Streamlit** for the web application framework
- All contributors and researchers in the computer vision community

---

## 📞 Support & Contact

- **Issues**: Open an issue on GitHub
- **Documentation**: Check docs/ directory
- **Setup Help**: See SETUP_GUIDE.md
- **Troubleshooting**: See VIDEO_PROCESSING_FIX.md

---

## 🔗 Useful Links

- [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [SAM 2 Paper](https://arxiv.org/abs/2408.00714)
- [Streamlit Documentation](https://docs.streamlit.io)
- [PyTorch Documentation](https://pytorch.org/docs)
- [OpenCV Documentation](https://docs.opencv.org)

---

## 📈 Project Status

- ✅ Core SAM 2 integration
- ✅ Web interface (Streamlit)
- ✅ Video processing pipeline
- ✅ Live streaming support
- ✅ Metrics and visualization
- 🔄 Model optimization (in progress)
- 🔄 Performance benchmarking (in progress)
- 📋 API documentation (planned)
- 📋 Advanced training UI (planned)

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@misc{chaudhary2025realtime,
  title={Real-Time Video Object Segmentation Using SAM 2},
  author={Chaudhary},
  year={2025},
  howpublished={\url{https://github.com/Chaudhary6559/Real-Time-Video-Object-Segmentation-Using-SAM-2}}
}
```

And cite the original SAM 2 paper:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and others},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

---

## 🎯 Future Roadmap

- [ ] Performance optimization for edge devices
- [ ] Integration with popular video annotation tools
- [ ] Support for additional model architectures
- [ ] Advanced training interface
- [ ] Cloud deployment support
- [ ] Mobile app version
- [ ] Real-time multi-object tracking improvements
- [ ] Dataset generation tools

---

**Version**: 1.0.0  
**Last Updated**: October 2026  
**Maintainer**: [@Chaudhary6559](https://github.com/Chaudhary6559)

**⭐ If you find this project helpful, please consider giving it a star!**

---

**Happy Segmenting! 🎯**
