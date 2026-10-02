# SAM 2 Complete Setup Guide - Windows 10/11

Complete step-by-step guide to get SAM 2 Video Segmentation running on Windows.

## ✅ Pre-Setup Checklist

Before starting, ensure you have:
- [ ] Windows 10 or Windows 11
- [ ] Internet connection
- [ ] Administrator access
- [ ] At least 5GB free disk space
- [ ] 8GB RAM minimum (16GB recommended)

## 📥 Step 1: Install Python (5 minutes)

### Download Python

1. Visit [python.org](https://www.python.org/downloads/)
2. Download **Python 3.10** or **3.11** (not 3.12 yet, as some packages aren't compatible)
3. Run the installer

### Install Python

1. **IMPORTANT**: Check the box "Add Python to PATH"
2. Click "Install Now"
3. Wait for installation to complete
4. Click "Disable path length limit" (optional but recommended)

### Verify Installation

Open Command Prompt (Win+R → type `cmd` → Enter):

```cmd
python --version
pip --version
```

Both should show version numbers. If not, reinstall Python and check "Add Python to PATH".

## 📂 Step 2: Extract Project (2 minutes)

1. Download `sam2_complete.zip`
2. Right-click → "Extract All"
3. Choose location: `C:\Users\YourName\Documents\sam2_complete`
4. Click "Extract"

## 🚀 Step 3: Run Application (2 minutes)

### Option A: Double-Click (Easiest)

1. Open extracted folder
2. Double-click **`run.bat`** (Command Prompt version)
   - OR double-click **`run.ps1`** (PowerShell version)
3. Wait for dependencies to install (first time takes 2-5 minutes)
4. Browser opens automatically to http://localhost:8501

### Option B: Manual Setup

1. Open Command Prompt
2. Navigate to project:
   ```cmd
   cd C:\Users\YourName\Documents\sam2_complete
   ```

3. Create virtual environment:
   ```cmd
   python -m venv venv
   ```

4. Activate virtual environment:
   ```cmd
   venv\Scripts\activate
   ```

5. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

6. Run application:
   ```cmd
   streamlit run app/main.py
   ```

7. Open browser to: http://localhost:8501

## 🎯 Step 4: First Use (5 minutes)

### Load Model

1. In sidebar, click "🔄 Load Model"
2. Select model size:
   - **SAM 2 Hiera-B** (recommended, balanced)
   - **SAM 2 Hiera-S** (faster, less accurate)
   - **SAM 2 Hiera-L** (slower, more accurate)
3. Select device:
   - **CPU** (slower, works on all systems)
   - **CUDA (GPU)** (faster, requires NVIDIA GPU)
4. Click "Load Model"
5. Wait for model to load (first time: 1-2 minutes)

### Upload Video

1. Click "📹 Video Upload & Processing" tab
2. Click "Choose a video file"
3. Select a video (MP4, AVI, MOV, WebM)
4. Configure processing options
5. Click "🚀 Process Video"
6. Wait for processing to complete
7. View results in "📊 Metrics & Results" tab

## 🔧 Troubleshooting

### Issue: "Python not found"

**Solution:**
1. Reinstall Python from [python.org](https://python.org/)
2. **IMPORTANT**: Check "Add Python to PATH"
3. Restart Command Prompt
4. Verify: `python --version`

### Issue: "Module not found" or "pip install fails"

**Solution:**
```cmd
cd C:\path\to\sam2_complete
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"

**Solution:**
```cmd
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

Then restart the application.

### Issue: "Out of memory" error

**Solutions:**
1. Close other applications
2. Reduce video resolution (use 512×512 instead of 1024×1024)
3. Lower FPS (use 5 FPS instead of 10)
4. Use smaller video files
5. Enable INT8 Quantization in sidebar

### Issue: "CUDA not available" (if you have GPU)

**Solutions:**
1. Install NVIDIA drivers from [nvidia.com](https://www.nvidia.com/Download/driverDetails.aspx)
2. Install CUDA 11.8+ from [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads)
3. Install cuDNN from [nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
4. Restart computer
5. Try again

### Issue: "Streamlit not found"

**Solution:**
```cmd
pip install streamlit
```

### Issue: "Video preview not working"

**Solutions:**
1. Use MP4 format instead of WebM
2. Try a different video file
3. Check browser console (F12 → Console tab) for errors
4. Try a different browser (Chrome recommended)

### Issue: Application won't start

**Solution:**
```cmd
cd C:\path\to\sam2_complete
venv\Scripts\activate
streamlit run app/main.py --logger.level=debug
```

This will show detailed error messages.

## 📊 System Performance

### CPU Performance
- **Hiera-S**: 2-3 FPS
- **Hiera-B**: 1-2 FPS
- **Hiera-L**: 0.5-1 FPS

### GPU Performance (NVIDIA RTX 3080)
- **Hiera-S**: 15-20 FPS
- **Hiera-B**: 8-12 FPS
- **Hiera-L**: 4-6 FPS

### Memory Usage
- **Hiera-S**: 2-3 GB
- **Hiera-B**: 3-4 GB
- **Hiera-L**: 4-6 GB

## 🎓 Usage Tips

### Video Processing

1. **Best Results**:
   - Use MP4 format
   - 720p or 1080p resolution
   - 24-30 FPS
   - Clear lighting
   - Distinct objects

2. **Faster Processing**:
   - Use 512×512 target size
   - Lower FPS (5-10)
   - Smaller video files
   - Use Hiera-S model

3. **Better Accuracy**:
   - Use 1024×1024 target size
   - Higher FPS (15-30)
   - Use Hiera-L model
   - Good lighting

### Live Streaming

1. **Camera Selection**:
   - Index 0: Default camera
   - Index 1-3: Other cameras if available

2. **Best Performance**:
   - Good lighting
   - Stable camera
   - Close-up objects
   - Avoid fast motion

3. **Tips**:
   - Adjust confidence threshold
   - Monitor FPS
   - Watch memory usage
   - Stop when not needed

### Training

1. **Data Preparation**:
   - Use high-quality images
   - Clear annotations
   - Balanced classes
   - At least 100 images per class

2. **Training Tips**:
   - Start with small dataset
   - Use lower learning rate (1e-5)
   - Enable augmentation
   - Monitor loss curves

3. **Checkpoints**:
   - Save every 5 epochs
   - Keep best model
   - Compare metrics

## 🔐 Security Notes

1. **Keep Credentials Safe**:
   - Don't share .env files
   - Use strong passwords
   - Protect API keys

2. **Data Privacy**:
   - Videos processed locally
   - No data sent to cloud (by default)
   - Results stored locally

3. **Updates**:
   - Keep Python updated
   - Update dependencies regularly
   - Check for security patches

## 📚 Learning Resources

### SAM 2 Documentation
- [GitHub Repository](https://github.com/facebookresearch/segment-anything-2)
- [Research Paper](https://arxiv.org/abs/2401.01808)
- [Official Docs](https://github.com/facebookresearch/segment-anything-2/blob/main/README.md)

### Streamlit
- [Official Website](https://streamlit.io/)
- [Documentation](https://docs.streamlit.io/)
- [Gallery](https://streamlit.io/gallery)

### PyTorch
- [Official Website](https://pytorch.org/)
- [Documentation](https://pytorch.org/docs/)
- [Tutorials](https://pytorch.org/tutorials/)

## ✨ Advanced Configuration

### Custom Model Path

Edit `app/main.py`:
```python
model = SAM2Model(
    model_type="SAM 2 Hiera-B",
    checkpoint_dir="custom/path/to/checkpoints"
)
```

### Custom Processing Parameters

Edit `backend/preprocessing.py`:
```python
processor = VideoProcessor(
    fps=10,
    target_size=1024,
    normalize=True
)
```

### Custom Training Parameters

Edit `app/main.py`:
```python
trainer = ModelTrainer(
    model=model,
    epochs=50,
    batch_size=16,
    learning_rate=1e-4
)
```

## 🐛 Debug Mode

Run with debug logging:
```cmd
streamlit run app/main.py --logger.level=debug
```

This shows detailed information about what's happening.

## 📞 Getting Help

1. **Check README.md** - General information
2. **Check Troubleshooting** - Common issues
3. **Check Logs** - Look in `logs/` folder
4. **Browser Console** - Press F12 in browser
5. **Command Prompt** - Check for error messages

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] Python installed (`python --version`)
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip list`)
- [ ] Application starts (`streamlit run app/main.py`)
- [ ] Browser opens to http://localhost:8501
- [ ] Can load model
- [ ] Can upload video
- [ ] Can view results
- [ ] Can download results
- [ ] Metrics display correctly

## 🎉 You're Ready!

If all items are checked, you're ready to use SAM 2 Video Segmentation!

### Next Steps:
1. Upload a video
2. View segmentation results
3. Check metrics
4. Try live streaming
5. Fine-tune model with your data

### Need Help?
- Check README.md for features
- Check this guide for troubleshooting
- Check browser console (F12) for errors
- Check Command Prompt for error messages

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Platform**: Windows 10/11  
**Python**: 3.9+

**Happy Segmenting! 🎯**
