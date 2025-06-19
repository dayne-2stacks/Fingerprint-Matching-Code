# 🔬 Pore Detection - Complete Setup Guide

## 📋 Prerequisites
- Conda environment named `detection` must be available
- CUDA-capable GPU (recommended)
- L3-SF dataset in `../dataset/L3-SF/`

## 🚀 Quick Start (3 Steps)

### Step 1: Activate Environment
```bash
conda activate detection
```

### Step 2: Verify Setup
```bash
python3 verify_setup.py
```

### Step 3: Run Pore Detection
```bash
# Process first 50 images
python3 pore_detect_clean.py --start_index 0 --end_index 50

# Process specific range (e.g., images 100-200)  
python3 pore_detect_clean.py --start_index 100 --end_index 200

# Process large batch (e.g., first 1000 images)
python3 pore_detect_clean.py --start_index 0 --end_index 1000
```

## 📁 Output Files

After running detection, you'll find:

1. **Pore Coordinates** (`.txt` files)
   - Location: `out_of_the_box_detect/Prediction/Coordinates/`
   - Format: Each line contains `x, y` coordinates of detected pores

2. **Visualization Images** (`.png` files)
   - Location: `out_of_the_box_detect/Prediction/Fingerprint/`
   - Shows original fingerprint with red circles marking detected pores

3. **Binary Pore Maps** 
   - Location: `out_of_the_box_detect/Prediction/Pore/`
   - Binary images highlighting pore locations

## 🛠️ Advanced Options

### Use CPU instead of GPU:
```bash
python3 pore_detect_clean.py --device cpu --start_index 0 --end_index 50
```

### Use different model (if available):
```bash
python3 pore_detect_clean.py --features 64 --start_index 0 --end_index 50
```

## 📊 Dataset Information

- **Total Images**: 7,400 fingerprint images
- **Resolution Folders**: R1, R2, R3, R4, R5 (different DPI)
- **Format**: PNG files
- **Naming**: `{id}_{finger}_{impression}.png`

## 🔧 Troubleshooting

### If you get "CUDA out of memory":
```bash
python3 pore_detect_clean.py --device cpu
```

### If model file not found:
- Check `out_of_the_box_detect/models/40` exists
- Try different feature size: `--features 32` or `--features 64`

### If dataset not found:
- Ensure L3-SF dataset is in `../dataset/L3-SF/`  
- Check with: `ls ../dataset/L3-SF/`

## 📈 Performance Tips

- **GPU**: ~10-20 images/second
- **CPU**: ~1-3 images/second  
- **Recommended batch size**: 50-100 images for testing, 500-1000 for production

## 🧹 Clean Results (Start Fresh)

To clear previous results:
```bash
find out_of_the_box_detect/Prediction -name "*.txt" -delete
find out_of_the_box_detect/Prediction -name "*.png" -delete  
```

---
*Generated on $(date) - Streamlined pore detection setup*
