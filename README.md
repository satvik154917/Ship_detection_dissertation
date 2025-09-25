# Ship Detection Dissertation Project

This repository contains the implementation of ship detection algorithms using deep learning techniques for both SAR (Synthetic Aperture Radar) and optical satellite imagery.

## Project Overview

This project implements and compares different deep learning approaches for ship detection:

- **YOLO (You Only Look Once)** - Real-time object detection
- **Faster R-CNN** - Two-stage object detection with high accuracy

Both models are trained and tested on:
- **SAR imagery** - For all-weather detection capabilities
- **Optical imagery** - For high-resolution visual detection

## Files Description

### Python Implementation Files
- `copy_of_yolo_sar_done.py` - YOLO implementation for SAR imagery
- `copy_of_yolo_optical_done.py` - YOLO implementation for optical imagery
- `copy_of_fast_r_cnn_sar_done.py` - Faster R-CNN implementation for SAR imagery
- `copy_of_faster_r_cnn_optical_done.py` - Faster R-CNN implementation for optical imagery

### Large Files (Not included in repository due to GitHub size limits)
Due to GitHub's file size limitations, the following large files are not included in this repository:

**Trained Model Files:**
- `final_sar_ship_detector.pt` (6.0MB) - Trained YOLO model for SAR data
- `ship_detection_model_from_scratch.pt` (6.0MB) - Custom trained detection model
- `faster_rcnn_ship_detector.pth` (158MB) - Trained Faster R-CNN model for optical data
- `faster_rcnn_sar_best (2).pth` (472MB) - Best performing Faster R-CNN model for SAR

**Dataset Files:**
- `SAR_Dataset (1).zip` (438MB) - SAR imagery dataset
- `Optical_yolo.zip` (1.8GB) - Optical imagery dataset formatted for YOLO

**Note:** If you need access to the trained models and datasets, please contact the author or use cloud storage solutions like Google Drive, Dropbox, or similar platforms for sharing these large files.

## Requirements

```
Python 3.8+
PyTorch
OpenCV
NumPy
Matplotlib
torchvision
Pillow
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/satvik154917/Ship_detection_dissertation.git
cd Ship_detection_dissertation
```

2. Install required dependencies:
```bash
pip install torch torchvision opencv-python numpy matplotlib pillow
```

3. Obtain the dataset and model files (contact author for access due to file size constraints)

## Usage

### For YOLO Implementation:
```bash
# For SAR imagery
python copy_of_yolo_sar_done.py

# For optical imagery
python copy_of_yolo_optical_done.py
```

### For Faster R-CNN Implementation:
```bash
# For SAR imagery
python copy_of_fast_r_cnn_sar_done.py

# For optical imagery
python copy_of_faster_r_cnn_optical_done.py
```

## Model Performance

The project demonstrates the effectiveness of different deep learning approaches for ship detection in satellite imagery, providing insights into the performance trade-offs between:

- **Speed vs Accuracy**: YOLO offers faster inference while Faster R-CNN provides higher accuracy
- **Data Modality**: Comparison of performance across SAR and optical imagery
- **Detection Challenges**: Handling various ship sizes, orientations, and environmental conditions

## Project Structure

```
Ship_detection_dissertation/
│
├── copy_of_yolo_sar_done.py           # YOLO SAR implementation
├── copy_of_yolo_optical_done.py       # YOLO optical implementation
├── copy_of_fast_r_cnn_sar_done.py     # Faster R-CNN SAR implementation
├── copy_of_faster_r_cnn_optical_done.py # Faster R-CNN optical implementation
├── README.md                           # Project documentation
└── .gitignore                          # Git ignore rules
```

## Results and Analysis

This dissertation project provides a comprehensive comparison of:
1. **YOLO vs Faster R-CNN** performance on maritime detection tasks
2. **SAR vs Optical imagery** effectiveness for ship detection
3. **Real-world applicability** of different detection approaches
4. **Computational efficiency** analysis

## Contributing

This is a dissertation project. For questions or collaboration inquiries, please contact the author.

## License

This project is for academic purposes only. Please respect the terms of use for any datasets utilized.

## Author

**Satvik Arora**  
Dissertation Project - Ship Detection in Satellite Imagery  
Email: arorabro1549@gmail.com  
GitHub: [@satvik154917](https://github.com/satvik154917)

## Acknowledgments

- Thanks to the providers of SAR and optical satellite imagery datasets
- PyTorch community for deep learning framework support
- YOLO and Faster R-CNN research communities for foundational work

---

*Note: This repository contains the code implementation only. Due to GitHub's file size limitations (100MB per file), trained models and datasets are stored separately. Contact the author for access to the complete project files including trained models and datasets.*
