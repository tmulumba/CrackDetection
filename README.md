# Crack Detection

Deep learning-based crack segmentation for geotechnical applications using semantic segmentation models.

## Overview

This project implements crack detection in rock images using multiple segmentation architectures:
- **U-Net**
- **Linknet**
- **PSPNet**
- **FPN (Feature Pyramid Network)**

All models use a **ResNet34** backbone with ImageNet pre-trained weights via the [segmentation-models](https://github.com/qubvel/segmentation_models) library.

## Project Structure

```
CrackDetection/
├── data/
│   ├── train_img/      # Training images (.jpg)
│   ├── train_lab/      # Training labels (.png)
│   ├── test_img/       # Test images (.jpg)
│   └── test_lab/       # Test labels (.png)
├── results/            # Output visualizations
├── utils/
│   └── visualization.py
├── main.py
├── requirements.txt
└── setup.py
```

## Installation

### Prerequisites
- Python 3.7+
- GPU with CUDA support (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/CrackDetection.git
cd CrackDetection

# Create virtual environment
python -m venv detect
source detect/bin/activate  # On Windows: detect\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Inference

```bash
python main.py
```

This will:
1. Load test images from `data/test_img/`
2. Evaluate all four models (U-Net, Linknet, PSPNet, FPN)
3. Output Loss and IoU scores for each model
4. Save visualizations to `results/`

### Data Format

- **Images**: RGB JPG files
- **Labels**: Grayscale PNG masks (binary: crack = white, background = black)
- Images are resized to 384×384 during inference

### Output

Results are saved in `./results/` with three visualization types per image:
- `{model}_comparison_{idx}.png` — True mask vs predicted mask overlay
- `{model}_original_vs_mask_{idx}.png` — Original image alongside binary prediction
- `{model}_overlay_{idx}.png` — Predicted cracks overlaid on original image

## Metrics

Models are evaluated using:
- **Binary Cross-Entropy + Jaccard Loss** (combined loss)
- **IoU Score** (Intersection over Union)

## Configuration

Key parameters in `main.py`:

```python
BACKBONE = 'resnet34'           # Encoder backbone
resize_dim = (384, 384)         # Input image dimensions
activation = 'sigmoid'          # Binary segmentation
encoder_weights = 'imagenet'    # Pre-trained weights
```

## Dependencies

- TensorFlow ≥1.13
- segmentation-models ≥1.0.1
- OpenCV
- NumPy
- Matplotlib
- Keras ≥2.2.0

## License

MIT

## References

- [Segmentation Models](https://github.com/qubvel/segmentation_models) — Qubvel
- AlDajani (2022), Li (2019) — Dataset sources

