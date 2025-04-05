# BirdCLEF-2025 Inference Pipeline

This repository contains code for the BirdCLEF-2025 bird sound recognition challenge. The pipeline processes audio recordings of bird sounds, extracts mel spectrogram features, and uses deep learning models to identify bird species.

## Features

- **High-performance Audio Processing**: Efficiently processes large audio files by segmenting them into smaller chunks
- **Advanced Feature Extraction**: Converts audio segments to mel spectrograms with configurable parameters
- **Deep Learning Models**: Uses EfficientNet architecture from the timm library for bird species classification
- **Test-Time Augmentation (TTA)**: Optional augmentation strategies to improve prediction robustness
- **Ensemble Support**: Ability to combine predictions from multiple models
- **Visualization Tools**: Comprehensive tools for data exploration and prediction visualization
- **Temporal Smoothing**: Post-processing to ensure temporal consistency in predictions

## Requirements

- Python 3.8+
- PyTorch 1.8+
- librosa
- numpy
- pandas
- matplotlib
- tqdm
- timm
- OpenCV (cv2)

## Quick Start

1. Configure the paths in the `CFG` class to match your environment
2. Run the main script to execute the complete pipeline:

```python
python birdclef2025_inference.py
```

## Configuration

The `CFG` class contains all configurable parameters:

```python
class CFG:
    # Paths
    test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    model_path = '/kaggle/input/birdclef-2025-efficientnet-b0'
    
    # Audio parameters
    FS = 32000  # Sampling rate
    WINDOW_SIZE = 5  # Segment size in seconds
    
    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 64
    N_MELS = 136
    FMIN = 20
    FMAX = 16000
    TARGET_SHAPE = (256, 256)
    
    # Model parameters
    model_name = 'efficientnet_b0'
    in_channels = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Inference parameters
    batch_size = 16
    use_tta = False
    tta_count = 3
    threshold = 0.7
    
    # Model selection
    use_specific_folds = False
    folds = [0, 1]
    
    # Debug options
    debug = False
    debug_count = 3
```

## Data Visualization

The code includes comprehensive data visualization capabilities:

```python
# Initialize the dataset previewer
preview_dataset(cfg)

# Initialize the pipeline
pipeline = BirdCLEF2025Pipeline(cfg)

# Visualize predictions for a single file
test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
if test_files:
    pipeline.predict_file(str(test_files[0]))
```

## Pipeline Components

### 1. Audio Processing

The `AudioProcessor` class handles audio data processing:
- Loading audio files
- Converting to mel spectrograms
- Applying test-time augmentation

### 2. Model Management

The `ModelManager` class handles model operations:
- Finding model files
- Loading models
- Running inference
- Ensemble prediction

### 3. Main Pipeline

The `BirdCLEF2025Pipeline` class coordinates the entire process:
- Loading taxonomy data
- Processing audio files
- Making predictions
- Creating and smoothing the submission file

### 4. Data Visualization

The `DataVisualizer` class provides visualization tools:
- Audio waveform plotting
- Mel spectrogram visualization
- Prediction timeline plots
- Species detection heatmaps

## Example Usage

### 1. Preview Dataset

```python
preview_dataset(cfg)
```

### 2. Run Prediction on a Single File

```python
pipeline = BirdCLEF2025Pipeline(cfg)
audio_path = '/path/to/audio.ogg'
pipeline.predict_file(audio_path)
```

### 3. Run Full Pipeline

```python
pipeline = BirdCLEF2025Pipeline(cfg)
pipeline.run()
```

## Customization

### Changing Model Architecture

To use a different model architecture, modify the `model_name` parameter in `CFG`:

```python
cfg = CFG()
cfg.model_name = 'resnet50'  # Use ResNet-50 instead of EfficientNet
```

### Enabling Test-Time Augmentation

To enable test-time augmentation for improved prediction robustness:

```python
cfg = CFG()
cfg.use_tta = True
cfg.tta_count = 3  # Number of augmentation variations
```

### Ensemble Multiple Models

The pipeline automatically uses all model files found in the model directory. To use specific model folds:

```python
cfg = CFG()
cfg.use_specific_folds = True
cfg.folds = [0, 2, 4]  # Use models from folds 0, 2, and 4
```

## Output

The pipeline generates a CSV submission file with predictions for each audio segment and species. Additionally, when using visualization functions, it produces:

- Audio waveform plots
- Mel spectrogram visualizations
- Species detection probability plots
- Species detection heatmaps

## License

[MIT License](LICENSE)

## Acknowledgments

This code is designed for the BirdCLEF-2025 competition organized by Kaggle.
