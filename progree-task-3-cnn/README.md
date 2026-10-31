# Progree Internship – Task 3
## CIFAR-10 Multi-Class Image Classification Using CNN

### Project Overview
This project implements a Convolutional Neural Network (CNN) for multi-class image classification using the CIFAR-10 dataset. The model is built with TensorFlow and Keras to classify images into 10 different categories.

### Dataset
The CIFAR-10 dataset contains 60,000 color images of size 32×32 pixels, divided into 10 classes.

- Training images: 50,000
- Testing images: 10,000
- Number of classes: 10

**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

### Model Architecture
The CNN includes:

1. Input layer (32×32×3)
2. Data augmentation
3. Rescaling layer
4. Convolutional layers with 32, 64, and 128 filters
5. Max-pooling layers
6. Dropout layers for regularization
7. Flatten layer
8. Fully connected Dense layer (128 units)
9. Output layer with 10 neurons and Softmax activation

Total parameters: 402,986.

### Data Augmentation
The following augmentation techniques were used:
- Random horizontal flipping
- Random rotation
- Random translation

### Training Configuration

| Parameter | Value |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Sparse Categorical Crossentropy |
| Batch size | 64 |
| Epochs | 20 |
| Random seed | 42 |

### Model Performance

| Metric | Result |
|---|---:|
| Best Validation Accuracy | 74.80% |
| Best Validation Loss | 0.7279 |
| Test Accuracy | 73.17% |
| Test Loss | 0.7788 |
| Best Validation Epoch | 20 |

### Output Files

The `outputs/` directory contains:

- `cifar10_cnn.keras` – Trained CNN model
- `metrics.json` – Evaluation metrics
- `sample_predictions.json` – Sample model predictions
- `training_history.csv` – Training history
- `training_validation_accuracy.png` – Training and validation accuracy graph
- `training_validation_loss.png` – Training and validation loss graph

### How to Run

1. Clone or download the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the project:

```bash
python main.py
```

The CIFAR-10 dataset will be downloaded automatically if it is not already cached.

### Technologies Used
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- CIFAR-10 Dataset

### Internship
**Program:** Progree Machine Learning Remote Internship  
**Task:** Task 3 – Multi-Class Image Classification Using CNN