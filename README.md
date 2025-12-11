# Multi-Model Crop Disease Prediction System

A comprehensive deep learning system for plant disease classification using 5 unique architectures with RLHF integration and council-based ensemble prediction.

##Application link :"https://huggingface.co/spaces/Danielchris145/leaf-disease-prediction"
## 🌿 Overview

This system implements 5 cutting-edge deep learning models for plant disease detection:

| Model | File | Technology | Key Innovation |
|-------|------|------------|----------------|
| LGNM | `train1.py` | Leaf-Graph Neural Morphing | GNN-based vein structure analysis |
| ADSD | `train2.py` | Adaptive Disease Signature Diffusion | Diffusion-based augmentation |
| S-ViT Lite | `train3.py` | Spectral-Aware Vision Transformer | RGB + pseudo-NIR fusion |
| ECAM | `train5.py` | Explainable Causal Attribution | Counterfactual causal reasoning |

## 📁 Project Structure

```
├── train1.py              # LGNM Model
├── train2.py              # ADSD Model
├── train3.py              # S-ViT Lite Model
├── train4.py              # FALC-Fed Model
├── train5.py              # ECAM Model
├── council_predict.py     # Ensemble prediction system
├── rlhf_active_learning.py # RLHF + Active Learning utilities
├── requirements.txt       # Dependencies
├── model_files/           # Saved models directory
│   ├── lgnm/
│   ├── adsd/
│   ├── svit_lite/
│   ├── falc_fed/
│   └── ecam/
└── dataset/               # Dataset directory
    └── New Plant Diseases Dataset(Augmented)/
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Individual Models
```bash
# Train each model separately
python train1.py  # LGNM
python train2.py  # ADSD
python train3.py  # S-ViT Lite
python train4.py  # FALC-Fed
python train5.py  # ECAM
```

### 3. Council Prediction
```bash
python council_predict.py path/to/leaf_image.jpg
```

### 4. RLHF + Active Learning
```bash
python rlhf_active_learning.py
```

## 🎯 Achieving 95%+ Accuracy

Each model is designed to achieve >95% accuracy through:

1. **Data Augmentation**: Extensive augmentation including rotation, flipping, color jitter
2. **Class Balancing**: Weighted sampling to handle class imbalance
3. **Label Smoothing**: 0.1 label smoothing for better generalization
4. **Learning Rate Scheduling**: Cosine annealing with warm restarts
5. **Early Stopping**: Patience-based early stopping to prevent overfitting
6. **Mixed Precision Training**: FP16 training for faster convergence
7. **Regularization**: Dropout + weight decay

## 📊 Model Architectures

### 1. LGNM (Leaf-Graph Neural Morphing)
- Extracts leaf vein structure using edge detection
- Converts veins to graph representation
- Uses GAT (Graph Attention Network) for reasoning
- Fuses CNN features with GNN features

### 2. ADSD (Adaptive Disease Signature Diffusion)
- Implements DDPM for disease pattern generation
- Learns disease-specific signatures
- Augments training with diffusion-generated samples
- EfficientNet backbone with signature attention

### 3. S-ViT Lite (Spectral-Aware Vision Transformer)
- Generates pseudo-NIR channel from RGB
- Spectral attention mechanism for band fusion
- Lightweight ViT architecture
- NDVI-like vegetation index integration

### 4. ECAM (Explainable Causal Attribution Module)
- ResNet50 backbone with causal modules
- Counterfactual intervention for explanation
- Graph-based causal reasoning
- Interpretable disease region attribution

## 🗳️ Council Voting System

The council system combines all 5 models using:

1. **Majority Vote**: Simple democratic voting
2. **Confidence-Weighted Vote**: Weights by prediction confidence
3. **Causal Consensus**: ECAM-guided weighting based on causal attribution

## 🤖 RLHF Components

### Reinforcement Learning with Human Feedback
- Reward model trained on human correctness ratings
- Policy gradient optimization with KL constraint

### Sequence Tutor
- Sequence-level RL for confidence calibration
- Entropy-based uncertainty reduction

### Pluralistic Alignment
- 4 preference styles: Strict, Tolerant, Confidence-Biased, Conservative
- Style-conditioned prediction adjustment

### Variational Preference Learning
- VAE-based preference encoding
- Generalizes to unseen disease types

### Active Learning
- Uncertainty-based sample selection
- Incremental model improvement
- Simulated human labeling interface

## 📈 Expected Results

| Model | Train Acc | Val Acc | Test Acc |
|-------|-----------|---------|----------|
| LGNM | >95% | >95% | >94% |
| ADSD | >96% | >95% | >94% |
| S-ViT Lite | >96% | >95% | >95% |
| ECAM | >96% | >95% | >94% |
| **Council** | - | **>97%** | **>96%** |

## 🔧 Configuration

Each model has a CONFIG dictionary at the top of its file. Key parameters:

```python
CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'dropout': 0.3,
    'patience': 10,  # Early stopping
    'device': 'cuda'  # or 'cpu'
}
```

## 📝 Output Files

Each model saves:
- `best_model.pth`: Best validation checkpoint
- `final_model.pth`: Final model with metadata
- `training_curves.png`: Loss/accuracy plots
- `confusion_matrix.png`: Per-class performance

## 🔬 Inference Example

```python
from council_predict import council_predict

# Single image prediction
result = council_predict('leaf_image.jpg', voting_strategy='causal_consensus')

print(f"Prediction: {result['final_prediction']['class']}")
print(f"Confidence: {result['final_prediction']['confidence']:.2%}")
print(f"Agreement: {result['model_agreement']:.2%}")
```

## ⚠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 8GB+ VRAM (for training)

## 📚 Citation

If you use this system, please cite the relevant papers for each technology:
- Graph Neural Networks (GNN)
- Denoising Diffusion Probabilistic Models (DDPM)
- Vision Transformers (ViT)
- Federated Learning
- Causal Inference in Deep Learning
