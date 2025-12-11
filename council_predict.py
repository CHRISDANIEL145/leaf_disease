"""
Council Prediction System
=========================
Ensemble prediction using all 4 trained models with multiple voting strategies:
- Majority vote
- Confidence-weighted vote
- Causal-explanation consensus (ECAM-guided)

Produces:
- Final prediction
- Confidence score
- Causal explanation
- Model-wise agreement score
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuration - Updated paths for your trained models
CONFIG = {
    'model_paths': {
        'ADSD': 'model_files/ADSD/ADSD_fast_model.pth',
        'ECAM': 'model_files/ECAM/ECAM_fast.pth',
        'LGNM': 'model_files/LGNM/LGNM_fast_model.pth',
        'S-ViT_Lite': 'model_files/S-ViT_Lite/S-ViT_Lite_fast_model.pth'
    },
    'model_image_sizes': {
        'ADSD': 160,
        'ECAM': 224,
        'LGNM': 160,
        'S-ViT_Lite': 160
    },
    'visualization_dir': 'council_visualization',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Create visualization directory
os.makedirs(CONFIG['visualization_dir'], exist_ok=True)

# Device
DEVICE = torch.device(CONFIG['device'])
print(f"Council Predict using device: {DEVICE}")



# ============================================================================
# Model Loading
# ============================================================================

def get_transform(image_size=224):
    """Get inference transform for specified image size."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_model(model_name, model_path, num_classes, device):
    """
    Load a trained model.
    
    Args:
        model_name: One of 'ADSD', 'ECAM', 'LGNM', 'S-ViT_Lite'
        model_path: Path to the model checkpoint
        num_classes: Number of output classes
        device: Device to load the model on
    
    Returns:
        model, classes tuple
    """
    if not os.path.exists(model_path):
        print(f"  Warning: Model {model_name} not found at {model_path}")
        return None, None
    
    # Import the correct model class based on model name
    try:
        if model_name == 'ADSD':
            from train2 import ADSD_FastClassifier
            model = ADSD_FastClassifier(num_classes=num_classes)
        elif model_name == 'ECAM':
            from train4 import ECAM_fast
            model = ECAM_fast(num_classes=num_classes)
        elif model_name == 'LGNM':
            from train1 import LGNM_FAST
            model = LGNM_FAST(num_classes=num_classes)
        elif model_name == 'S-ViT_Lite':
            from train3 import SViT_Lite_Fast_Modified
            model = SViT_Lite_Fast_Modified(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except ImportError as e:
        print(f"  Error importing {model_name}: {e}")
        return None, None
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        classes = None
        state_dict = None
        
        # Handle different checkpoint formats based on model
        if isinstance(checkpoint, dict):
            # Get classes if available
            classes = checkpoint.get('classes', None)
            
            # Try different state dict keys based on model training script format
            if model_name == 'ADSD':
                # ADSD uses 'model_state'
                state_dict = checkpoint.get('model_state', None)
            elif model_name == 'ECAM':
                # ECAM uses 'best_state' or 'final_state'
                state_dict = checkpoint.get('best_state', checkpoint.get('final_state', None))
            elif model_name == 'LGNM':
                # LGNM uses 'model'
                state_dict = checkpoint.get('model', None)
            elif model_name == 'S-ViT_Lite':
                # S-ViT_Lite uses 'best_model_state' or 'final_model_state'
                state_dict = checkpoint.get('best_model_state', checkpoint.get('final_model_state', None))
            
            # Fallback to common keys
            if state_dict is None:
                for key in ['model_state_dict', 'state_dict', 'model', 'best_state']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
            
            # If still None, try the checkpoint itself as state dict
            if state_dict is None:
                # Check if checkpoint keys look like model parameters
                if any('weight' in k or 'bias' in k for k in checkpoint.keys()):
                    state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        if state_dict is None:
            print(f"  Error: Could not find state dict in checkpoint for {model_name}")
            return None, None
        
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, classes
        
    except Exception as e:
        print(f"  Error loading {model_name}: {e}")
        return None, None


class LGNMWrapper(nn.Module):
    """
    Wrapper for LGNM model to handle standard forward pass without graph.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    
    def forward(self, x):
        # For council prediction, use only CNN features
        batch_size = x.size(0)
        device = x.device
        
        # Extract CNN features
        cnn_out = self.base.cnn(x)
        cnn_out = self.base.pool(cnn_out).flatten(1)
        cnn_feat = self.base.proj(cnn_out)
        
        # Use zeros for GNN features (simplified for ensemble)
        gnn_feat = torch.zeros(batch_size, 64, device=device)
        
        # Fuse and classify
        fused = torch.cat([cnn_feat, gnn_feat], dim=1)
        fused = F.relu(self.base.fuse(fused))
        return self.base.cls(fused)


class ModelCouncil:
    """
    Ensemble of 4 models for council-based prediction.
    Implements multiple voting strategies.
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.models = {}
        self.transforms = {}
        self.classes = None
        
    def load_all_models(self, num_classes=38):
        """Load all 4 models."""
        print("Loading models...")
        
        for name, path in CONFIG['model_paths'].items():
            model, classes = load_model(name, path, num_classes, self.device)
            if model is not None:
                # Wrap LGNM model
                if name == 'LGNM':
                    model = LGNMWrapper(model)
                
                self.models[name] = model
                self.transforms[name] = get_transform(CONFIG['model_image_sizes'][name])
                
                if self.classes is None and classes is not None:
                    self.classes = classes
                print(f"  ✓ Loaded {name}")
            else:
                print(f"  ✗ Skipped {name} (not found or error)")
        
        print(f"\nLoaded {len(self.models)}/{len(CONFIG['model_paths'])} models")
        return len(self.models) > 0
    
    def predict_single_model(self, model_name, image):
        """
        Get prediction from a single model.
        
        Args:
            model_name: Name of the model
            image: PIL Image
        
        Returns:
            Prediction dictionary or None
        """
        model = self.models.get(model_name)
        if model is None:
            return None
        
        # Get appropriate transform
        transform = self.transforms.get(model_name, get_transform(224))
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
        
        return {
            'prediction': pred_idx,
            'confidence': confidence,
            'probabilities': probs.cpu().numpy()[0]
        }
    
    def majority_vote(self, predictions):
        """Simple majority voting."""
        votes = [p['prediction'] for p in predictions.values() if p is not None]
        if not votes:
            return None, 0
        
        vote_counts = Counter(votes)
        winner, count = vote_counts.most_common(1)[0]
        agreement = count / len(votes)
        
        return winner, agreement
    
    def confidence_weighted_vote(self, predictions):
        """Confidence-weighted voting."""
        if not predictions:
            return None, 0, {}
        
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        if not valid_preds:
            return None, 0, {}
        
        num_classes = len(list(valid_preds.values())[0]['probabilities'])
        weighted_probs = np.zeros(num_classes)
        total_weight = 0
        
        # Model weights (can be tuned based on individual model performance)
        model_weights = {
            'ADSD': 1.0,
            'ECAM': 1.1,       # Slight boost for causal model
            'LGNM': 0.9,       # Slightly lower due to missing graph features
            'S-ViT_Lite': 1.0
        }
        
        for name, pred in valid_preds.items():
            weight = model_weights.get(name, 1.0) * pred['confidence']
            weighted_probs += weight * pred['probabilities']
            total_weight += weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        winner = np.argmax(weighted_probs)
        confidence = weighted_probs[winner]
        
        return winner, confidence, weighted_probs
    
    def causal_consensus(self, predictions, image):
        """
        Causal-explanation consensus using ECAM model.
        Weights predictions based on causal attribution agreement.
        """
        ecam_model = self.models.get('ECAM')
        if ecam_model is None:
            return self.confidence_weighted_vote(predictions)
        
        # Get ECAM causal explanation
        transform = self.transforms.get('ECAM', get_transform(224))
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        causal_explanation = None
        causal_confidence = 0.5
        
        try:
            with torch.no_grad():
                if hasattr(ecam_model, 'get_causal_explanation'):
                    explanation = ecam_model.get_causal_explanation(image_tensor)
                    causal_explanation = explanation
                    if 'strength' in explanation:
                        causal_strength = explanation['strength'].cpu().numpy()
                        causal_confidence = float(causal_strength.max())
        except Exception as e:
            print(f"  Warning: Could not get causal explanation: {e}")
        
        # Weight predictions by causal confidence
        valid_preds = {k: v for k, v in predictions.items() if v is not None}
        if not valid_preds:
            return None, 0, {}, causal_explanation
        
        num_classes = len(list(valid_preds.values())[0]['probabilities'])
        weighted_probs = np.zeros(num_classes)
        total_weight = 0
        
        for name, pred in valid_preds.items():
            # ECAM gets higher weight when causal confidence is high
            if name == 'ECAM':
                weight = pred['confidence'] * (1 + causal_confidence)
            else:
                weight = pred['confidence']
            
            weighted_probs += weight * pred['probabilities']
            total_weight += weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        winner = np.argmax(weighted_probs)
        confidence = weighted_probs[winner]
        
        return winner, confidence, weighted_probs, causal_explanation

    def predict(self, image_path, voting_strategy='confidence_weighted'):
        """
        Make council prediction on an image.
        
        Args:
            image_path: Path to image file
            voting_strategy: 'majority', 'confidence_weighted', or 'causal_consensus'
        
        Returns:
            Dictionary with prediction results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Get predictions from all models
        predictions = {}
        for name in self.models.keys():
            pred = self.predict_single_model(name, image)
            predictions[name] = pred
        
        # Apply voting strategy
        causal_explanation = None
        
        if voting_strategy == 'majority':
            final_pred, agreement = self.majority_vote(predictions)
            confidence = agreement
        elif voting_strategy == 'confidence_weighted':
            final_pred, confidence, _ = self.confidence_weighted_vote(predictions)
            agreement = self.calculate_agreement(predictions, final_pred)
        else:  # causal_consensus
            result = self.causal_consensus(predictions, image)
            if len(result) == 4:
                final_pred, confidence, _, causal_explanation = result
            else:
                final_pred, confidence, _ = result
            agreement = self.calculate_agreement(predictions, final_pred)
        
        # Handle case where no valid predictions
        if final_pred is None:
            return {
                'error': 'No valid predictions from any model',
                'individual_predictions': {}
            }
        
        # Get class name
        class_name = self.classes[final_pred] if self.classes else str(final_pred)
        
        # Build result
        result = {
            'final_prediction': {
                'class': class_name,
                'class_idx': int(final_pred),
                'confidence': float(confidence)
            },
            'model_agreement': float(agreement),
            'individual_predictions': {},
            'voting_strategy': voting_strategy
        }
        
        # Add individual model predictions
        for name, pred in predictions.items():
            if pred is not None:
                pred_class = self.classes[pred['prediction']] if self.classes else str(pred['prediction'])
                result['individual_predictions'][name] = {
                    'class': pred_class,
                    'class_idx': pred['prediction'],
                    'confidence': float(pred['confidence']),
                    'agrees_with_final': bool(pred['prediction'] == final_pred)
                }
        
        # Add causal explanation if available
        if causal_explanation is not None:
            result['causal_explanation_available'] = True
        
        return result
    
    def calculate_agreement(self, predictions, final_pred):
        """Calculate agreement score among models."""
        if final_pred is None:
            return 0
        agreeing = sum(1 for p in predictions.values() if p is not None and p['prediction'] == final_pred)
        total = sum(1 for p in predictions.values() if p is not None)
        return agreeing / total if total > 0 else 0


# ============================================================================
# Visualization
# ============================================================================

def visualize_council_prediction(result, image_path, save_path):
    """Visualize council prediction results."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    image = Image.open(image_path)
    axes[0].imshow(image)
    axes[0].set_title('Input Image', fontsize=12)
    axes[0].axis('off')
    
    # Final prediction
    axes[1].text(0.5, 0.7, f"Prediction:", ha='center', va='center', fontsize=12,
                transform=axes[1].transAxes)
    axes[1].text(0.5, 0.55, f"{result['final_prediction']['class']}", 
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=axes[1].transAxes, wrap=True)
    axes[1].text(0.5, 0.35, f"Confidence: {result['final_prediction']['confidence']:.1%}", 
                ha='center', va='center', fontsize=11,
                transform=axes[1].transAxes)
    axes[1].text(0.5, 0.2, f"Agreement: {result['model_agreement']:.1%}", 
                ha='center', va='center', fontsize=11,
                transform=axes[1].transAxes)
    axes[1].set_title('Council Decision', fontsize=12)
    axes[1].axis('off')
    
    # Model votes
    model_names = list(result['individual_predictions'].keys())
    confidences = [result['individual_predictions'][m]['confidence'] for m in model_names]
    colors = ['#2ecc71' if result['individual_predictions'][m]['agrees_with_final'] else '#e74c3c' 
              for m in model_names]
    
    bars = axes[2].barh(model_names, confidences, color=colors)
    axes[2].set_xlim(0, 1)
    axes[2].set_title('Model Confidences', fontsize=12)
    axes[2].set_xlabel('Confidence')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#2ecc71', label='Agrees'),
                      Patch(facecolor='#e74c3c', label='Disagrees')]
    axes[2].legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")


# ============================================================================
# Main Functions
# ============================================================================

def council_predict(image_path, voting_strategy='confidence_weighted', visualize=True):
    """
    Main function for council prediction.
    
    Args:
        image_path: Path to input image
        voting_strategy: 'majority', 'confidence_weighted', or 'causal_consensus'
        visualize: Whether to save visualization
    
    Returns:
        Prediction result dictionary
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    council = ModelCouncil(CONFIG['device'])
    
    if not council.load_all_models():
        print("Error: No models loaded. Please train models first.")
        return None
    
    result = council.predict(image_path, voting_strategy)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return result
    
    # Print results
    print("\n" + "=" * 60)
    print("COUNCIL PREDICTION RESULT")
    print("=" * 60)
    print(f"\nFinal Prediction: {result['final_prediction']['class']}")
    print(f"Confidence: {result['final_prediction']['confidence']:.1%}")
    print(f"Model Agreement: {result['model_agreement']:.1%}")
    print(f"Voting Strategy: {result['voting_strategy']}")
    
    print("\nIndividual Model Predictions:")
    for name, pred in result['individual_predictions'].items():
        agree_str = "✓" if pred['agrees_with_final'] else "✗"
        print(f"  {name:12s}: {pred['class']:35s} ({pred['confidence']:.1%}) {agree_str}")
    
    # Visualize
    if visualize:
        # Save to council_visualization folder
        image_basename = os.path.basename(image_path)
        image_name = image_basename.rsplit('.', 1)[0]
        save_path = os.path.join(CONFIG['visualization_dir'], f"{image_name}_council_prediction.png")
        try:
            visualize_council_prediction(result, image_path, save_path)
        except Exception as e:
            print(f"Visualization error: {e}")
    
    return result


def batch_predict(image_dir, output_file='council_predictions.json', voting_strategy='confidence_weighted'):
    """Batch prediction on a directory of images."""
    if not os.path.exists(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return None
    
    council = ModelCouncil(CONFIG['device'])
    
    if not council.load_all_models():
        print("Error: No models loaded.")
        return None
    
    results = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}
    
    image_files = [f for f in os.listdir(image_dir) 
                   if any(f.endswith(ext) for ext in image_extensions)]
    
    print(f"\nProcessing {len(image_files)} images...")
    
    for i, filename in enumerate(image_files):
        image_path = os.path.join(image_dir, filename)
        print(f"[{i+1}/{len(image_files)}] {filename}")
        
        try:
            result = council.predict(image_path, voting_strategy)
            result['image_file'] = filename
            results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            results.append({'image_file': filename, 'error': str(e)})
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    successful = sum(1 for r in results if 'error' not in r)
    print(f"\n{'=' * 60}")
    print(f"Batch Prediction Complete")
    print(f"{'=' * 60}")
    print(f"Processed: {len(results)} images")
    print(f"Successful: {successful}")
    print(f"Results saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Council Prediction System for Plant Disease')
    parser.add_argument('image_path', type=str, help='Path to image file or directory for batch processing')
    parser.add_argument('--strategy', type=str, default='confidence_weighted',
                        choices=['majority', 'confidence_weighted', 'causal_consensus'],
                        help='Voting strategy (default: confidence_weighted)')
    parser.add_argument('--batch', action='store_true', help='Process directory of images')
    parser.add_argument('--output', type=str, default='council_predictions.json',
                        help='Output file for batch predictions')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_predict(args.image_path, args.output, args.strategy)
    else:
        council_predict(args.image_path, args.strategy, visualize=not args.no_viz)
