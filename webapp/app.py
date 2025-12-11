"""
AI-Driven Crop Disease Prediction - Flask Web Application
=========================================================
Integrated with Council Prediction System using all 4 trained models.
"""

import os
import sys
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'leaf-disease-prediction-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ============================================================================
# Model Configuration - Updated paths for your trained models
# ============================================================================
MODEL_CONFIG = {
    'model_paths': {
        'ADSD': '../model_files/ADSD/ADSD_fast_model.pth',
        'ECAM': '../model_files/ECAM/ECAM_fast.pth',
        'LGNM': '../model_files/LGNM/LGNM_fast_model.pth',
        'S-ViT_Lite': '../model_files/S-ViT_Lite/S-ViT_Lite_fast_model.pth'
    },
    'model_image_sizes': {
        'ADSD': 160,
        'ECAM': 224,
        'LGNM': 160,
        'S-ViT_Lite': 160
    }
}

# Disease classes (38 classes from the dataset)
DISEASE_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Treatment suggestions for each disease
TREATMENT_SUGGESTIONS = {
    'Apple___Apple_scab': {'risk': 'Medium', 'treatment': 'Apply fungicides like captan, myclobutanil, or mancozeb. Remove infected leaves. Prune for air circulation.', 'prevention': 'Plant scab-resistant varieties. Rake fallen leaves. Apply dormant sprays.'},
    'Apple___Black_rot': {'risk': 'High', 'treatment': 'Prune infected branches and mummified fruits. Apply captan or thiophanate-methyl fungicides.', 'prevention': 'Remove mummified fruit. Maintain orchard sanitation.'},
    'Apple___Cedar_apple_rust': {'risk': 'Medium', 'treatment': 'Apply myclobutanil or propiconazole at pink bud stage every 7-10 days.', 'prevention': 'Remove juniper hosts nearby. Plant rust-resistant varieties.'},
    'Apple___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Continue regular maintenance and monitoring.'},
    'Blueberry___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Maintain acidic soil pH (4.5-5.5). Proper irrigation.'},
    'Cherry_(including_sour)___Powdery_mildew': {'risk': 'Medium', 'treatment': 'Apply sulfur-based fungicides or potassium bicarbonate. Use myclobutanil for severe infections.', 'prevention': 'Improve air circulation. Avoid overhead irrigation. Apply preventive fungicides.'},
    'Cherry_(including_sour)___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Prune for air circulation. Apply dormant oil sprays.'},
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {'risk': 'High', 'treatment': 'Apply azoxystrobin, pyraclostrobin, or propiconazole fungicides at first sign.', 'prevention': 'Plant resistant hybrids. Rotate crops. Till residue.'},
    'Corn_(maize)___Common_rust_': {'risk': 'Medium', 'treatment': 'Apply azoxystrobin or propiconazole when pustules appear.', 'prevention': 'Plant rust-resistant hybrids. Early planting helps.'},
    'Corn_(maize)___Northern_Leaf_Blight': {'risk': 'High', 'treatment': 'Apply strobilurin or triazole fungicides at tasseling stage.', 'prevention': 'Use resistant hybrids. Rotate crops. Till residue.'},
    'Corn_(maize)___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Maintain nutrition and irrigation. Use quality seed.'},
    'Grape___Black_rot': {'risk': 'High', 'treatment': 'Apply mancozeb, myclobutanil, or captan from bud break through veraison.', 'prevention': 'Remove mummified fruit. Prune for air circulation.'},
    'Grape___Esca_(Black_Measles)': {'risk': 'High', 'treatment': 'No cure. Remove infected vines. Apply wound protectants after pruning.', 'prevention': 'Avoid large pruning wounds. Apply wound sealants.'},
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {'risk': 'Medium', 'treatment': 'Apply copper-based fungicides or mancozeb. Remove infected leaves.', 'prevention': 'Good air circulation. Avoid overhead irrigation.'},
    'Grape___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Prune for air circulation. Apply preventive fungicides.'},
    'Orange___Haunglongbing_(Citrus_greening)': {'risk': 'Critical', 'treatment': 'No cure. Remove infected trees. Control Asian citrus psyllid with insecticides.', 'prevention': 'Use certified disease-free stock. Control psyllid populations.'},
    'Peach___Bacterial_spot': {'risk': 'High', 'treatment': 'Apply copper-based bactericides during dormant season. Oxytetracycline during growth.', 'prevention': 'Plant resistant varieties. Avoid overhead irrigation.'},
    'Peach___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Apply dormant sprays. Prune for air circulation.'},
    'Pepper,_bell___Bacterial_spot': {'risk': 'High', 'treatment': 'Apply copper bactericides plus mancozeb. Remove infected plants.', 'prevention': 'Use disease-free seed. Crop rotation 2-3 years.'},
    'Pepper,_bell___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Rotate crops. Use disease-free transplants.'},
    'Potato___Early_blight': {'risk': 'Medium', 'treatment': 'Apply chlorothalonil, mancozeb, or azoxystrobin at first sign.', 'prevention': 'Use certified seed. Crop rotation 3+ years.'},
    'Potato___Late_blight': {'risk': 'Critical', 'treatment': 'Apply chlorothalonil, mancozeb, or cymoxanil immediately. Destroy infected plants.', 'prevention': 'Use resistant varieties. Plant certified seed.'},
    'Potato___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Use certified seed. Rotate crops.'},
    'Raspberry___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Prune old canes. Good drainage. Mulch.'},
    'Soybean___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Rotate crops. Use resistant varieties.'},
    'Squash___Powdery_mildew': {'risk': 'Medium', 'treatment': 'Apply sulfur, potassium bicarbonate, or neem oil.', 'prevention': 'Plant resistant varieties. Space plants for air flow.'},
    'Strawberry___Leaf_scorch': {'risk': 'Medium', 'treatment': 'Apply captan or thiram fungicides. Remove infected leaves.', 'prevention': 'Use disease-free plants. Avoid overhead irrigation.'},
    'Strawberry___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Use disease-free plants. Mulch. Renovate after harvest.'},
    'Tomato___Bacterial_spot': {'risk': 'High', 'treatment': 'Apply copper bactericides plus mancozeb weekly. Remove infected leaves.', 'prevention': 'Use disease-free seed. Rotate crops 2-3 years.'},
    'Tomato___Early_blight': {'risk': 'Medium', 'treatment': 'Apply chlorothalonil, mancozeb, or copper fungicides. Remove lower leaves.', 'prevention': 'Rotate crops 3+ years. Mulch. Stake plants.'},
    'Tomato___Late_blight': {'risk': 'Critical', 'treatment': 'Apply chlorothalonil or mancozeb immediately. Destroy infected plants.', 'prevention': 'Use resistant varieties. Good air circulation.'},
    'Tomato___Leaf_Mold': {'risk': 'Medium', 'treatment': 'Apply chlorothalonil or copper fungicides. Improve ventilation.', 'prevention': 'Maintain humidity below 85%. Good ventilation.'},
    'Tomato___Septoria_leaf_spot': {'risk': 'Medium', 'treatment': 'Apply chlorothalonil or mancozeb weekly. Remove infected leaves.', 'prevention': 'Rotate crops 3+ years. Mulch. Stake plants.'},
    'Tomato___Spider_mites Two-spotted_spider_mite': {'risk': 'Medium', 'treatment': 'Apply miticides or insecticidal soap. Release predatory mites.', 'prevention': 'Monitor regularly. Avoid dusty conditions.'},
    'Tomato___Target_Spot': {'risk': 'Medium', 'treatment': 'Apply chlorothalonil or azoxystrobin. Remove infected leaves.', 'prevention': 'Stake plants. Mulch. Rotate crops.'},
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {'risk': 'High', 'treatment': 'No cure. Remove infected plants. Control whiteflies.', 'prevention': 'Use resistant varieties. Control whiteflies.'},
    'Tomato___Tomato_mosaic_virus': {'risk': 'High', 'treatment': 'No cure. Remove infected plants. Disinfect tools with bleach.', 'prevention': 'Use resistant varieties. Virus-free seed.'},
    'Tomato___healthy': {'risk': 'None', 'treatment': 'No treatment needed. Your plant is healthy!', 'prevention': 'Rotate crops. Stake plants. Water at base.'},
    'default': {'risk': 'Medium', 'treatment': 'Consult agricultural extension service for specific recommendations.', 'prevention': 'Practice crop rotation, proper irrigation, and regular monitoring.'}
}

# ============================================================================
# Model Council - Load all 4 trained models
# ============================================================================

class ModelCouncil:
    """Ensemble of 4 models for council-based prediction."""
    
    def __init__(self):
        self.models = {}
        self.transforms = {}
        self.classes = DISEASE_CLASSES
        self.loaded = False
        
    def get_transform(self, image_size=224):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def load_all_models(self):
        """Load all 4 trained models."""
        if self.loaded:
            return True
            
        print("\n" + "="*60)
        print("Loading trained models...")
        print("="*60)
        
        num_classes = len(DISEASE_CLASSES)
        
        for name, rel_path in MODEL_CONFIG['model_paths'].items():
            # Try multiple path variations
            paths_to_try = [
                rel_path,
                os.path.join('..', rel_path.lstrip('../')),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), rel_path.lstrip('../')),
            ]
            
            model_path = None
            for p in paths_to_try:
                if os.path.exists(p):
                    model_path = p
                    break
            
            if model_path is None:
                print(f"  ✗ {name}: Model file not found")
                continue
            
            try:
                # Load checkpoint
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                
                # Import and create model based on type
                if name == 'ADSD':
                    from train2 import ADSD_FastClassifier
                    model = ADSD_FastClassifier(num_classes=num_classes)
                    state_dict = checkpoint.get('model_state', checkpoint)
                elif name == 'ECAM':
                    from train4 import ECAM_fast
                    model = ECAM_fast(num_classes=num_classes)
                    state_dict = checkpoint.get('best_state', checkpoint.get('final_state', checkpoint))
                elif name == 'LGNM':
                    from train1 import LGNM_FAST
                    model = LGNM_FAST(num_classes=num_classes)
                    state_dict = checkpoint.get('model', checkpoint)
                elif name == 'S-ViT_Lite':
                    from train3 import SViT_Lite_Fast_Modified
                    model = SViT_Lite_Fast_Modified(num_classes=num_classes)
                    state_dict = checkpoint.get('best_model_state', checkpoint.get('final_model_state', checkpoint))
                else:
                    continue
                
                if isinstance(state_dict, dict) and any('weight' in k for k in state_dict.keys()):
                    model.load_state_dict(state_dict)
                elif hasattr(state_dict, 'keys'):
                    # Try to find state dict in nested structure
                    for key in ['model_state_dict', 'state_dict', 'model']:
                        if key in state_dict:
                            model.load_state_dict(state_dict[key])
                            break
                
                model.to(DEVICE)
                model.eval()
                
                self.models[name] = model
                self.transforms[name] = self.get_transform(MODEL_CONFIG['model_image_sizes'][name])
                print(f"  ✓ {name}: Loaded successfully")
                
            except Exception as e:
                print(f"  ✗ {name}: Error loading - {str(e)[:50]}")
        
        self.loaded = len(self.models) > 0
        print(f"\nLoaded {len(self.models)}/4 models")
        print("="*60 + "\n")
        return self.loaded
    
    def predict_single(self, model_name, image):
        """Get prediction from single model."""
        model = self.models.get(model_name)
        if model is None:
            return None
        
        transform = self.transforms.get(model_name)
        
        # Handle LGNM specially (needs CNN-only forward)
        if model_name == 'LGNM':
            try:
                img_tensor = transform(image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    # Use CNN features only
                    cnn_out = model.cnn(img_tensor)
                    cnn_out = model.pool(cnn_out).flatten(1)
                    cnn_feat = model.proj(cnn_out)
                    gnn_feat = torch.zeros(1, 64, device=DEVICE)
                    fused = torch.cat([cnn_feat, gnn_feat], dim=1)
                    fused = F.relu(model.fuse(fused))
                    output = model.cls(fused)
                    probs = F.softmax(output, dim=1)
                    pred_idx = output.argmax(dim=1).item()
                    confidence = probs[0, pred_idx].item()
                return {'prediction': pred_idx, 'confidence': confidence, 'probs': probs[0]}
            except Exception as e:
                print(f"LGNM prediction error: {e}")
                return None
        
        # Standard forward pass for other models
        try:
            img_tensor = transform(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred_idx = output.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
            return {'prediction': pred_idx, 'confidence': confidence, 'probs': probs[0]}
        except Exception as e:
            print(f"{model_name} prediction error: {e}")
            return None
    
    def council_predict(self, image_path):
        """Make council prediction using all models."""
        image = Image.open(image_path).convert('RGB')
        
        # Get predictions from all models
        predictions = {}
        for name in self.models.keys():
            pred = self.predict_single(name, image)
            if pred:
                predictions[name] = pred
        
        if not predictions:
            return {'success': False, 'error': 'No models could make predictions'}
        
        # Confidence-weighted voting
        num_classes = len(DISEASE_CLASSES)
        weighted_probs = torch.zeros(num_classes, device=DEVICE)
        total_weight = 0
        
        model_weights = {'ADSD': 1.0, 'ECAM': 1.1, 'LGNM': 0.9, 'S-ViT_Lite': 1.0}
        
        for name, pred in predictions.items():
            weight = model_weights.get(name, 1.0) * pred['confidence']
            weighted_probs += weight * pred['probs']
            total_weight += weight
        
        if total_weight > 0:
            weighted_probs /= total_weight
        
        final_pred_idx = weighted_probs.argmax().item()
        final_confidence = weighted_probs[final_pred_idx].item()
        
        # Calculate agreement
        agreeing = sum(1 for p in predictions.values() if p['prediction'] == final_pred_idx)
        agreement = agreeing / len(predictions) if predictions else 0
        
        # Get disease info
        disease_name = DISEASE_CLASSES[final_pred_idx]
        disease_info = TREATMENT_SUGGESTIONS.get(disease_name, TREATMENT_SUGGESTIONS['default'])
        
        # Parse disease name
        parts = disease_name.split('___')
        plant = parts[0].replace('_', ' ').replace('(', '').replace(')', '').strip()
        condition = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'
        is_healthy = 'healthy' in disease_name.lower()
        
        # Build individual model results
        model_results = {}
        for name, pred in predictions.items():
            pred_disease = DISEASE_CLASSES[pred['prediction']]
            pred_parts = pred_disease.split('___')
            pred_condition = pred_parts[1].replace('_', ' ') if len(pred_parts) > 1 else pred_disease
            model_results[name] = {
                'condition': pred_condition,
                'disease_name': pred_disease,
                'confidence': pred['confidence'] * 100,
                'agrees': pred['prediction'] == final_pred_idx
            }
        
        return {
            'success': True,
            'plant': plant,
            'condition': condition,
            'disease_name': disease_name,
            'confidence': final_confidence * 100,
            'agreement': agreement * 100,
            'is_healthy': is_healthy,
            'risk_level': 'None' if is_healthy else disease_info['risk'],
            'treatment': disease_info['treatment'],
            'prevention': disease_info['prevention'],
            'models': model_results,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# Global council instance
council = ModelCouncil()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Store recent predictions for dashboard
recent_predictions = []

# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load models if not loaded
            if not council.loaded:
                council.load_all_models()
            
            # Make council prediction
            result = council.council_predict(filepath)
            result['image_path'] = f"/static/uploads/{filename}"
            
            # Store for dashboard
            if result.get('success'):
                recent_predictions.insert(0, result)
                if len(recent_predictions) > 10:
                    recent_predictions.pop()
            
            return jsonify(result)
        
        return jsonify({'success': False, 'error': 'Invalid file type'})
    
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', predictions=recent_predictions)

@app.route('/api/predictions')
def api_predictions():
    return jsonify(recent_predictions)

@app.route('/api/stats')
def api_stats():
    if not recent_predictions:
        return jsonify({
            'total_predictions': 0, 'healthy_count': 0, 'diseased_count': 0,
            'avg_confidence': 0, 'most_common_disease': 'N/A'
        })
    
    healthy = sum(1 for p in recent_predictions if p.get('is_healthy', False))
    avg_conf = sum(p.get('confidence', 0) for p in recent_predictions) / len(recent_predictions)
    diseases = [p.get('condition', 'Unknown') for p in recent_predictions if not p.get('is_healthy', True)]
    most_common = max(set(diseases), key=diseases.count) if diseases else 'N/A'
    
    return jsonify({
        'total_predictions': len(recent_predictions),
        'healthy_count': healthy,
        'diseased_count': len(recent_predictions) - healthy,
        'avg_confidence': round(avg_conf, 2),
        'most_common_disease': most_common
    })

# ============================================================================
# Run Application
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌿 AI-Driven Crop Disease Prediction System")
    print("   Council Prediction with 4 Deep Learning Models")
    print("="*60)
    print(f"Device: {DEVICE}")
    
    # Load models at startup
    council.load_all_models()
    
    print("\nStarting web server...")
    print("Access the application at: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
