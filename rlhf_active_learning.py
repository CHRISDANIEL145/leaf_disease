"""
RLHF + Active Learning Utility
==============================
Implements:
1. RLHF (Reinforcement Learning with Human Feedback)
2. Sequence Tutor (RL Fine-Tuning)
3. Pluralistic Alignment
4. Variational Preference Learning
5. Active Learning Loop

This module can be integrated with any of the 4 models.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.distributions import Normal, kl_divergence
import numpy as np
from collections import defaultdict
import random
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'reward_model_dim': 256,
    'preference_latent_dim': 64,
    'num_preference_styles': 4,
    'active_learning_budget': 100,
    'uncertainty_threshold': 0.3,
    'rl_learning_rate': 1e-5,
    'reward_learning_rate': 1e-4,
    'kl_weight': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


# ============================================================================
# 1. RLHF - Reinforcement Learning with Human Feedback
# ============================================================================

class RewardModel(nn.Module):
    """
    Reward model that learns from human feedback.
    Predicts reward scores for model predictions.
    """
    
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features):
        return self.network(features)


class HumanFeedbackCollector:
    """
    Collects and manages human feedback for RLHF.
    In production, this would interface with a labeling system.
    """
    
    def __init__(self):
        self.feedback_buffer = []
        self.preference_pairs = []
    
    def add_feedback(self, prediction, true_label, is_correct, confidence_appropriate=True):
        """Add human feedback for a prediction."""
        self.feedback_buffer.append({
            'prediction': prediction,
            'true_label': true_label,
            'is_correct': is_correct,
            'confidence_appropriate': confidence_appropriate,
            'reward': self._compute_reward(is_correct, confidence_appropriate)
        })
    
    def add_preference_pair(self, pred_a, pred_b, preferred):
        """Add preference comparison between two predictions."""
        self.preference_pairs.append({
            'pred_a': pred_a,
            'pred_b': pred_b,
            'preferred': preferred  # 'a', 'b', or 'equal'
        })
    
    def _compute_reward(self, is_correct, confidence_appropriate):
        """Compute reward from feedback."""
        reward = 1.0 if is_correct else -1.0
        if confidence_appropriate:
            reward += 0.2
        else:
            reward -= 0.2
        return reward
    
    def get_training_data(self):
        """Get feedback data for reward model training."""
        return self.feedback_buffer
    
    def simulate_feedback(self, predictions, true_labels, confidences):
        """
        Simulate human feedback for training.
        In production, replace with actual human labeling.
        """
        for pred, true_label, conf in zip(predictions, true_labels, confidences):
            is_correct = (pred == true_label)
            
            # Simulate confidence appropriateness judgment
            if is_correct and conf > 0.7:
                confidence_appropriate = True
            elif not is_correct and conf < 0.5:
                confidence_appropriate = True
            else:
                confidence_appropriate = random.random() > 0.5
            
            self.add_feedback(pred, true_label, is_correct, confidence_appropriate)


class RLHFTrainer:
    """
    RLHF trainer that fine-tunes models using human feedback.
    """
    
    def __init__(self, model, reward_model, device='cuda'):
        self.model = model
        self.reward_model = reward_model
        self.device = device
        self.feedback_collector = HumanFeedbackCollector()
    
    def train_reward_model(self, feature_extractor, dataloader, epochs=5):
        """Train reward model on collected feedback."""
        optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=CONFIG['reward_learning_rate'])
        
        feedback_data = self.feedback_collector.get_training_data()
        if len(feedback_data) < 10:
            print("Not enough feedback data for reward model training")
            return
        
        self.reward_model.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in dataloader:
                images, labels = batch
                images = images.to(self.device)
                
                with torch.no_grad():
                    features = feature_extractor(images)
                
                # Get rewards from feedback
                rewards = torch.tensor([
                    feedback_data[i % len(feedback_data)]['reward'] 
                    for i in range(len(images))
                ], device=self.device, dtype=torch.float32)
                
                predicted_rewards = self.reward_model(features).squeeze()
                loss = F.mse_loss(predicted_rewards, rewards)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Reward Model Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    def rl_finetune(self, dataloader, epochs=3, kl_weight=0.1):
        """Fine-tune model using RL with reward model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['rl_learning_rate'])
        
        # Store original model for KL constraint
        original_model = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        self.model.train()
        self.reward_model.eval()
        
        for epoch in range(epochs):
            total_reward = 0
            total_kl = 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get model predictions
                logits = self.model(images)
                probs = F.softmax(logits, dim=1)
                
                # Sample actions (predictions)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                
                # Get rewards
                with torch.no_grad():
                    if hasattr(self.model, 'forward') and 'return_features' in str(self.model.forward.__code__.co_varnames):
                        features = self.model(images, return_features=True)
                    else:
                        features = probs  # Use probabilities as features
                    rewards = self.reward_model(features).squeeze()
                
                # Policy gradient loss
                pg_loss = -(log_probs * rewards).mean()
                
                # KL divergence constraint
                with torch.no_grad():
                    self.model.load_state_dict(original_model)
                    original_logits = self.model(images)
                    original_probs = F.softmax(original_logits, dim=1)
                
                kl_loss = F.kl_div(probs.log(), original_probs, reduction='batchmean')
                
                # Combined loss
                loss = pg_loss + kl_weight * kl_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_reward += rewards.mean().item()
                total_kl += kl_loss.item()
            
            print(f"RL Epoch {epoch+1}, Avg Reward: {total_reward/len(dataloader):.4f}, KL: {total_kl/len(dataloader):.4f}")


# ============================================================================
# 2. Sequence Tutor - RL Fine-Tuning for Classification
# ============================================================================

class SequenceTutor:
    """
    Sequence-level RL fine-tuning to optimize classification confidence
    and reduce uncertainty.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.baseline = None
    
    def compute_sequence_reward(self, logits, labels, predictions):
        """
        Compute sequence-level reward based on:
        - Correctness
        - Confidence calibration
        - Uncertainty reduction
        """
        probs = F.softmax(logits, dim=1)
        confidences = probs.max(dim=1)[0]
        
        # Correctness reward
        correct = (predictions == labels).float()
        correctness_reward = correct * 2 - 1  # +1 for correct, -1 for incorrect
        
        # Confidence calibration reward
        # High confidence should correlate with correctness
        calibration_reward = torch.where(
            correct.bool(),
            confidences,  # Reward high confidence when correct
            1 - confidences  # Reward low confidence when incorrect
        )
        
        # Entropy penalty (encourage confident predictions)
        entropy = -(probs * probs.log()).sum(dim=1)
        entropy_penalty = -0.1 * entropy
        
        total_reward = correctness_reward + 0.5 * calibration_reward + entropy_penalty
        return total_reward
    
    def update_baseline(self, rewards):
        """Update running baseline for variance reduction."""
        if self.baseline is None:
            self.baseline = rewards.mean().item()
        else:
            self.baseline = 0.9 * self.baseline + 0.1 * rewards.mean().item()
    
    def train_step(self, images, labels, optimizer):
        """Single training step with sequence-level RL."""
        self.model.train()
        
        # Forward pass
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)
        
        # Sample predictions
        dist = torch.distributions.Categorical(probs)
        predictions = dist.sample()
        log_probs = dist.log_prob(predictions)
        
        # Compute rewards
        rewards = self.compute_sequence_reward(logits, labels, predictions)
        self.update_baseline(rewards)
        
        # Advantage
        advantages = rewards - self.baseline
        
        # Policy gradient loss
        pg_loss = -(log_probs * advantages.detach()).mean()
        
        # Cross-entropy loss for stability
        ce_loss = F.cross_entropy(logits, labels)
        
        # Combined loss
        loss = 0.5 * pg_loss + 0.5 * ce_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'reward': rewards.mean().item(),
            'accuracy': (predictions == labels).float().mean().item()
        }
    
    def finetune(self, dataloader, epochs=5, lr=1e-5):
        """Fine-tune model using sequence tutor."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss, total_reward, total_acc = 0, 0, 0
            
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                metrics = self.train_step(images, labels, optimizer)
                total_loss += metrics['loss']
                total_reward += metrics['reward']
                total_acc += metrics['accuracy']
            
            n = len(dataloader)
            print(f"Sequence Tutor Epoch {epoch+1}: Loss={total_loss/n:.4f}, "
                  f"Reward={total_reward/n:.4f}, Acc={total_acc/n:.4f}")


# ============================================================================
# 3. Pluralistic Alignment
# ============================================================================

class PreferenceStyle:
    """Different human preference styles."""
    STRICT = 'strict'           # Prefers high confidence, penalizes errors heavily
    TOLERANT = 'tolerant'       # More forgiving of errors, values uncertainty awareness
    CONFIDENCE_BIASED = 'confidence_biased'  # Prefers confident predictions
    CONSERVATIVE = 'conservative'  # Prefers cautious predictions


class PlurallisticAlignmentModule(nn.Module):
    """
    Module that supports multiple human preference styles.
    Learns to adapt predictions based on preference context.
    """
    
    def __init__(self, num_classes, hidden_dim=256, num_styles=4):
        super().__init__()
        self.num_styles = num_styles
        
        # Style embeddings
        self.style_embeddings = nn.Embedding(num_styles, hidden_dim)
        
        # Style-conditioned prediction adjustment
        self.style_adapter = nn.Sequential(
            nn.Linear(num_classes + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Style-specific confidence calibration
        self.confidence_calibrators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_classes, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            ) for _ in range(num_styles)
        ])
    
    def forward(self, logits, style_idx):
        """
        Adjust predictions based on preference style.
        
        Args:
            logits: Model output logits (B, num_classes)
            style_idx: Preference style index (B,) or scalar
        """
        if isinstance(style_idx, int):
            style_idx = torch.full((logits.size(0),), style_idx, device=logits.device)
        
        # Get style embedding
        style_emb = self.style_embeddings(style_idx)
        
        # Combine logits with style
        combined = torch.cat([logits, style_emb], dim=1)
        adjusted_logits = logits + self.style_adapter(combined)
        
        # Style-specific confidence
        probs = F.softmax(adjusted_logits, dim=1)
        
        # Apply style-specific calibration
        calibrated_confidences = []
        for i in range(logits.size(0)):
            style = style_idx[i].item()
            conf = self.confidence_calibrators[style](probs[i:i+1])
            calibrated_confidences.append(conf)
        
        calibrated_confidence = torch.cat(calibrated_confidences, dim=0)
        
        return adjusted_logits, calibrated_confidence
    
    def get_style_reward(self, predictions, labels, confidences, style):
        """Compute reward based on preference style."""
        correct = (predictions == labels).float()
        
        if style == 0:  # STRICT
            reward = correct * 2 - 1
            reward += (confidences > 0.9).float() * 0.5 * correct
            reward -= (1 - correct) * confidences  # Penalize confident errors
            
        elif style == 1:  # TOLERANT
            reward = correct * 1.5 - 0.5
            reward += (confidences < 0.5).float() * (1 - correct) * 0.3  # Reward uncertainty on errors
            
        elif style == 2:  # CONFIDENCE_BIASED
            reward = confidences * correct - (1 - confidences) * (1 - correct)
            
        else:  # CONSERVATIVE
            reward = correct - 0.5 * confidences * (1 - correct)
            reward += (confidences < 0.7).float() * 0.2
        
        return reward


class PlurallisticTrainer:
    """Trainer for pluralistic alignment."""
    
    def __init__(self, model, alignment_module, device='cuda'):
        self.model = model
        self.alignment = alignment_module
        self.device = device
    
    def train_step(self, images, labels, style_idx, optimizer):
        """Training step with pluralistic alignment."""
        self.model.train()
        self.alignment.train()
        
        # Get base predictions
        logits = self.model(images)
        
        # Apply style-specific adjustment
        adjusted_logits, calibrated_conf = self.alignment(logits, style_idx)
        
        # Classification loss
        ce_loss = F.cross_entropy(adjusted_logits, labels)
        
        # Style-specific reward
        predictions = adjusted_logits.argmax(dim=1)
        rewards = self.alignment.get_style_reward(
            predictions, labels, calibrated_conf.squeeze(), style_idx[0].item()
        )
        
        # Combined loss
        loss = ce_loss - 0.1 * rewards.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item(), rewards.mean().item()


# ============================================================================
# 4. Variational Preference Learning
# ============================================================================

class VariationalPreferenceEncoder(nn.Module):
    """
    Encodes preferences into a latent distribution.
    Enables generalization to unseen disease types.
    """
    
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VariationalPreferenceDecoder(nn.Module):
    """Decodes latent preferences to prediction adjustments."""
    
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, z):
        return self.decoder(z)


class VariationalPreferenceLearner(nn.Module):
    """
    Variational preference learning module.
    Learns a latent preference distribution that generalizes to unseen diseases.
    """
    
    def __init__(self, num_classes, latent_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        # Preference encoder (from prediction + feedback)
        self.encoder = VariationalPreferenceEncoder(num_classes * 2, latent_dim)
        
        # Preference decoder (to prediction adjustment)
        self.decoder = VariationalPreferenceDecoder(latent_dim, num_classes)
        
        # Prior network (learns class-conditional prior)
        self.prior_net = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # mu and logvar
        )
    
    def encode(self, predictions, feedback):
        """Encode prediction-feedback pair to latent preference."""
        x = torch.cat([predictions, feedback], dim=1)
        mu, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """Decode latent preference to prediction adjustment."""
        return self.decoder(z)
    
    def get_prior(self, predictions):
        """Get class-conditional prior."""
        prior_params = self.prior_net(predictions)
        mu = prior_params[:, :self.latent_dim]
        logvar = prior_params[:, self.latent_dim:]
        return mu, logvar
    
    def forward(self, predictions, feedback=None):
        """
        Forward pass.
        If feedback is provided, encode it. Otherwise, sample from prior.
        """
        if feedback is not None:
            z, mu, logvar = self.encode(predictions, feedback)
            prior_mu, prior_logvar = self.get_prior(predictions)
            
            # KL divergence
            kl_loss = -0.5 * torch.sum(
                1 + logvar - prior_logvar - 
                (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp(),
                dim=1
            ).mean()
        else:
            # Sample from prior
            prior_mu, prior_logvar = self.get_prior(predictions)
            z = self.encoder.reparameterize(prior_mu, prior_logvar)
            kl_loss = torch.tensor(0.0, device=predictions.device)
        
        # Decode to adjustment
        adjustment = self.decode(z)
        adjusted_predictions = predictions + adjustment
        
        return adjusted_predictions, kl_loss
    
    def sample_preferences(self, predictions, num_samples=5):
        """Sample multiple preference-adjusted predictions."""
        prior_mu, prior_logvar = self.get_prior(predictions)
        
        samples = []
        for _ in range(num_samples):
            z = self.encoder.reparameterize(prior_mu, prior_logvar)
            adjustment = self.decode(z)
            samples.append(predictions + adjustment)
        
        return torch.stack(samples, dim=0)


class VariationalPreferenceTrainer:
    """Trainer for variational preference learning."""
    
    def __init__(self, model, vpl_module, device='cuda'):
        self.model = model
        self.vpl = vpl_module
        self.device = device
    
    def create_feedback(self, predictions, labels):
        """Create feedback signal from labels."""
        batch_size = predictions.size(0)
        num_classes = predictions.size(1)
        
        # One-hot encode correct labels
        feedback = torch.zeros_like(predictions)
        feedback.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # Add noise for robustness
        feedback = feedback + 0.1 * torch.randn_like(feedback)
        
        return feedback
    
    def train_step(self, images, labels, optimizer, beta=0.1):
        """Training step with variational preference learning."""
        self.model.train()
        self.vpl.train()
        
        # Get base predictions
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)
        
        # Create feedback
        feedback = self.create_feedback(probs, labels)
        
        # Variational preference adjustment
        adjusted_probs, kl_loss = self.vpl(probs, feedback)
        
        # Reconstruction loss
        recon_loss = F.cross_entropy(adjusted_probs, labels)
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }


# ============================================================================
# 5. Active Learning Loop
# ============================================================================

class UncertaintySampler:
    """Sample uncertain examples for human labeling."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def compute_uncertainty(self, images):
        """Compute prediction uncertainty using entropy."""
        self.model.eval()
        with torch.no_grad():
            logits = self.model(images.to(self.device))
            probs = F.softmax(logits, dim=1)
            
            # Entropy-based uncertainty
            entropy = -(probs * probs.log()).sum(dim=1)
            
            # Margin-based uncertainty (difference between top 2 predictions)
            top2 = probs.topk(2, dim=1)[0]
            margin = top2[:, 0] - top2[:, 1]
            margin_uncertainty = 1 - margin
            
            # Combined uncertainty
            uncertainty = 0.5 * entropy + 0.5 * margin_uncertainty
        
        return uncertainty.cpu().numpy()
    
    def select_samples(self, dataset, budget, batch_size=32):
        """Select most uncertain samples for labeling."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_uncertainties = []
        for images, _ in loader:
            uncertainties = self.compute_uncertainty(images)
            all_uncertainties.extend(uncertainties)
        
        all_uncertainties = np.array(all_uncertainties)
        
        # Select top uncertain samples
        selected_indices = np.argsort(all_uncertainties)[-budget:]
        
        return selected_indices, all_uncertainties[selected_indices]


class ActiveLearningLoop:
    """
    Active learning loop for incremental model improvement.
    Selects uncertain samples, queries human (simulated), and retrains.
    """
    
    def __init__(self, model, train_dataset, valid_dataset, device='cuda'):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.device = device
        self.sampler = UncertaintySampler(model, device)
        self.labeled_indices = set()
        self.query_history = []
    
    def query_human(self, indices, simulate=True):
        """
        Query human for labels.
        In production, this would interface with a labeling system.
        """
        if simulate:
            # Simulated human labeling (uses ground truth)
            labels = []
            for idx in indices:
                _, label = self.train_dataset[idx]
                labels.append(label)
            return labels
        else:
            # Placeholder for real human labeling interface
            raise NotImplementedError("Real human labeling not implemented")
    
    def retrain(self, new_indices, new_labels, epochs=5, lr=1e-4):
        """Retrain model with new labeled data."""
        # Add to labeled set
        self.labeled_indices.update(new_indices)
        
        # Create subset with all labeled data
        all_indices = list(self.labeled_indices)
        subset = Subset(self.train_dataset, all_indices)
        loader = DataLoader(subset, batch_size=32, shuffle=True)
        
        # Fine-tune model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Retrain Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}")
    
    def evaluate(self):
        """Evaluate model on validation set."""
        loader = DataLoader(self.valid_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        return accuracy
    
    def run(self, num_rounds=5, budget_per_round=20, initial_budget=100):
        """Run active learning loop."""
        print("=" * 60)
        print("Active Learning Loop")
        print("=" * 60)
        
        # Initial random sampling
        print(f"\nInitial sampling: {initial_budget} samples")
        initial_indices = np.random.choice(len(self.train_dataset), initial_budget, replace=False)
        initial_labels = self.query_human(initial_indices)
        self.retrain(initial_indices, initial_labels, epochs=10)
        
        initial_acc = self.evaluate()
        print(f"Initial Accuracy: {initial_acc:.4f}")
        self.query_history.append({'round': 0, 'samples': initial_budget, 'accuracy': initial_acc})
        
        # Active learning rounds
        for round_num in range(1, num_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            
            # Select uncertain samples
            unlabeled_indices = [i for i in range(len(self.train_dataset)) if i not in self.labeled_indices]
            unlabeled_subset = Subset(self.train_dataset, unlabeled_indices)
            
            selected_relative, uncertainties = self.sampler.select_samples(
                unlabeled_subset, budget_per_round
            )
            selected_indices = [unlabeled_indices[i] for i in selected_relative]
            
            print(f"Selected {len(selected_indices)} samples (avg uncertainty: {uncertainties.mean():.4f})")
            
            # Query human
            new_labels = self.query_human(selected_indices)
            
            # Retrain
            self.retrain(selected_indices, new_labels, epochs=5)
            
            # Evaluate
            accuracy = self.evaluate()
            print(f"Accuracy after round {round_num}: {accuracy:.4f}")
            
            self.query_history.append({
                'round': round_num,
                'samples': len(self.labeled_indices),
                'accuracy': accuracy,
                'avg_uncertainty': float(uncertainties.mean())
            })
        
        return self.query_history


# ============================================================================
# Integration with Models
# ============================================================================

def integrate_rlhf_with_model(model, train_loader, valid_loader, device='cuda'):
    """
    Integrate RLHF components with a trained model.
    
    Args:
        model: Trained classification model
        train_loader: Training data loader
        valid_loader: Validation data loader
        device: Device to use
    
    Returns:
        Enhanced model with RLHF components
    """
    print("\n" + "=" * 60)
    print("Integrating RLHF Components")
    print("=" * 60)
    
    # Get number of classes from model output
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_input = sample_batch[0][:1].to(device)
        else:
            sample_input = sample_batch[:1].to(device)
        
        try:
            sample_output = model(sample_input)
            num_classes = sample_output.size(1)
        except:
            num_classes = 38  # Default for plant disease dataset
    
    print(f"Detected {num_classes} classes")
    
    # 1. Initialize Reward Model
    print("\n1. Initializing Reward Model...")
    reward_model = RewardModel(num_classes, CONFIG['reward_model_dim']).to(device)
    
    # 2. Initialize RLHF Trainer
    rlhf_trainer = RLHFTrainer(model, reward_model, device)
    
    # Simulate feedback collection
    print("   Collecting simulated feedback...")
    model.eval()
    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            confidences = F.softmax(outputs, dim=1).max(dim=1)[0].cpu().numpy()
            rlhf_trainer.feedback_collector.simulate_feedback(
                predictions, labels.numpy(), confidences
            )
            if len(rlhf_trainer.feedback_collector.feedback_buffer) > 500:
                break
    
    print(f"   Collected {len(rlhf_trainer.feedback_collector.feedback_buffer)} feedback samples")
    
    # 3. Initialize Sequence Tutor
    print("\n2. Initializing Sequence Tutor...")
    sequence_tutor = SequenceTutor(model, device)
    
    # 4. Initialize Pluralistic Alignment
    print("\n3. Initializing Pluralistic Alignment...")
    alignment_module = PlurallisticAlignmentModule(
        num_classes, CONFIG['reward_model_dim'], CONFIG['num_preference_styles']
    ).to(device)
    pluralistic_trainer = PlurallisticTrainer(model, alignment_module, device)
    
    # 5. Initialize Variational Preference Learning
    print("\n4. Initializing Variational Preference Learning...")
    vpl_module = VariationalPreferenceLearner(
        num_classes, CONFIG['preference_latent_dim']
    ).to(device)
    vpl_trainer = VariationalPreferenceTrainer(model, vpl_module, device)
    
    return {
        'model': model,
        'reward_model': reward_model,
        'rlhf_trainer': rlhf_trainer,
        'sequence_tutor': sequence_tutor,
        'alignment_module': alignment_module,
        'pluralistic_trainer': pluralistic_trainer,
        'vpl_module': vpl_module,
        'vpl_trainer': vpl_trainer
    }


def run_rlhf_pipeline(components, train_loader, valid_loader, device='cuda'):
    """
    Run the complete RLHF pipeline.
    
    Args:
        components: Dictionary of RLHF components
        train_loader: Training data loader
        valid_loader: Validation data loader
        device: Device to use
    """
    print("\n" + "=" * 60)
    print("Running RLHF Pipeline")
    print("=" * 60)
    
    model = components['model']
    
    # Evaluate baseline
    print("\nBaseline Evaluation:")
    baseline_acc = evaluate_model(model, valid_loader, device)
    print(f"  Accuracy: {baseline_acc:.4f}")
    
    # 1. Train Reward Model
    print("\n1. Training Reward Model...")
    # Create simple feature extractor
    def feature_extractor(x):
        model.eval()
        with torch.no_grad():
            output = model(x)
            return F.softmax(output, dim=1)
    
    components['rlhf_trainer'].train_reward_model(feature_extractor, train_loader, epochs=3)
    
    # 2. Sequence Tutor Fine-tuning
    print("\n2. Sequence Tutor Fine-tuning...")
    components['sequence_tutor'].finetune(train_loader, epochs=2, lr=1e-5)
    
    seq_acc = evaluate_model(model, valid_loader, device)
    print(f"  Accuracy after Sequence Tutor: {seq_acc:.4f}")
    
    # 3. Pluralistic Alignment Training
    print("\n3. Pluralistic Alignment Training...")
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(components['alignment_module'].parameters()),
        lr=1e-5
    )
    
    for style_idx in range(CONFIG['num_preference_styles']):
        style_name = ['STRICT', 'TOLERANT', 'CONFIDENCE_BIASED', 'CONSERVATIVE'][style_idx]
        print(f"  Training for style: {style_name}")
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            style = torch.full((images.size(0),), style_idx, device=device)
            
            loss, reward = components['pluralistic_trainer'].train_step(
                images, labels, style, optimizer
            )
            break  # One batch per style for demo
    
    # 4. Variational Preference Learning
    print("\n4. Variational Preference Learning...")
    vpl_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(components['vpl_module'].parameters()),
        lr=1e-5
    )
    
    for epoch in range(2):
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            metrics = components['vpl_trainer'].train_step(images, labels, vpl_optimizer)
            total_loss += metrics['loss']
        
        print(f"  VPL Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_acc = evaluate_model(model, valid_loader, device)
    print(f"  Accuracy: {final_acc:.4f}")
    print(f"  Improvement: {(final_acc - baseline_acc) * 100:.2f}%")
    
    return {
        'baseline_accuracy': baseline_acc,
        'final_accuracy': final_acc,
        'improvement': final_acc - baseline_acc
    }


def evaluate_model(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[-1]
            else:
                continue
            
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    return accuracy_score(all_labels, all_preds)


# ============================================================================
# Model Loading Utilities
# ============================================================================

# Available model configurations
MODEL_CONFIGS = {
    'ADSD': {
        'path': 'model_files/ADSD/ADSD_fast_model.pth',
        'class_name': 'ADSD_FastClassifier',
        'module': 'train2',
        'image_size': 160,
        'requires_graph': False
    },
    'ECAM': {
        'path': 'model_files/ECAM/ECAM_fast.pth',
        'class_name': 'ECAM_fast',
        'module': 'train4',
        'image_size': 224,
        'requires_graph': False
    },
    'LGNM': {
        'path': 'model_files/LGNM/LGNM_fast_model.pth',
        'class_name': 'LGNM_FAST',
        'module': 'train1',
        'image_size': 160,
        'requires_graph': True  # LGNM requires graph input
    },
    'S-ViT_Lite': {
        'path': 'model_files/S-ViT_Lite/S-ViT_Lite_fast_model.pth',
        'class_name': 'SViT_Lite_Fast_Modified',
        'module': 'train3',
        'image_size': 160,
        'requires_graph': False
    }
}


def load_model(model_name, num_classes, device='cuda'):
    """
    Load a trained model by name.
    
    Args:
        model_name: One of 'ADSD', 'ECAM', 'LGNM', 'S-ViT_Lite'
        num_classes: Number of output classes
        device: Device to load the model on
    
    Returns:
        Loaded model on the specified device
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    model_path = config['path']
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        return None
    
    # Import the model class dynamically
    import importlib
    try:
        module = importlib.import_module(config['module'])
        model_class = getattr(module, config['class_name'])
    except (ImportError, AttributeError) as e:
        print(f"Error importing model class: {e}")
        return None
    
    # Create model instance
    model = model_class(num_classes=num_classes).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Successfully loaded {model_name} model from {model_path}")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return None
    
    return model


def create_model_wrapper(model, model_name):
    """
    Create a wrapper class for models that have non-standard forward methods.
    This is particularly needed for LGNM which requires graph input.
    """
    if model_name == 'LGNM':
        # LGNM requires special handling due to graph input
        # We'll create a simple wrapper that ignores graph for RLHF
        class LGNMWrapper(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base = base_model
            
            def forward(self, x):
                # For RLHF, we'll use only the CNN features
                # Create dummy graph data
                batch_size = x.size(0)
                device = x.device
                
                # Use CNN features only via pooling
                cnn_out = self.base.cnn(x)
                cnn_out = self.base.pool(cnn_out).flatten(1)
                cnn_feat = self.base.proj(cnn_out)
                
                # Skip GNN for RLHF (use zeros as placeholder)
                gnn_feat = torch.zeros(batch_size, 64, device=device)
                
                fused = torch.cat([cnn_feat, gnn_feat], dim=1)
                fused = F.relu(self.base.fuse(fused))
                return self.base.cls(fused)
        
        return LGNMWrapper(model)
    
    return model


# ============================================================================
# Main Function
# ============================================================================

def main(model_name='ADSD'):
    """
    Main function demonstrating RLHF and Active Learning.
    
    Args:
        model_name: Name of the model to use. One of 'ADSD', 'ECAM', 'LGNM', 'S-ViT_Lite'
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    print("=" * 60)
    print(f"RLHF + Active Learning with {model_name}")
    print("=" * 60)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    # Get image size for the selected model
    if model_name in MODEL_CONFIGS:
        image_size = MODEL_CONFIGS[model_name]['image_size']
    else:
        image_size = 224
    
    print(f"Using image size: {image_size}x{image_size}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Try different dataset paths
    dataset_paths = [
        ('dataset/train', 'dataset/valid', 'dataset/test'),
        ('dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
         'dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
         'dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/test')
    ]
    
    train_dir, valid_dir, test_dir = None, None, None
    for train_p, valid_p, test_p in dataset_paths:
        if os.path.exists(train_p):
            train_dir, valid_dir, test_dir = train_p, valid_p, test_p
            break
    
    if train_dir is None:
        print("Dataset not found. Please ensure the dataset is in one of the following locations:")
        for paths in dataset_paths:
            print(f"  - {paths[0]}")
        return None
    
    print(f"Using dataset from: {train_dir}")
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
    
    # Use fewer workers on Windows
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=num_workers)
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    
    # Load the specified model
    print(f"\nLoading {model_name} model...")
    model = load_model(model_name, num_classes, device)
    
    if model is None:
        print(f"\nCould not load {model_name}. Trying to create a fallback model...")
        from torchvision import models
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        print("Using ResNet18 as fallback model.")
    else:
        # Wrap the model if needed
        model = create_model_wrapper(model, model_name)
    
    # Integrate RLHF components
    components = integrate_rlhf_with_model(model, train_loader, valid_loader, device)
    
    # Run RLHF pipeline
    results = run_rlhf_pipeline(components, train_loader, valid_loader, device)
    
    # Run Active Learning
    print("\n" + "=" * 60)
    print("Running Active Learning Loop")
    print("=" * 60)
    
    active_learner = ActiveLearningLoop(model, train_dataset, valid_dataset, device)
    history = active_learner.run(num_rounds=3, budget_per_round=50, initial_budget=200)
    
    # Save results
    results['active_learning_history'] = history
    results['model_name'] = model_name
    
    results_path = f'rlhf_results_{model_name}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    return results


def run_all_models():
    """
    Run RLHF + Active Learning on all available models.
    """
    all_results = {}
    
    for model_name in MODEL_CONFIGS.keys():
        print(f"\n{'#' * 70}")
        print(f"# Running RLHF for: {model_name}")
        print(f"{'#' * 70}\n")
        
        try:
            results = main(model_name=model_name)
            if results:
                all_results[model_name] = results
        except Exception as e:
            print(f"Error running {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Save combined results
    with open('rlhf_results_all_models.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("All Models Completed")
    print("=" * 60)
    print(f"Results saved to rlhf_results_all_models.json")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='RLHF + Active Learning for Plant Disease Classification')
    parser.add_argument('--model', type=str, default='ADSD',
                        choices=['ADSD', 'ECAM', 'LGNM', 'S-ViT_Lite', 'all'],
                        help='Model to use (default: ADSD)')
    args = parser.parse_args()
    
    if args.model == 'all':
        run_all_models()
    else:
        main(model_name=args.model)
