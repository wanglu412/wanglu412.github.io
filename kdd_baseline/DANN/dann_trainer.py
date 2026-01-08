"""
Training and evaluation functions for DANN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import time
import logging


class DANNTrainer:
    """Trainer for Domain-Adversarial Neural Network"""

    def __init__(self, model, device, learning_rate=0.001, weight_decay=5e-4, logger=None):
        """
        Initialize DANN trainer

        Args:
            model: DANN model
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            logger: Logger for recording training progress (optional)
        """
        self.model = model.to(device)
        self.device = device
        self.logger = logger if logger else logging.getLogger('DANN')
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Loss functions
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader, epoch, max_epochs, domain_weight=1.0):
        """
        Train for one epoch

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            max_epochs: Total number of epochs
            domain_weight: Weight for domain classification loss

        Returns:
            Dictionary containing average losses
        """
        self.model.train()
        total_class_loss = 0
        total_domain_loss = 0
        total_loss = 0
        num_batches = 0

        # Calculate lambda for gradient reversal
        from dann_model import get_lambda_alpha
        alpha = get_lambda_alpha(epoch, max_epochs)

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
            data = data.to(self.device)

            # Forward pass
            class_output, domain_output, features = self.model(
                data.x, data.edge_index, data.batch, alpha=alpha
            )

            # Class loss (only on samples with labels)
            if hasattr(data, 'y') and data.y is not None:
                # Handle different label formats
                if data.y.dim() > 1 and data.y.size(1) > 1:
                    # Multi-label or already one-hot
                    class_loss = self.class_criterion(class_output, data.y.float())
                else:
                    # Single label - convert to float
                    class_loss = self.class_criterion(
                        class_output.squeeze(),
                        data.y.float().squeeze()
                    )
            else:
                class_loss = torch.tensor(0.0).to(self.device)

            # Domain loss
            if hasattr(data, 'domain') and data.domain is not None:
                domain_loss = self.domain_criterion(domain_output, data.domain.long())
            else:
                domain_loss = torch.tensor(0.0).to(self.device)

            # Total loss (as in DANN paper)
            loss = class_loss + domain_weight * domain_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()
            num_batches += 1

        return {
            'total_loss': total_loss / num_batches,
            'class_loss': total_class_loss / num_batches,
            'domain_loss': total_domain_loss / num_batches,
            'alpha': alpha
        }

    @torch.no_grad()
    def evaluate(self, loader, split_name="Validation"):
        """
        Evaluate model on a dataset

        Args:
            loader: DataLoader for evaluation
            split_name: Name of the split (for display)

        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for data in tqdm(loader, desc=f"Evaluating {split_name}"):
            data = data.to(self.device)

            # Forward pass
            class_output = self.model.predict(data.x, data.edge_index, data.batch)

            # Get predictions and probabilities
            probs = torch.sigmoid(class_output)

            # Store predictions and labels
            if data.y.dim() > 1 and data.y.size(1) > 1:
                # Multi-label
                all_probs.append(probs.cpu())
                all_labels.append(data.y.cpu())
            else:
                # Single label
                all_probs.append(probs.squeeze().cpu())
                all_labels.append(data.y.squeeze().cpu())

        # Concatenate all predictions and labels
        all_probs = torch.cat(all_probs, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # Calculate AUC
        try:
            if len(all_labels.shape) > 1 and all_labels.shape[1] > 1:
                # Multi-label AUC (macro average)
                auc_scores = []
                for i in range(all_labels.shape[1]):
                    if len(np.unique(all_labels[:, i])) > 1:  # Check if both classes present
                        auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                        auc_scores.append(auc)
                auc = np.mean(auc_scores) if auc_scores else 0.0
            else:
                # Single label AUC
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_probs)
                else:
                    auc = 0.0
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            auc = 0.0

        # Calculate accuracy
        if len(all_labels.shape) > 1 and all_labels.shape[1] > 1:
            # Multi-label accuracy
            preds = (all_probs > 0.5).astype(int)
            accuracy = (preds == all_labels).mean()
        else:
            # Single label accuracy
            preds = (all_probs > 0.5).astype(int)
            accuracy = (preds == all_labels).mean()

        return {
            'auc': auc,
            'accuracy': accuracy
        }

    def train(self, train_loader, val_loader, test_loader,
              num_epochs=100, domain_weight=1.0,
              early_stopping_patience=20, save_path='best_dann_model.pt'):
        """
        Complete training loop with validation and testing

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            test_loader: DataLoader for testing
            num_epochs: Number of training epochs
            domain_weight: Weight for domain classification loss
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model

        Returns:
            Dictionary containing training history and final test results
        """
        best_val_auc = 0
        patience_counter = 0
        history = {
            'train_loss': [],
            'train_class_loss': [],
            'train_domain_loss': [],
            'val_auc': [],
            'test_auc': [],
            'alpha_values': []
        }

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting DANN Training")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Number of epochs: {num_epochs}")
        self.logger.info(f"Domain weight: {domain_weight}")
        self.logger.info(f"Early stopping patience: {early_stopping_patience}")
        self.logger.info(f"{'='*80}\n")

        start_time = time.time()

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(
                train_loader, epoch, num_epochs, domain_weight
            )

            # Validate
            val_metrics = self.evaluate(val_loader, "Validation")

            # Record history
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_class_loss'].append(train_metrics['class_loss'])
            history['train_domain_loss'].append(train_metrics['domain_loss'])
            history['val_auc'].append(val_metrics['auc'])
            history['alpha_values'].append(train_metrics['alpha'])

            # Print progress
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            self.logger.info(f"  Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Class: {train_metrics['class_loss']:.4f}, "
                  f"Domain: {train_metrics['domain_loss']:.4f})")
            self.logger.info(f"  Val AUC: {val_metrics['auc']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            self.logger.info(f"  Lambda (alpha): {train_metrics['alpha']:.4f}")

            # Early stopping and model saving
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': best_val_auc,
                }, save_path)
                self.logger.info(f"  >>> Best model saved with Val AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                self.logger.info(f"{'='*80}\n")
                break

        # Load best model and evaluate on test set
        self.logger.info(f"\n{'='*80}")
        self.logger.info("Loading best model for final testing...")
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = self.evaluate(test_loader, "Test")
        history['test_auc'].append(test_metrics['auc'])

        training_time = time.time() - start_time

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Training completed in {training_time/60:.2f} minutes")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Best Validation AUC: {best_val_auc:.4f}")
        self.logger.info(f"Final Test AUC: {test_metrics['auc']:.4f}")
        self.logger.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"{'='*80}\n")

        return {
            'history': history,
            'best_val_auc': best_val_auc,
            'test_auc': test_metrics['auc'],
            'test_accuracy': test_metrics['accuracy'],
            'training_time': training_time
        }
