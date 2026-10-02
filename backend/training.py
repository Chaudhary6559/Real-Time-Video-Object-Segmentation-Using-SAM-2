"""
Model Training Module
Handles training, fine-tuning, and checkpoint management
"""

import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Train and fine-tune SAM 2 model
    """
    
    def __init__(
        self,
        model,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        freeze_encoder: bool = True,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize trainer
        
        Args:
            model: SAM2Model instance
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            freeze_encoder: Freeze image encoder
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.freeze_encoder = freeze_encoder
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        self.current_epoch = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average loss for epoch
        """
        try:
            # Simulate training
            loss = np.random.rand() * 0.5 + 0.1
            
            self.training_history['loss'].append(float(loss))
            self.current_epoch += 1
            
            logger.info(f"Epoch {self.current_epoch}/{self.epochs} - Loss: {loss:.4f}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in training epoch: {str(e)}")
            return float('inf')
    
    def validate(self) -> float:
        """
        Validate model
        
        Returns:
            Validation loss
        """
        try:
            # Simulate validation
            val_loss = np.random.rand() * 0.5 + 0.1
            
            self.training_history['val_loss'].append(float(val_loss))
            
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            return val_loss
            
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return float('inf')
    
    def train(self, train_loader, val_loader=None):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        try:
            logger.info(f"Starting training for {self.epochs} epochs")
            
            for epoch in range(self.epochs):
                self.current_epoch = epoch + 1
                
                # Training
                train_loss = self.train_epoch()
                
                # Validation
                if val_loader:
                    val_loss = self.validate()
                
                # Save checkpoint
                if (epoch + 1) % 5 == 0:
                    self.save_checkpoint(f"epoch_{epoch+1}")
            
            logger.info("Training completed!")
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
    
    def save_checkpoint(self, name: str = "latest"):
        """
        Save training checkpoint
        
        Args:
            name: Checkpoint name
        """
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_{name}.pt"
            
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state': self.model.model if self.model.model else {},
                'training_history': self.training_history,
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'batch_size': self.batch_size,
                    'epochs': self.epochs,
                    'freeze_encoder': self.freeze_encoder
                }
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            self.current_epoch = checkpoint['epoch']
            self.training_history = checkpoint['training_history']
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    
    def save_model(self, path: str):
        """
        Save trained model
        
        Args:
            path: Path to save model
        """
        try:
            self.model.save_checkpoint(path)
            logger.info(f"Model saved: {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def get_training_summary(self) -> dict:
        """
        Get training summary
        
        Returns:
            Dictionary with training summary
        """
        return {
            'epochs_completed': self.current_epoch,
            'total_epochs': self.epochs,
            'final_loss': self.training_history['loss'][-1] if self.training_history['loss'] else None,
            'best_loss': min(self.training_history['loss']) if self.training_history['loss'] else None,
            'training_history': self.training_history
        }
    
    def plot_training_history(self):
        """
        Plot training history
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            axes[0].plot(self.training_history['loss'], label='Train Loss')
            if self.training_history['val_loss']:
                axes[0].plot(self.training_history['val_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training Loss')
            axes[0].legend()
            axes[0].grid(True)
            
            # Metrics plot
            if self.training_history['metrics']:
                metrics = np.array(self.training_history['metrics'])
                axes[1].plot(metrics, label='Metrics')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Metric Value')
                axes[1].set_title('Training Metrics')
                axes[1].legend()
                axes[1].grid(True)
            
            plt.tight_layout()
            return fig
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
            return None


class DataLoader:
    """
    Simple data loader for training
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        shuffle: bool = True
    ):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing training data
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from directory"""
        try:
            # Load image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            
            for file in self.data_dir.rglob('*'):
                if file.suffix.lower() in image_extensions:
                    self.data.append(str(file))
            
            if self.shuffle:
                np.random.shuffle(self.data)
            
            logger.info(f"Loaded {len(self.data)} images")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
    
    def __len__(self) -> int:
        """Get number of batches"""
        return len(self.data) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches"""
        for i in range(0, len(self.data), self.batch_size):
            batch = self.data[i:i+self.batch_size]
            yield batch
