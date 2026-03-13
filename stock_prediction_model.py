"""
Stock Price Prediction Model using News Sentiment
Predicts ret_1d, ret_2d, ret_3d from news headlines
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StockNewsDataset(Dataset):
    """Custom dataset for stock news headlines and returns"""
    
    def __init__(self, headlines, returns_1d, returns_2d, returns_3d, tokenizer, max_length=128):
        self.headlines = headlines.reset_index(drop=True)
        self.returns_1d = returns_1d.reset_index(drop=True)
        self.returns_2d = returns_2d.reset_index(drop=True)
        self.returns_3d = returns_3d.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.headlines)
    
    def __getitem__(self, idx):
        headline = str(self.headlines.iloc[idx])
        
        # Tokenize headline
        encoding = self.tokenizer(
            headline,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'ret_1d': torch.tensor(self.returns_1d.iloc[idx], dtype=torch.float32),
            'ret_2d': torch.tensor(self.returns_2d.iloc[idx], dtype=torch.float32),
            'ret_3d': torch.tensor(self.returns_3d.iloc[idx], dtype=torch.float32),
        }


class MultiTaskStockPredictor(nn.Module):
    """Multi-task learning model for predicting 1-day, 2-day, 3-day returns"""
    
    def __init__(self, model_name='bert-base-uncased', hidden_size=768, dropout=0.1):
        super(MultiTaskStockPredictor, self).__init__()
        
        # Load pre-trained BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Shared layers
        self.shared_dense = nn.Linear(hidden_size, 256)
        self.shared_activation = nn.ReLU()
        self.shared_dropout = nn.Dropout(dropout)
        
        # Task-specific heads
        # 1-day return prediction
        self.head_1d = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 2-day return prediction
        self.head_2d = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        # 3-day return prediction
        self.head_3d = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        
        # Shared layers
        shared_features = self.shared_dense(cls_output)
        shared_features = self.shared_activation(shared_features)
        shared_features = self.shared_dropout(shared_features)
        
        # Task-specific predictions
        pred_1d = self.head_1d(shared_features)
        pred_2d = self.head_2d(shared_features)
        pred_3d = self.head_3d(shared_features)
        
        return pred_1d, pred_2d, pred_3d


class StockDatasetLoader:
    """Load and combine data from all stock CSV files"""
    
    def __init__(self, aligned_dir='Data/processed/aligned'):
        self.aligned_dir = Path(aligned_dir)
    
    def get_available_stocks(self) -> List[str]:
        """Get list of available aligned CSV files"""
        csv_files = list(self.aligned_dir.glob('*_aligned.csv'))
        return [f.stem.replace('_aligned', '') for f in csv_files]
    
    def load_stock_data(self, stock_name: str) -> pd.DataFrame:
        """Load data for a single stock"""
        file_path = self.aligned_dir / f'{stock_name}_aligned.csv'
        if not file_path.exists():
            return None
        
        df = pd.read_csv(file_path)
        # Clean data - remove rows with NaN in critical columns
        df = df.dropna(subset=['headline', 'ret_1d', 'ret_2d', 'ret_3d'])
        # Remove duplicates
        df = df.drop_duplicates(subset=['headline', 'event_date'])
        
        return df
    
    def load_all_stocks(self, test_size=0.15, val_size=0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and combine data from all available stocks"""
        all_data = []
        stocks = self.get_available_stocks()
        
        print(f"Found {len(stocks)} stock(s): {stocks}")
        
        for stock in stocks:
            df = self.load_stock_data(stock)
            if df is not None and len(df) > 0:
                df['stock'] = stock
                all_data.append(df)
                print(f"  {stock}: {len(df)} records")
        
        if not all_data:
            raise ValueError("No data found in aligned directory")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates()
        
        print(f"\nTotal records: {len(combined_df)}")
        
        # Shuffle and split
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)
        
        test_idx = int(len(combined_df) * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        train_df = combined_df[:val_idx]
        val_df = combined_df[val_idx:test_idx]
        test_df = combined_df[test_idx:]
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}\n")
        
        return train_df, val_df, test_df


class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, model, device, learning_rate=2e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            ret_1d = batch['ret_1d'].to(self.device)
            ret_2d = batch['ret_2d'].to(self.device)
            ret_3d = batch['ret_3d'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred_1d, pred_2d, pred_3d = self.model(input_ids, attention_mask)
            
            # Calculate loss (multi-task: equal weight for all tasks)
            loss_1d = self.criterion(pred_1d.squeeze(-1), ret_1d)
            loss_2d = self.criterion(pred_2d.squeeze(-1), ret_2d)
            loss_3d = self.criterion(pred_3d.squeeze(-1), ret_3d)
            
            loss = loss_1d + loss_2d + loss_3d
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = {'1d': [], '2d': [], '3d': []}
        all_true = {'1d': [], '2d': [], '3d': []}
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ret_1d = batch['ret_1d'].to(self.device)
                ret_2d = batch['ret_2d'].to(self.device)
                ret_3d = batch['ret_3d'].to(self.device)
                
                # Forward pass
                pred_1d, pred_2d, pred_3d = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss_1d = self.criterion(pred_1d.squeeze(-1), ret_1d)
                loss_2d = self.criterion(pred_2d.squeeze(-1), ret_2d)
                loss_3d = self.criterion(pred_3d.squeeze(-1), ret_3d)
                
                loss = loss_1d + loss_2d + loss_3d
                total_loss += loss.item()
                
                # Store predictions
                all_preds['1d'].extend(pred_1d.squeeze(-1).cpu().numpy())
                all_preds['2d'].extend(pred_2d.squeeze(-1).cpu().numpy())
                all_preds['3d'].extend(pred_3d.squeeze(-1).cpu().numpy())
                
                all_true['1d'].extend(ret_1d.cpu().numpy())
                all_true['2d'].extend(ret_2d.cpu().numpy())
                all_true['3d'].extend(ret_3d.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = {}
        for key in all_preds:
            mse = mean_squared_error(all_true[key], all_preds[key])
            mae = mean_absolute_error(all_true[key], all_preds[key])
            r2 = r2_score(all_true[key], all_preds[key])
            metrics[f'ret_{key}_mse'] = mse
            metrics[f'ret_{key}_mae'] = mae
            metrics[f'ret_{key}_r2'] = r2
        
        return val_loss, metrics
    
    def train(self, train_loader, val_loader, epochs=10, patience=3):
        """Full training loop with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training for {epochs} epochs...\n")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Metrics:")
            for key, val in val_metrics.items():
                print(f"    {key}: {val:.4f}")
            print()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        all_preds = {'1d': [], '2d': [], '3d': []}
        all_true = {'1d': [], '2d': [], '3d': []}
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                ret_1d = batch['ret_1d']
                ret_2d = batch['ret_2d']
                ret_3d = batch['ret_3d']
                
                pred_1d, pred_2d, pred_3d = self.model(input_ids, attention_mask)
                
                all_preds['1d'].extend(pred_1d.squeeze(-1).cpu().numpy())
                all_preds['2d'].extend(pred_2d.squeeze(-1).cpu().numpy())
                all_preds['3d'].extend(pred_3d.squeeze(-1).cpu().numpy())
                
                all_true['1d'].extend(ret_1d.numpy())
                all_true['2d'].extend(ret_2d.numpy())
                all_true['3d'].extend(ret_3d.numpy())
        
        # Calculate test metrics
        test_metrics = {}
        for key in all_preds:
            mse = mean_squared_error(all_true[key], all_preds[key])
            mae = mean_absolute_error(all_true[key], all_preds[key])
            r2 = r2_score(all_true[key], all_preds[key])
            rmse = np.sqrt(mse)
            
            test_metrics[f'ret_{key}'] = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
        
        return test_metrics, all_preds, all_true


def main():
    """Main training pipeline"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load data
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data_loader = StockDatasetLoader('Data/processed/aligned')
    train_df, val_df, test_df = data_loader.load_all_stocks()
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = StockNewsDataset(
        train_df['headline'],
        train_df['ret_1d'],
        train_df['ret_2d'],
        train_df['ret_3d'],
        tokenizer
    )
    
    val_dataset = StockNewsDataset(
        val_df['headline'],
        val_df['ret_1d'],
        val_df['ret_2d'],
        val_df['ret_3d'],
        tokenizer
    )
    
    test_dataset = StockNewsDataset(
        test_df['headline'],
        test_df['ret_1d'],
        test_df['ret_2d'],
        test_df['ret_3d'],
        tokenizer
    )
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataloaders created with batch size: {batch_size}\n")
    
    # Initialize model
    print("=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    model = MultiTaskStockPredictor('bert-base-uncased')
    print("Model initialized\n")
    
    # Train model
    print("=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    trainer = ModelTrainer(model, device, learning_rate=2e-5)
    history = trainer.train(train_loader, val_loader, epochs=10, patience=3)
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    test_metrics, test_preds, test_true = trainer.evaluate(test_loader)
    
    print("\nTest Results:")
    for key, metrics in test_metrics.items():
        print(f"\n{key}:")
        for metric_name, metric_val in metrics.items():
            print(f"  {metric_name}: {metric_val:.4f}")
    
    # Save model and results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("✓ Model saved: model_weights.pth")
    
    # Save results
    results = {
        'test_metrics': {k: {mk: float(mv) for mk, mv in v.items()} for k, v in test_metrics.items()},
        'training_history': {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved: training_results.json")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'true_ret_1d': test_true['1d'],
        'pred_ret_1d': test_preds['1d'],
        'true_ret_2d': test_true['2d'],
        'pred_ret_2d': test_preds['2d'],
        'true_ret_3d': test_true['3d'],
        'pred_ret_3d': test_preds['3d'],
    })
    pred_df.to_csv('test_predictions.csv', index=False)
    print("✓ Predictions saved: test_predictions.csv")


if __name__ == '__main__':
    main()
