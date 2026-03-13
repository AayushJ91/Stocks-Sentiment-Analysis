"""
Inference and prediction utilities for stock price model
Used for making predictions on new news headlines
"""

import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import numpy as np
from stock_prediction_model import MultiTaskStockPredictor


class StockPricePredictor:
    """Make predictions using trained model"""
    
    def __init__(self, model_weights_path='model_weights.pth', model_name='bert-base-uncased', device=None):
        """
        Initialize predictor with trained weights
        
        Args:
            model_weights_path: Path to saved model weights
            model_name: HuggingFace model name for tokenizer
            device: torch device (cuda/cpu). Auto-detected if None
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and load weights
        self.model = MultiTaskStockPredictor(model_name)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"Model loaded from {model_weights_path}")
        print(f"Using device: {self.device}")
    
    def predict_single(self, headline: str, return_probabilities=False):
        """
        Predict returns for a single headline
        
        Args:
            headline: News headline text
            return_probabilities: If True, returns raw model outputs
            
        Returns:
            Dictionary with predictions for ret_1d, ret_2d, ret_3d
        """
        # Tokenize
        inputs = self.tokenizer(
            headline,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            pred_1d, pred_2d, pred_3d = self.model(input_ids, attention_mask)
        
        return {
            'headline': headline,
            'ret_1d': float(pred_1d.squeeze(-1).cpu().numpy()[0]),
            'ret_2d': float(pred_2d.squeeze(-1).cpu().numpy()[0]),
            'ret_3d': float(pred_3d.squeeze(-1).cpu().numpy()[0]),
        }
    
    def predict_batch(self, headlines: list):
        """
        Predict returns for multiple headlines (batch)
        
        Args:
            headlines: List of news headlines
            
        Returns:
            List of dictionaries with predictions
        """
        results = []
        
        for headline in headlines:
            prediction = self.predict_single(headline)
            results.append(prediction)
        
        return results
    
    def predict_from_csv(self, csv_path: str, headline_column='headline'):
        """
        Predict returns for headlines from CSV file
        
        Args:
            csv_path: Path to CSV file with headlines
            headline_column: Name of column containing headlines
            
        Returns:
            DataFrame with original data + predictions
        """
        df = pd.read_csv(csv_path)
        
        # Get predictions
        headlines = df[headline_column].tolist()
        predictions = self.predict_batch(headlines)
        
        # Create results dataframe
        pred_df = pd.DataFrame(predictions)
        
        # Combine with original data
        result_df = pd.concat([df, pred_df[['ret_1d', 'ret_2d', 'ret_3d']]], axis=1)
        result_df = result_df.rename(columns={
            'ret_1d': 'pred_ret_1d',
            'ret_2d': 'pred_ret_2d',
            'ret_3d': 'pred_ret_3d'
        })
        
        return result_df


class PredictionAnalyzer:
    """Analyze predictions and provide insights"""
    
    @staticmethod
    def analyze_predictions(predictions_df):
        """
        Analyze predictions vs actual values (if available)
        
        Args:
            predictions_df: DataFrame with predictions and optionally actual values
            
        Returns:
            Dictionary with analysis statistics
        """
        analysis = {}
        
        # Check if we have actual values
        has_actual = 'ret_1d' in predictions_df.columns
        
        for horizon in ['1d', '2d', '3d']:
            pred_col = f'pred_ret_{horizon}'
            
            if pred_col not in predictions_df.columns:
                continue
            
            pred_stats = {
                'mean': float(predictions_df[pred_col].mean()),
                'std': float(predictions_df[pred_col].std()),
                'min': float(predictions_df[pred_col].min()),
                'max': float(predictions_df[pred_col].max()),
                'median': float(predictions_df[pred_col].median()),
            }
            
            if has_actual:
                actual_col = f'ret_{horizon}'
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(predictions_df[actual_col], predictions_df[pred_col])
                mae = mean_absolute_error(predictions_df[actual_col], predictions_df[pred_col])
                r2 = r2_score(predictions_df[actual_col], predictions_df[pred_col])
                
                pred_stats['MSE'] = float(mse)
                pred_stats['MAE'] = float(mae)
                pred_stats['R2'] = float(r2)
                pred_stats['RMSE'] = float(np.sqrt(mse))
            
            analysis[f'ret_{horizon}'] = pred_stats
        
        return analysis
    
    @staticmethod
    def get_signal_direction(predictions_df, threshold=0.005):
        """
        Get trading signals (UP/DOWN/NEUTRAL) based on predictions
        
        Args:
            predictions_df: DataFrame with predictions
            threshold: Threshold for signal generation
            
        Returns:
            DataFrame with signal column
        """
        signals = {
            '1d': [],
            '2d': [],
            '3d': []
        }
        
        for horizon in ['1d', '2d', '3d']:
            pred_col = f'pred_ret_{horizon}'
            
            if pred_col not in predictions_df.columns:
                continue
            
            for pred in predictions_df[pred_col]:
                if pred > threshold:
                    signals[horizon].append('UP')
                elif pred < -threshold:
                    signals[horizon].append('DOWN')
                else:
                    signals[horizon].append('NEUTRAL')
        
        result_df = predictions_df.copy()
        for horizon in signals:
            result_df[f'signal_{horizon}'] = signals[horizon]
        
        return result_df


def example_usage():
    """Example of how to use the predictor"""
    
    # Initialize predictor
    predictor = StockPricePredictor('model_weights.pth')
    
    # Example 1: Predict single headline
    headline = "Company reports strong quarterly earnings"
    prediction = predictor.predict_single(headline)
    print("Single Prediction:")
    print(prediction)
    print()
    
    # Example 2: Predict batch
    headlines = [
        "Stock rallies on positive news",
        "Market faces headwinds due to economic concerns",
        "Company announces new product launch"
    ]
    batch_predictions = predictor.predict_batch(headlines)
    print("Batch Predictions:")
    for pred in batch_predictions:
        print(pred)
    print()
    
    # Example 3: Predict from CSV
    # result_df = predictor.predict_from_csv('new_headlines.csv', headline_column='headline')
    # result_df.to_csv('predictions_with_signals.csv', index=False)
    
    # Analyze predictions
    # analyzer = PredictionAnalyzer()
    # analysis = analyzer.analyze_predictions(result_df)
    # print("Analysis:", analysis)


if __name__ == '__main__':
    example_usage()
