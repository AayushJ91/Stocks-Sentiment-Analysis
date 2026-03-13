# Stock Price Prediction Model - Complete Guide

A deep learning model for predicting stock returns (ret_1d, ret_2d, ret_3d) based on news headlines using BERT and multi-task learning.

## 📁 Project Files

### Core Training Files
- **`stock_prediction_model.py`** - Main model training script with:
  - `StockNewsDataset`: Custom PyTorch dataset for news/returns data
  - `MultiTaskStockPredictor`: BERT-based multi-task learning model
  - `StockDatasetLoader`: Handles loading all 50 aligned CSVs
  - `ModelTrainer`: Training and evaluation class
  - `main()`: Complete training pipeline

- **`Stock_Price_Prediction_Model.ipynb`** - Interactive Jupyter notebook with:
  - Data loading and exploration
  - Data preprocessing
  - Model definition and training
  - Evaluation and visualization
  - Prediction on new headlines

### Inference & Utilities
- **`prediction_inference.py`** - Inference utilities with:
  - `StockPricePredictor`: Load trained model and make predictions
  - `PredictionAnalyzer`: Analyze predictions vs actuals
  - Example usage functions

## 🚀 Quick Start

### Option 1: Run Training Script
```bash
python stock_prediction_model.py
```
This will:
1. Load all aligned CSVs from `Data/processed/aligned/`
2. Split data into train/val/test (70/15/15)
3. Train model with early stopping
4. Evaluate on test set
5. Save:
   - `model_weights.pth` - Trained model weights
   - `best_model.pth` - Best checkpoint
   - `training_results.json` - Training metrics
   - `test_predictions.csv` - Test set predictions

### Option 2: Run Jupyter Notebook
```bash
jupyter notebook Stock_Price_Prediction_Model.ipynb
```
Run cells sequentially to visualize each step.

## 📊 Model Architecture

**Multi-Task Learning Model:**
```
Input Headlines (Text)
        ↓
     BERT (768D)
        ↓
  Shared Layers (256D)
        ↓
    ┌───┴────┬───────┐
    ↓        ↓       ↓
  Head_1d  Head_2d Head_3d
    ↓        ↓       ↓
  ret_1d  ret_2d   ret_3d
```

**Key Features:**
- Transfer learning from `bert-base-uncased`
- Shared feature extraction
- Task-specific prediction heads
- Multi-task loss = Loss_1d + Loss_2d + Loss_3d
- MSE loss for regression
- AdamW optimizer (lr=2e-5)

## 📈 Data Format

**Expected CSV structure** (from `Data/processed/aligned/`):
```
headline, news_time, event_date, close_T, ret_1d, ret_2d, ret_3d, stock
```

**Example:**
```
"Tech company reports strong earnings",2025-01-15 10:30:00,2025-01-15,100.50,0.025,0.035,0.045
```

## 🔧 Making Predictions

### Method 1: Single Headline
```python
from prediction_inference import StockPricePredictor

predictor = StockPricePredictor('model_weights.pth')
prediction = predictor.predict_single("New headline here")

print(prediction)
# Output: {'headline': '...', 'ret_1d': 0.025, 'ret_2d': 0.035, 'ret_3d': 0.045}
```

### Method 2: Batch Predictions
```python
headlines = [
    "Company A reports positive results",
    "Stock market faces headwinds",
    "New product launch announced"
]
batch_preds = predictor.predict_batch(headlines)
```

### Method 3: From CSV File
```python
results_df = predictor.predict_from_csv('new_headlines.csv', headline_column='headline')
results_df.to_csv('predictions_output.csv', index=False)
```

## 📊 Outputs

### Training Artifacts
- **model_weights.pth** - Final trained model (all parameters)
- **best_model.pth** - Best model from early stopping
- **training_results.json** - Training history and metrics

### Evaluation Outputs
- **test_predictions.csv** - Predictions on test set with true values
- **model_metrics.json** - Performance metrics (MSE, MAE, RMSE, R²)

### Prediction Outputs
- **pred_ret_1d, pred_ret_2d, pred_ret_3d** - Predicted returns

## 📈 Performance Metrics

For each output (ret_1d, ret_2d, ret_3d), the model reports:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

## 🔄 Training Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| Batch Size | 16 | Adjust based on GPU memory |
| Learning Rate | 2e-5 | Standard for BERT fine-tuning |
| Epochs | 10 | With early stopping (patience=3) |
| Train/Val/Test Split | 70/15/15 | Time-ordered to avoid leakage |
| Max Sequence Length | 128 | BERT tokenizer padding length |
| Dropout | 0.1 | Prevents overfitting |

## 📋 Requirements

```
torch>=2.0.0
transformers>=4.25.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib
seaborn
```

Install with:
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
```

## ⚠️ Important Notes

### Data Leakage Prevention
- Data split by time (not shuffled) to prevent temporal leakage
- Each stock's data kept separate until combining
- Validation set comes after training set chronologically

### Model Limitations
- Trained only on news headlines (not price/volume data)
- Performance depends on data quality and breadth
- Single model across all 50 stocks (consider stock-specific models for production)

### Optimization Tips
1. **More data**: Collect more historical headlines
2. **Feature engineering**: Add sentiment scores, keyword extraction
3. **Ensemble**: Train separate models for each stock
4. **Hyperparameter tuning**: Try different learning rates, batch sizes
5. **Data normalization**: Normalize returns per stock to handle different price ranges

## 🐛 Troubleshooting

### GPU Memory Issues
- Reduce batch_size from 16 to 8 or 4
- Use CPU: `device = torch.device('cpu')`

### Model Not Converging
- Try smaller learning rate (1e-5)
- Train for more epochs
- Check data preprocessing for anomalies

### Low Performance
- Verify data format and alignment
- Check for missing values in headlines/returns
- Ensure sufficient training data per stock

## 📝 License & Citation

This model uses:
- BERT (Devlin et al., 2018): https://arxiv.org/abs/1810.04805
- PyTorch: https://pytorch.org/
- HuggingFace Transformers: https://huggingface.co/

## 📞 Support

For issues or questions:
1. Check data format matches expected structure
2. Verify all required CSV files exist in `Data/processed/aligned/`
3. Review error messages in training logs
4. Test with sample data first

---

**Model Version**: 1.0  
**Last Updated**: 2026-03-13
