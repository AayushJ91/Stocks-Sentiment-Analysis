#!/usr/bin/env python3
"""
Quick Start Script for Stock Price Prediction Model
Run this script to train the model end-to-end
"""

import os
import sys
from pathlib import Path

def main():
    print("="*70)
    print("STOCK PRICE PREDICTION MODEL - QUICK START")
    print("="*70)
    print()
    
    # Check if data exists
    data_path = Path('Data/processed/aligned')
    if not data_path.exists():
        print("❌ ERROR: Data directory not found!")
        print(f"   Expected: {data_path.resolve()}")
        sys.exit(1)
    
    csv_files = list(data_path.glob('*_aligned.csv'))
    if not csv_files:
        print("❌ ERROR: No aligned CSV files found!")
        print(f"   Expected CSV files in: {data_path.resolve()}")
        sys.exit(1)
    
    print(f"✓ Data directory found with {len(csv_files)} stock(s)")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        import torch
        import pandas
        import numpy
        import transformers
        from sklearn import metrics
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install torch transformers pandas numpy scikit-learn matplotlib seaborn")
        sys.exit(1)
    
    print()
    
    # Ask user for mode
    print("Select training mode:")
    print("  1. Quick training (5 epochs, default batch size)")
    print("  2. Full training (10 epochs, optimized)")
    print("  3. Notebook (open Jupyter)")
    print()
    
    choice = input("Enter choice (1/2/3/default=1): ").strip() or "1"
    
    if choice == "3":
        os.system("jupyter notebook Stock_Price_Prediction_Model.ipynb")
    elif choice in ["1", "2"]:
        print()
        print("Starting training...")
        print("="*70)
        
        # Import and run training
        from stock_prediction_model import main as train_main
        
        try:
            train_main()
            print()
            print("="*70)
            print("✓ TRAINING COMPLETE!")
            print("="*70)
            print()
            print("Results saved:")
            print("  • model_weights.pth - Trained model")
            print("  • training_results.json - Metrics")
            print("  • test_predictions.csv - Test set predictions")
            print()
            print("Next steps:")
            print("  1. Load model: predictor = StockPricePredictor('model_weights.pth')")
            print("  2. Make predictions: pred = predictor.predict_single('headline')")
            print("  3. See MODEL_GUIDE.md for details")
        
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Invalid choice!")
        sys.exit(1)

if __name__ == '__main__':
    main()
