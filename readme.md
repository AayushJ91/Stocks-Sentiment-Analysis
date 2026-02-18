StockProject/
│
├── data/
│   ├── reliance_price.csv
│   ├── reliance_news.csv
│
├── scraping/
│   ├── fetch_prices.py
│   ├── fetch_news.py
│
├── preprocessing/
│   └── label_generator.py

Got the historical news for Reliance.NS

data/
├── raw/
│   ├── moneycontrol/
│   │   ├── RELIANCE.json
│   │   ├── TCS.json
│   │   ├── INFY.json
│   │   ├── HDFCBANK.json
│   │   └── ...
│   └── yahoo/
│       ├── RELIANCE_prices.csv
│       ├── TCS_prices.csv
│       └── ...
│
├── processed/
│   ├── aligned/
│   │   ├── RELIANCE_aligned.csv
│   │   ├── TCS_aligned.csv
│   │   └── ...
│
├── features/
│   ├── RELIANCE_features.parquet
│   └── ...
│
├── models/
│   ├── bert_sentiment/
│   └── ...
│
└── logs/
    ├── scraping.log
    └── alignment.log
    