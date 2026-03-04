Stocks Sentiment Analysis
=========================

This project builds an end‑to‑end pipeline that:
- **collects news and price data for stocks** (currently prototyped on `Reliance.NS`),  
- **extracts & scores sentiment / impact for each news headline**,  
- **verifies these scores against actual market moves with a short delay**, and  
- is designed to scale to **100–500+ stocks** across different sectors.

The high‑level workflow follows four main stages:
1. **Data**
2. **Information Extraction & Scoring**
3. **Score Verification**
4. **Visualisation & Result Interpretation**

---

### High‑Level Workflow Diagram

```text
[ 1. DATA ]
    |
    v
[ 2. INFORMATION EXTRACTION & SCORING ]
    |
    v
[ 3. SCORE VERIFICATION (VS. MARKET RETURNS) ]
    |
    v
[ 4. VISUALISATION & RESULT INTERPRETATION ]
```

---

### 1. Data

- **Goal**: Build a clean, rich dataset of text + prices for each stock.

- **Sources (current + planned)**:
  - **Moneycontrol**  
    - Web‑scraped news pages using `requests` + `BeautifulSoup`.  
    - Example: Reliance news stored in `Trials/Data/raw/moneycontrol/RELIANCE.json`.
  - **Yahoo Finance (`yfinance`)**  
    - Historical OHLCV price data.  
    - Example: `Trials/Data/raw/yahoo/Reliance_yahoo.csv`.
  - **Kaggle** (planned)  
    - Extra news / price datasets to extend coverage and history.
  - **Moneycontrol API or similar APIs** (planned)  
    - Cleaner, more scalable way to get structured news.
  - **Hugging Face** (planned)  
    - Datasets and models for language understanding and transfer learning.

- **Target scale**:
  - **Minimum**: 100–200 stocks.
  - **Goal**: ~**500 stocks**, diversified by sector and index.

- **Data cleaning & preprocessing**:
  - Remove:
    - Duplicate articles,
    - Invalid / dead URLs,
    - Articles where date/time cannot be parsed (logged in `Trials/logs/missed_urls.json`).
  - Normalize **publication timestamps**:
    - Convert to a consistent timezone (IST),
    - Store as ISO‑8601 strings for easy parsing.
  - Store structured records (JSON/CSV) with fields like:
    - `news_id`, `headline`, `link`, `source`, `raw_date_text`, `news_datetime`, `article_text`, `status`, etc.
  - Ensure **price series** are clean and continuous on trading days (no gaps on valid sessions).

- **Prototype notebooks**:
  - `Trials/ExtractingData.ipynb`  
    - Scrapes Moneycontrol, extracts headlines and links, parses article timestamps, and saves `RELIANCE.json` and `Reliance.csv`.
  - `Trials/Data_Alignment.ipynb`  
    - Downloads prices from Yahoo Finance (`yfinance`), prepares them for alignment with news.

#### Data Flow Diagram

```text
        +-------------------+
        |  Moneycontrol     |
        |  (web pages/API)  |
        +---------+---------+
                  |
                  v
        +---------------------------+
        |  Scraper (Requests/BS4)  |
        +-------------+------------+
                      |
                      v
        +---------------------------+
        |  Raw News JSON/CSV       |
        |  (Trials/Data/raw/...)   |
        +-------------+------------+
                      |
                      v
        +---------------------------+
        |  Cleaning & Normalising  |
        |  (timestamps, dedupe)    |
        +-------------+------------+
                      |
                      v
        +---------------------------+
        |  Processed News CSV      |
        |  (extracted_news/*.csv)  |
        +-------------+------------+

        +------------------+
        |  Yahoo Finance   |
        |  (yfinance API)  |
        +---------+--------+
                  |
                  v
        +---------------------------+
        |  Price Data CSV           |
        |  (raw/yahoo/*.csv)       |
        +---------------------------+
```

---

### 2. Information Extraction From the Data

- **Goal**: Extract useful **information and scores** from raw text:
  - Identify **positive / negative / neutral** news,
  - Score each news item,
  - Normalise and weight scores by importance.

- **What is extracted**:
  - From `Trials/Data/processed/extracted_news/Reliance.csv` and future multi‑stock files:
    - `news_id`, `headline`, `link`, `source`, `news_datetime`, etc.

- **NLP / scoring pipeline (conceptual)**:
  - **Polarity detection**:
    - Classify each news item as **positive**, **negative**, or **neutral** for that stock.
    - Use **pretrained transformers** (e.g. BERT, FinBERT, or Hugging Face sentiment models) as a base.
  - **Score each news item**:
    - Assign a **continuous score** (e.g. in $[-1, 1]$) representing the strength and direction of impact.
    - Factors:
      - Model’s sentiment/confidence,
      - Presence of strong keywords (`downgrade`, `profit warning`, `record high`, etc.),
      - Source quality / reliability (if available).
  - **Normalise scores**:
    - Make scores comparable over time and across stocks:
      - Rolling Z‑score,
      - Per‑stock min–max scaling,
      - Clipping extreme outliers.
  - **Weight by importance**:
    - **Recency**: newer news gets higher weight.
    - **Uniqueness**: reduce weight for near‑duplicate wire stories.
    - **Context**: earnings days, macro announcements, AGM events, etc.

- **Role of NLP**:
  - NLP models (from **Hugging Face / `transformers`**) convert text into embeddings and/or directly output:
    - Sentiment labels and scores, or
    - A **regressed expected return**, which becomes the news score (see Stage 3).

---

### 3. Verifying the Score

- **Goal**: Test whether news‑based scores are actually predictive of **real market moves**, using a safe time delay.

- **Alignment of news to prices** (implemented in `Trials/Data_Alignment.ipynb`):
  - Load:
    - News: `Trials/Data/processed/extracted_news/Reliance.csv`
    - Prices: `Trials/Data/raw/yahoo/Reliance_yahoo.csv`
  - Convert timestamps:
    - `news_datetime` → timezone‑aware `datetime`,
    - Price `Date` → `trade_date` (plain date) on the trading calendar.
  - **Map each news item to a trading day**:
    - If news time is **after market close** (e.g. after 15:30 IST), move it to the **next trading day**.
    - If the mapped date is a **holiday/weekend**, roll forward to the next valid trading day.
  - **Compute future returns**:
    - For each `event_date`:
      - 1‑day return:  
        $r_{1d} = \dfrac{Close_{T+1} - Close_T}{Close_T}$
      - 2‑day return:  
        $r_{2d} = \dfrac{Close_{T+2} - Close_T}{Close_T}$
      - 3‑day return:  
        $r_{3d} = \dfrac{Close_{T+3} - Close_T}{Close_T}$
    - Save aligned data to:  
      `Trials/Data/processed/aligned/Reliance_aligned.csv`  
      (columns: `headline`, `news_time`, `event_date`, `close_T`, `ret_1d`, `ret_2d`, `ret_3d`, etc.)

- **Delay for safety (1–2 days)**:
  - Scores are verified with a **1–2 trading day delay**:
    - Reduces noise from intraday swings,
    - Resembles a realistic trading scenario: act after you see the news and then observe performance over the next few days.

- **Model verification / learning a score** (in `Trials/model.ipynb`):
  - Use a **BERT‑based regression model** to predict **next‑day return** from the headline:
    - Inputs: `headline` text.
    - Target: `ret_1d` from `Reliance_aligned.csv`.
  - Steps:
    - Tokenise with `AutoTokenizer("bert-base-uncased")`.
    - Convert labels to tensors (`torch.float32`).
    - Fine‑tune `BertForSequenceClassification` with:
      - `problem_type="regression"`, `num_labels=1`.
    - Train using `transformers.Trainer` with train/validation split.
  - Interpretation:
    - The model’s **predicted return** becomes a **data‑driven news score**, and its performance against true returns is how we **verify** the usefulness of the score.

#### Alignment + Model Diagram

```text
[Processed News] + [Price Data]
                |
                v
       +------------------------+
       |  Alignment Logic       |
       |  (map news -> T, T+1)  |
       +-----------+------------+
                   |
                   v
       +------------------------+
       |  Aligned Dataset      |
       |  (headline, ret_1d,   |
       |   ret_2d, ret_3d...)  |
       +-----------+-----------+
                   |
                   v
       +------------------------+
       |  BERT / NLP Model      |
       |  (predict returns or   |
       |   sentiment scores)    |
       +-----------+-----------+
                   |
                   v
       +------------------------+
       |  Evaluation &          |
       |  Visualisations        |
       +------------------------+
```

---

### 4. Making Visualizations and Depicting Results

- **Goal**: Show clearly how news scores and market movements relate, and how well the model behaves.

- **Possible visualisations**:
  - **Time‑series overlays**:
    - Stock price vs. **aggregated news score** over time (e.g. rolling sum / average of scores).
    - Mark major positive/negative news events on the price chart.
  - **Event‑study style plots**:
    - Average return around high‑score vs. low‑score news.
    - Distribution of returns conditioned on sentiment buckets (strong positive / neutral / strong negative).
  - **Model performance plots**:
    - Scatter of **predicted score vs. realised return**.
    - Calibration curves (do higher scores correspond to higher average returns?).
  - **Cross‑section comparisons** (once scaled to many stocks):
    - Compare signal performance by:
      - Sector,
      - Market‑cap bucket,
      - Index membership (e.g. NIFTY50 vs midcaps).

- **Tools**:
  - Python plotting libraries: `matplotlib`, `seaborn`, `plotly`, etc.
  - Jupyter notebooks for interactive analysis and reporting.

---

### Repository Sketch (current prototype)

- **Top‑level**
  - `readme.md` – this workflow documentation.
  - `Stocks Sentiment Analysis WorkFlow.docx` – additional project notes / diagrams.

- **Notebooks (`Trials/`)**
  - `ExtractingData.ipynb`  
    - Scrape Moneycontrol, extract headlines and links, parse dates, save to JSON/CSV.
  - `Data_Alignment.ipynb`  
    - Align news timestamps with Yahoo Finance prices, compute 1/2/3‑day returns.
  - `model.ipynb`  
    - BERT‑based regression model on aligned news/return data.

- **Data (`Trials/Data/`)**
  - `raw/`
    - `moneycontrol/RELIANCE.json` – raw scraped news data for Reliance.
    - `yahoo/Reliance_yahoo.csv` – raw daily prices from Yahoo Finance.
  - `processed/`
    - `extracted_news/Reliance.csv` – cleaned and structured news records.
    - `aligned/Reliance_aligned.csv` – news aligned to future price moves.
  - `logs/`
    - `missed_urls.json` – URLs where scraping or date extraction failed.

- **Scaling to many stocks**
  - Intended general pattern:
    - `raw/moneycontrol/{TICKER}.json`
    - `raw/yahoo/{TICKER}_yahoo.csv`
    - `processed/extracted_news/{TICKER}.csv`
    - `processed/aligned/{TICKER}_aligned.csv`

---




