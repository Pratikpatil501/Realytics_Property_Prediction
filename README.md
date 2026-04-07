# Realytics — Mumbai Property Intelligence Platform

AI-powered property valuation, liveability scoring, investment analysis and price forecasting for Mumbai MMR.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Retrain model on latest data
python train_model.py

# 3. Start the server
python app.py

# 4. Open browser
# http://localhost:5000  (API)
# Open index.html in browser (Frontend)
```

## Project Structure

```
├── index.html              # Full frontend (single-file, no build needed)
├── app.py                  # Flask API server
├── train_model.py          # Model training script (XGBoost / GradientBoosting)
├── model.pkl               # Trained model (R² 0.918 on 90K rows)
├── encoders.pkl            # Label encoders for categorical features
├── meta.json               # Region/type/status lists for dropdowns
├── Mumbai House Prices.csv # 90,000 property listings dataset
├── users.db                # SQLite user database
├── requirements.txt        # Python dependencies
├── model_report.txt        # Training metrics report
├── scraper/
│   ├── scrape_listings.py  # MagicBricks + 99acres scraper (requests)
│   ├── scrape_selenium.py  # Anti-bot resistant scraper (Selenium)
│   └── README.md           # Scraper usage guide
└── README.md
```

## Model Details

| Metric | Value |
|--------|-------|
| Training Data | 90,000 listings |
| Regions | 208 Mumbai MMR localities |
| R² Score | 0.9178 |
| Features | bhk, type, area, region, status, age |
| Algorithm | GradientBoosting (or XGBoost if installed) |

## Features

- **Price Prediction** — ML-powered valuation with liveability & investment scores
- **Property Compare** — Side-by-side analysis of two properties
- **Price Zones** — Interactive Mumbai heat map with 30+ colour-coded areas
- **Infrastructure Tracker** — 12 mega-projects with real estate impact data
- **Mortgage Calculator** — EMI, amortisation, total interest breakdown
- **AI Advisor** — Chat with Claude about Mumbai real estate (requires Anthropic API)
- **Auth System** — Register/login with search history tracking

## Updating Data

```bash
# Scrape latest listings from MagicBricks & 99acres
cd scraper
pip install requests beautifulsoup4 pandas
python scrape_listings.py --source both --pages 5 --output ../new_data.csv

# Merge with existing data
cd ..
python -c "
import pandas as pd
old = pd.read_csv('Mumbai House Prices.csv')
new = pd.read_csv('new_data.csv')
merged = pd.concat([old, new]).drop_duplicates(subset=['bhk','type','locality','area','price','region'])
merged.to_csv('Mumbai House Prices.csv', index=False)
print(f'Merged: {len(merged)} rows')
"

# Retrain
python train_model.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/meta` | Region/type/status lists |
| POST | `/predict` | Price prediction + analysis |
| POST | `/compare` | Compare two properties |
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Sign in |
| GET | `/auth/profile` | User profile + search history |
