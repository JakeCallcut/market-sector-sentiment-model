# Social Sentiment Analysis for Sector-Level Stock Market Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-BSD–3-green)
![Status](https://img.shields.io/badge/Status-Complete-orange)

This repository contains the code for an academic dissertation project investigating the relationship between social media sentiment and US equity market sector returns.

## Project Details

This NLP & ETL pipeline extracts data from Yahoo! Finance and from provided social media posts, uses finBERT to perform finance-specific sentiment analysis, and applies machine learning techniques to predict future market returns.

#### Models used:
- FinBERT
- Multinomial Logistic Regression
- Random Forest
- Gradient Boosting

#### Securities Examined:
- SPY (S&P 500 ETF Benchmark)
- XLC (Communication Services Select Sector SPDR ETF)
- XLY (Consumer Discretionary Select Sector SPDR ETF)
- XLP (Consumer Staples Select Sector SPDR ETF)
- XLE (Energy Select Sector SPDR ETF)
- XLF (Financial Select Sector SPDR ETF)
- XLV (Health Care Select Sector SPDR ETF)
- XLI (Industrial Select Sector SPDR ETF)
- XLK (Technology Select Sector SPDR ETF)
- XLB (Materials Select Sector SPDR ETF)
- XLRE (Real Estate Select Sector SPDR ETF)
- XLU (Utilities Select Sector SPDR ETF)
- GLD (SPDR Gold Shares ETF)
- USO (United States Oil Fund)
- VIX (CBOE Volatility Index) (For sentiment comparison only)

#### Dataset Source
https://huggingface.co/datasets/StephanAkkerman/stock-market-tweets-data

## Repository Structure

- `/data` - Contains raw data, processed data, and evaluation results
- `/notebooks` - Contains Exploratory Data Analysis
- `/src` - Contains all source code files including full pipeline run
- `/src/models` - Contains machine learning models
- `/src/scripts` - Contains all scripts used in the pipeline

## Installation & Usage

1. Navigate to project root
2. Install dependencies using:
   
   `pip install -r requirements.txt`

3. Insert raw data into `/data/raw` (ensure data has "created_at", and "text" columns)
4. Run `/src/run_pipeline.py`
5. Run `/notebooks/EDA.ipynb` to visualise results 

## Disclaimers & Licensing

> This project, along with any associated material, is intended solely as an academic investigation of financial relationships and the evaluation of machine learning methods. No content within this repository constitutes financial advice, nor is it intended to incite market speculation.

> This work is not produced, or distributed by a regulated financial institution and does not fall within the scope of the UK Financial Conduct Authority (FCA).

> The project does not seek to influence market behaviour or engage in any activity prohibited under the UK Market Abuse Regulation (UK MAR). 

> All data sources used are publicly available or accessed under appropriate academic or institutional permissions.

> This project is released under the BSD 3-Clause Licence. See the `LICENSE` file for details. 