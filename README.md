# Cape-Town-Open-Data-Pipeline
## Data Science Pipeline

A Python data science pipeline demonstrating **data cleaning**, **data transformation**, and **data exploration & visualisation** on real City of Cape Town open data.




## Overview

Each stage takes the output of the one before it:

```
raw data  →  1. Cleaning  →  2. Transformation  →  3. Exploration & Visualisation
(data/raw)                   (data/processed)         (figures/)
```

| Stage | What it covers |
|---|---|
| **1. Data cleaning** | Parsing dates and timezones, converting percentage text to numbers, handling missing values, flagging outliers and suspicious values, validating totals |
| **2. Data transformation** | Feature engineering (year, month, season), log transforms for skewed data, scaling (min-max vs standardisation), categorical encoding |
| **3. Exploration & visualisation** | Trends over time, seasonality, distributions, correlations, before-vs-after comparisons |

Each stage is written as functions that take a DataFrame and return a DataFrame, so they can be chained, tested, and reused. Dataset-specific choices (column names, thresholds) are kept in one config section at the top of each script.

## Data

| | |
|---|---|
| **Source** | [City of Cape Town Open Data Portal](https://odp.capetown.gov.za/) |
| **Dataset** | System Energy (`System_Energy.csv`) (for now...) |
| **Size** | 107 monthly records, July 2017 to May 2026, 10 columns |
| **Content** | Monthly electricity system figures in kWh (Eskom supply, imports, outputs, total system output, growth rates) |

Raw data is stored in `data/raw/` and is never edited. All cleaning is done in code so the process is reproducible.

> Check the dataset's licence and terms on the portal and add them here.

## Project structure

```
data-science-pipeline/
├── data/
│   ├── raw/                  # original files, untouched
│   └── processed/            # cleaned and transformed outputs
├── notebooks/
│   └── pipeline.ipynb        # walkthrough with explanations and outputs
├── src/
│   ├── clean.py              # stage 1
│   ├── transform.py          # stage 2
│   └── explore.py            # stage 3
├── figures/                  # saved charts
├── main.py                   # runs the full pipeline
├── requirements.txt
└── README.md
```

## Getting started

**Requirements:** Python 3.10 or newer.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/data-science-pipeline.git
cd data-science-pipeline

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

Run the full pipeline from the command line:

```bash
python main.py
```

Or open the notebook for the step-by-step version with explanations:

```bash
jupyter notebook notebooks/pipeline.ipynb
```

In VS Code, open `notebooks/pipeline.ipynb` with the Python and Jupyter extensions installed.

## Key findings

*TBD

## Reflection

*TBD

## Tools

Python, pandas, NumPy, Matplotlib, seaborn, scikit-learn, Jupyter

## Author

Thando Njikelana
