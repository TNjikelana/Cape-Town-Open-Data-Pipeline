# Cape-Town-Open-Data-Pipeline
## Data Science Pipeline

A Python data science pipeline demonstrating **data cleaning**, **data transformation**, and **data exploration & visualisation** on real City of Cape Town open data.




## Overview

The implemented cleaning stage reads the raw dataset and writes a processed file:

```
raw data  →  1. Cleaning
(data/raw)    (data/processed/)
```

| Stage | What it covers |
|---|---|
| **1. Data cleaning** | Parse dates and percentage values, standardize the growth column name, and flag suspicious values without clipping or replacing measurements |
| **2. Data transformation** | Planned; not yet implemented |
| **3. Exploration & visualisation** | Planned; not yet implemented |

The cleaning logic is a function that takes a DataFrame and returns a cleaned DataFrame. Run `python main.py` from the project directory to write `data/processed/system_energy_clean.csv`. Input and output paths are resolved relative to `main.py`, so it can also be run from another working directory.

## Data

| | |
|---|---|
| **Source** | [City of Cape Town Open Data Portal](https://odp.capetown.gov.za/) |
| **Dataset** | System Energy (`System_Energy.csv`) (for now...) |
| **Size** | 107 monthly records, July 2017 to May 2026, 10 columns |
| **Content** | Monthly electricity system figures in kWh (Eskom supply, imports, outputs, total system output, growth rates) |

Raw data is stored in `data/raw/` and is never edited. The pipeline preserves source measurements, including negative values and published totals, while adding flags for values that may need review. The first 66 missing `F12_Month_Ave` entries are retained as missing values.

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
