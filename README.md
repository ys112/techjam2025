# 🛡️ Review Quality Assessment System - TechJam 2025

## 📋 Project Overview

This project implements an ML-based system to automatically evaluate the quality and relevancy of Google location reviews for TechJam 2025. The system leverages both Large Language Models (LLMs) and traditional machine learning techniques (XGBoost + Transformers) to detect policy violations in business reviews, including advertisements, irrelevant content, and fake rants.


### Architecture

- **Data Pipeline**: South Dakota Google Reviews dataset processing and cleaning
- **LLM Classification**: Structured JSON output with confidence scoring
- **Evaluation Framework**: Inter-model agreement analysis and performance metrics
- **Batch Processing**: Scalable processing with automatic retry and error handling
- **Alternative ML Approach**: XGBoost + Transformer pipeline for faster inference and local deployment

## Setup Instructions

### Prerequisites

- Python 3.13.0
- OpenAI API key (for LLM approach)
- Git

### Step 1: Environment Setup

```bash
git clone https://github.com/ys112/techjam2025
cd techjam2025
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Create .env file
OPENAI_API_KEY=your_key_here
SAMPLE_SIZE=10000
```

### Step 3: Run Data Pipeline

1. Open jupyter notebook `notebooks/data_pipeline_LLM_approach.ipynb`
2. Run all cells sequentially

### Step 4: Alternative Approaches

1. Open jupyter notebook `notebooks/xgboost_transformer_model_run.ipynb` for XGBoost + Transformer approach
2. Run all cells sequentially

3. Open jupyter notebook `notebooks/data_labeling_setfit.ipynb` for SetFit labeling approach

4. Run all cells sequentially


## 🔧 Technical Implementation

### Core Technologies

- **Python 3.13**: Main programming language
- **pandas & numpy**: Data processing and analysis
- **OpenAI API**: LLM inference and batch processing
- **scikit-learn**: Evaluation metrics and analysis
- **matplotlib & seaborn**: Data visualization
- **Jupyter**: Interactive development environment

### Key Components

1. **BatchReviewClassifier**: Main class for LLM batch processing
2. **ReviewAnalyzer**: Comprehensive analysis and visualization
3. **ModelComparisonAnalyzer**: Inter-model agreement analysis
4. **Data Pipeline**: Automated data cleaning and preprocessing

### API Usage

- **OpenAI Batch API**: Cost-effective large-scale processing

## 🔄 Workflow

### 1. Data Source

- **Primary Dataset**: South Dakota Google Reviews from [Google Local Data](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/)
- **Input Files**:
  - [`data/review_South_Dakota.json.gz`](data/review_South_Dakota.json.gz) - Compressed review data
  - [`data/meta_South_Dakota.json.gz`](data/meta_South_Dakota.json.gz) - Business metadata

### 2. Data Pipeline

#### Input

- **Review Data**: User reviews with text, ratings, timestamps, and metadata
- **Business Metadata**: Business names, categories, locations, and ratings

#### Processing Steps

1. **Data Loading**: Parse compressed JSON files
2. **Standardization**: Normalize column names and data types
3. **Cleaning**: Remove missing essential fields (rating, time, gmap_id)
4. **Feature Engineering**: Create boolean features (has_pics, price_level)
5. **Merging**: Combine reviews with business metadata
6. **Text Filtering**: Remove reviews without text content

#### Output

- **Merged Dataset**: [`data/cleaned_reviews_data.csv`](data/cleaned_reviews_data.csv)
- **Columns**: `user_id`, `name`, `time`, `rating`, `text`, `pics`, `gmap_id`, `business_name`, `category`, `avg_rating`, etc.

### 3. LLM Classification Pipeline

#### Process Flow

```
Text Reviews → Batch Processing → LLM Classification → Results Aggregation
```

#### Implementation

- **Batch Creation**: Split data into manageable chunks (2,500 reviews per batch)
- **Prompt Engineering**: Structured prompts for policy violation detection
- **Sequential Processing**: Automatic batch submission with rate limiting
- **JSON Schema Validation**: Enforced structured output format
- **Error Handling**: Automatic retry and default value assignment

### Expected Outputs
- `classified_reviews_gpt-5-nano.parquet` - Main classification results
- `classified_reviews_gpt-4o-mini.parquet` - Comparison model results
- `model_comparison_report_*.txt` - Detailed comparison analysis
- Various visualization plots and performance metrics in notebook


### 4. Model Comparison and Evaluation

#### Inter-Model Agreement Analysis

- **Models Compared**: GPT-5-nano vs GPT-4o-mini
- **Metrics**: Agreement rate, Cohen's Kappa, confidence correlation
- **Disagreement Analysis**: Identification of edge cases for manual review

#### Evaluation Metrics

- **Agreement Rate**: Percentage of identical classifications
- **Cohen's Kappa**: Inter-rater reliability measurement
- **Confidence Correlation**: Correlation between model confidence scores
- **Flagging Rates**: Percentage of reviews flagged by each model

### 5. Alternative Approaches
- **XGBoost + Transformer**: Traditional ML pipeline using embeddings and XGBoost classifier (see `notebooks/xgboost_transformer_model_run.ipynb`)
- **SetFit Labeling**: Semi-supervised approach using SetFit for data labeling (see `notebooks/data_labeling_setfit.ipynb`)


## 📁 Repository Structure

```
techjam2025/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
├── data/                              # Data directory
│   ├── review_South_Dakota.json.gz    # Raw review data
│   ├── meta_South_Dakota.json.gz      # Business metadata
│   ├── cleaned_reviews_data.csv       # Processed dataset
│   └── *.parquet                      # Classification results
│   └── *.csv                          # Classification results
├── notebooks/                         # Jupyter notebooks
│   |── draft/                         # Initial drafts and experiments
│   |   └── LLM approach draft.ipynb   # Early LLM pipeline draft with 10k reviews result
│   ├── .env                           # Environment variables
│   ├── data_pipeline_LLM_approach.ipynb     # Main LLM pipeline
│   ├── xgboost_transformer_model_run.ipynb  # Alternative ML approach
│   ├── data_labeling_setfit.ipynb           # SetFit labeling approach
│   └── data_labeling_wl.ipynb
└── failed_notebooks/                 # Experimental notebooks
```



## 🏆 Key Achievements

- ✅ **Scalable Processing**: Successfully processed 10,000+ reviews using batch API
- ✅ **High Accuracy**: Achieved >90% inter-model agreement across all violation types
- ✅ **Cost-Effective**: Utilized batch processing for 50% cost reduction
- ✅ **Robust Pipeline**: Automated error handling and recovery mechanisms
- ✅ **Comprehensive Evaluation**: Multi-model comparison with statistical analysis
- ✅ **Production-Ready**: Structured output format suitable for real-world deployment

---

**Note**: This implementation demonstrates a production-ready approach to review quality assessment using state-of-the-art LLMs. The pipeline is designed for scalability, accuracy, and cost-effectiveness in real-world deployments.
