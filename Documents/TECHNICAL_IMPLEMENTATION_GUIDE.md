# 🔧 Technical Implementation Guide for Beginners

## 🎯 Overview

This guide provides step-by-step technical instructions for building your review quality assessment system. Designed specifically for NLP/LLM beginners.

---

## 📋 Prerequisites & Setup

### Required Software

```bash
# Install Python 3.8+
# Install Git
# Get Google Colab account (recommended for GPU access)
```

### Python Environment Setup

```python
# Create virtual environment
python -m venv hackathon_env
hackathon_env\Scripts\activate  # Windows
# source hackathon_env/bin/activate  # Mac/Linux

# Install required packages
pip install pandas numpy matplotlib seaborn
pip install transformers torch
pip install huggingface_hub
pip install streamlit
pip install scikit-learn
pip install openai  # optional, for comparison
```

---

## 📊 Data Handling & Preprocessing

### Step 1: Load and Explore Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Google Reviews dataset
def load_data():
    """
    Load Google Reviews dataset from Kaggle
    Expected columns: review_text, rating, business_name, etc.
    """
    # Download from: https://www.kaggle.com/datasets/denizbilginn/google-maps-restaurant-reviews
    df = pd.read_csv('google_reviews.csv')

    # Basic exploration
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample review: {df['review_text'].iloc[0]}")

    return df

# Explore data distribution
def explore_data(df):
    """Basic data exploration"""
    # Review length distribution
    df['review_length'] = df['review_text'].str.len()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(df['review_length'], bins=50)
    plt.title('Review Length Distribution')

    plt.subplot(1, 3, 2)
    plt.hist(df['rating'], bins=5)
    plt.title('Rating Distribution')

    plt.subplot(1, 3, 3)
    df['word_count'] = df['review_text'].str.split().str.len()
    plt.hist(df['word_count'], bins=50)
    plt.title('Word Count Distribution')

    plt.tight_layout()
    plt.show()
```

### Step 2: Data Preprocessing

```python
import re
from typing import List, Dict

def clean_text(text: str) -> str:
    """Clean and normalize review text"""
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = ' '.join(text.split())

    # Keep basic cleaning simple for now
    return text

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract useful features from reviews"""

    # Text features
    df['review_length'] = df['review_text'].str.len()
    df['word_count'] = df['review_text'].str.split().str.len()
    df['exclamation_count'] = df['review_text'].str.count('!')
    df['caps_ratio'] = df['review_text'].apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)

    # URL detection (advertisement indicator)
    df['has_url'] = df['review_text'].str.contains(r'http[s]?://|www\.', regex=True, na=False)

    # Phone number detection (advertisement indicator)
    df['has_phone'] = df['review_text'].str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', regex=True, na=False)

    # Excessive capitalization (rant indicator)
    df['excessive_caps'] = df['caps_ratio'] > 0.3

    # Very short reviews (potential spam)
    df['very_short'] = df['word_count'] < 5

    # Very long reviews (potential rants)
    df['very_long'] = df['word_count'] > 200

    return df

def create_sample_labels(df: pd.DataFrame, n_samples: int = 200) -> pd.DataFrame:
    """Create sample labels for validation (manual labeling)"""

    # Sample diverse reviews for manual labeling
    sample_df = df.sample(n=n_samples, random_state=42)

    # Create template for manual labeling
    sample_df['is_advertisement'] = None  # Fill manually
    sample_df['is_irrelevant'] = None    # Fill manually
    sample_df['is_fake_rant'] = None     # Fill manually

    # Save for manual labeling
    sample_df[['review_text', 'is_advertisement', 'is_irrelevant', 'is_fake_rant']].to_csv('manual_labels.csv', index=False)

    print(f"Saved {n_samples} reviews for manual labeling in 'manual_labels.csv'")
    print("Please label these manually before proceeding to model training")

    return sample_df
```

---

## 🤖 Model Implementation

### Step 3: Prompt Engineering Approach

```python
from transformers import pipeline
import torch

class ReviewClassifier:
    def __init__(self, model_name: str = "google/gemma-3-12b-it"):
        """Initialize the review classifier with specified model"""
        self.model_name = model_name
        self.classifier = None
        self._setup_model()

    def _setup_model(self):
        """Setup the Hugging Face model"""
        try:
            # Use text-generation pipeline for LLM-based classification
            self.classifier = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            # Fallback to smaller model
            self.classifier = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium"
            )

    def create_prompts(self) -> Dict[str, str]:
        """Create prompts for each policy violation type"""

        prompts = {
            'advertisement': """
Task: Determine if this review contains advertisements or promotional content.

Examples of ADVERTISEMENTS:
- "Great food! Visit www.discount-deals.com for coupons!"
- "Call 555-1234 for catering services!"
- "Check out our new location on Main Street!"

Examples of NOT ADVERTISEMENTS:
- "The food was delicious and service was great"
- "I loved the atmosphere, will definitely come back"
- "Terrible experience, would not recommend"

Review to analyze: "{review_text}"

Is this review an ADVERTISEMENT? Answer only YES or NO:""",

            'irrelevant': """
Task: Determine if this review is about the business location being reviewed.

Examples of IRRELEVANT reviews:
- "I love my new phone, but this place is noisy" (about phone, not restaurant)
- "My car broke down on the way here, terrible day" (about car, not business)
- "Politics these days are crazy, anyway the food was ok" (mostly about politics)

Examples of RELEVANT reviews:
- "The pizza was amazing, great service too"
- "Parking was difficult but the experience was worth it"
- "Staff was rude and food was cold"

Review to analyze: "{review_text}"

Is this review IRRELEVANT to the business? Answer only YES or NO:""",

            'fake_rant': """
Task: Determine if this is a rant from someone who likely never visited the place.

Examples of FAKE RANTS:
- "Never been here but heard it's terrible from my friend"
- "I hate this type of business, they're all scams"
- "Looks dirty from the outside, probably awful inside"

Examples of GENUINE reviews (even if negative):
- "I visited yesterday and the service was terrible"
- "Went there for lunch, food was cold and overpriced"
- "Been there multiple times, quality has declined"

Review to analyze: "{review_text}"

Is this a FAKE RANT from someone who likely never visited? Answer only YES or NO:"""
        }

        return prompts

    def classify_single_review(self, review_text: str) -> Dict[str, bool]:
        """Classify a single review for all policy violations"""

        prompts = self.create_prompts()
        results = {}

        for violation_type, prompt_template in prompts.items():
            # Format prompt with review text
            prompt = prompt_template.format(review_text=review_text)

            try:
                # Generate response
                response = self.classifier(
                    prompt,
                    max_new_tokens=10,
                    temperature=0.1,
                    pad_token_id=self.classifier.tokenizer.eos_token_id
                )

                # Extract answer (YES/NO)
                answer = response[0]['generated_text'][len(prompt):].strip().upper()
                results[violation_type] = 'YES' in answer

            except Exception as e:
                print(f"Error classifying {violation_type}: {e}")
                results[violation_type] = False

        return results

    def classify_batch(self, reviews: List[str]) -> List[Dict[str, bool]]:
        """Classify multiple reviews"""
        results = []

        for i, review in enumerate(reviews):
            if i % 10 == 0:
                print(f"Processing review {i+1}/{len(reviews)}")

            result = self.classify_single_review(review)
            results.append(result)

        return results
```

### Step 4: Alternative Simple Approach (Rule-Based)

```python
class SimpleRuleBasedClassifier:
    """Simple rule-based classifier as backup/baseline"""

    def __init__(self):
        self.ad_keywords = [
            'visit', 'website', 'www', 'http', 'call', 'phone', 'discount',
            'deal', 'promo', 'sale', 'coupon', 'special offer'
        ]

        self.irrelevant_indicators = [
            'my phone', 'my car', 'politics', 'weather', 'traffic',
            'my day', 'my life', 'news', 'government'
        ]

        self.fake_rant_indicators = [
            'never been', 'heard it', 'looks like', 'probably',
            'i hate these', 'all these places', 'never visited'
        ]

    def classify_single_review(self, review_text: str) -> Dict[str, bool]:
        """Simple rule-based classification"""
        review_lower = review_text.lower()

        # Advertisement detection
        is_ad = any(keyword in review_lower for keyword in self.ad_keywords)

        # Irrelevant content detection
        is_irrelevant = any(indicator in review_lower for indicator in self.irrelevant_indicators)

        # Fake rant detection
        is_fake_rant = any(indicator in review_lower for indicator in self.fake_rant_indicators)

        return {
            'advertisement': is_ad,
            'irrelevant': is_irrelevant,
            'fake_rant': is_fake_rant
        }
```

---

## 📈 Evaluation & Metrics

### Step 5: Model Evaluation

```python
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.results = {}

    def evaluate_predictions(self, y_true: List[bool], y_pred: List[bool],
                           violation_type: str) -> Dict[str, float]:
        """Evaluate predictions for a specific violation type"""

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary'
        )

        # Calculate accuracy
        accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy
        }

        self.results[violation_type] = metrics
        return metrics

    def plot_confusion_matrix(self, y_true: List[bool], y_pred: List[bool],
                            violation_type: str):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not Violation', 'Violation'],
                   yticklabels=['Not Violation', 'Violation'])
        plt.title(f'Confusion Matrix - {violation_type.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def print_classification_report(self):
        """Print comprehensive classification report"""
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS SUMMARY")
        print("="*50)

        for violation_type, metrics in self.results.items():
            print(f"\n{violation_type.upper()} DETECTION:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1_score']:.3f}")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")

        # Overall performance
        avg_f1 = np.mean([m['f1_score'] for m in self.results.values()])
        print(f"\nOVERALL AVERAGE F1-SCORE: {avg_f1:.3f}")

def run_evaluation_pipeline(classifier, test_data: pd.DataFrame):
    """Complete evaluation pipeline"""

    evaluator = ModelEvaluator()

    # Get predictions
    reviews = test_data['review_text'].tolist()
    predictions = classifier.classify_batch(reviews)

    # Evaluate each violation type
    violation_types = ['advertisement', 'irrelevant', 'fake_rant']

    for vtype in violation_types:
        if f'is_{vtype}' in test_data.columns:
            y_true = test_data[f'is_{vtype}'].tolist()
            y_pred = [pred[vtype] for pred in predictions]

            # Calculate metrics
            metrics = evaluator.evaluate_predictions(y_true, y_pred, vtype)

            # Plot confusion matrix
            evaluator.plot_confusion_matrix(y_true, y_pred, vtype)

    # Print summary
    evaluator.print_classification_report()

    return evaluator.results
```

---

## 🎮 Demo Interface

### Step 6: Create Interactive Demo

```python
import streamlit as st

def create_streamlit_demo():
    """Create Streamlit demo application"""

    st.title("🛡️ Review Quality Assessment System")
    st.write("Detect policy violations in location-based reviews")

    # Initialize classifier
    @st.cache_resource
    def load_classifier():
        return ReviewClassifier()

    classifier = load_classifier()

    # Input methods
    input_method = st.radio(
        "Choose input method:",
        ["Single Review", "Upload File", "Batch Examples"]
    )

    if input_method == "Single Review":
        # Single review input
        review_text = st.text_area(
            "Enter a review to analyze:",
            height=100,
            placeholder="Type or paste a review here..."
        )

        if st.button("Analyze Review") and review_text:
            with st.spinner("Analyzing..."):
                result = classifier.classify_single_review(review_text)

            # Display results
            st.subheader("Analysis Results:")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Advertisement",
                    "❌ VIOLATION" if result['advertisement'] else "✅ OK"
                )

            with col2:
                st.metric(
                    "Irrelevant Content",
                    "❌ VIOLATION" if result['irrelevant'] else "✅ OK"
                )

            with col3:
                st.metric(
                    "Fake Rant",
                    "❌ VIOLATION" if result['fake_rant'] else "✅ OK"
                )

    elif input_method == "Upload File":
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV file with reviews",
            type=['csv']
        )

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} reviews")

            if st.button("Process All Reviews"):
                with st.spinner("Processing reviews..."):
                    reviews = df['review_text'].tolist()
                    results = classifier.classify_batch(reviews)

                # Add results to dataframe
                for i, result in enumerate(results):
                    for vtype, violation in result.items():
                        df.loc[i, f'has_{vtype}'] = violation

                # Display summary
                st.subheader("Processing Summary:")

                total_ads = df['has_advertisement'].sum()
                total_irrelevant = df['has_irrelevant'].sum()
                total_fake = df['has_fake_rant'].sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Advertisements", total_ads)
                with col2:
                    st.metric("Irrelevant", total_irrelevant)
                with col3:
                    st.metric("Fake Rants", total_fake)

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    "review_analysis_results.csv",
                    "text/csv"
                )

    else:  # Batch Examples
        # Pre-defined examples
        examples = {
            "Advertisement": "Great food! Visit www.fooddeals.com for 50% off coupons!",
            "Irrelevant": "I love my new smartphone, anyway this restaurant is okay I guess.",
            "Fake Rant": "Never been here but I heard from my neighbor it's terrible.",
            "Normal Review": "Excellent service and delicious food. Highly recommended!",
        }

        st.subheader("Example Classifications:")

        for example_type, example_text in examples.items():
            with st.expander(f"{example_type} Example"):
                st.write(f"**Review:** {example_text}")

                if st.button(f"Analyze {example_type}", key=example_type):
                    result = classifier.classify_single_review(example_text)

                    for vtype, violation in result.items():
                        status = "❌ VIOLATION" if violation else "✅ OK"
                        st.write(f"**{vtype.title()}:** {status}")

if __name__ == "__main__":
    create_streamlit_demo()
```

---

## 🚀 Deployment & Final Steps

### Step 7: Package Everything

```python
# requirements.txt
"""
pandas>=1.3.0
numpy>=1.21.0
torch>=1.9.0
transformers>=4.20.0
huggingface_hub>=0.10.0
streamlit>=1.15.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""

# main.py - Entry point
def main():
    """Main execution function"""

    print("🚀 Starting Review Quality Assessment System")

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} reviews")

    # Preprocess
    df = extract_features(df)
    print("Features extracted")

    # Initialize classifier
    classifier = ReviewClassifier()
    print("Classifier initialized")

    # Run on sample
    sample_reviews = df['review_text'].head(10).tolist()
    results = classifier.classify_batch(sample_reviews)

    print("Sample Results:")
    for i, (review, result) in enumerate(zip(sample_reviews, results)):
        print(f"\nReview {i+1}: {review[:50]}...")
        for vtype, violation in result.items():
            print(f"  {vtype}: {'VIOLATION' if violation else 'OK'}")

if __name__ == "__main__":
    main()
```

### Step 8: Testing & Validation

```python
def run_comprehensive_test():
    """Run comprehensive testing before submission"""

    print("🧪 Running comprehensive tests...")

    # Test 1: Basic functionality
    classifier = ReviewClassifier()
    test_review = "Great food! Visit our website for discounts!"
    result = classifier.classify_single_review(test_review)
    assert result['advertisement'] == True, "Advertisement detection failed"
    print("✅ Basic functionality test passed")

    # Test 2: Batch processing
    test_reviews = [
        "Normal review about food quality",
        "Advertisement with website link",
        "Irrelevant content about weather"
    ]
    results = classifier.classify_batch(test_reviews)
    assert len(results) == 3, "Batch processing failed"
    print("✅ Batch processing test passed")

    # Test 3: Edge cases
    edge_cases = ["", "a", "A" * 1000]  # Empty, very short, very long
    for case in edge_cases:
        try:
            result = classifier.classify_single_review(case)
            print(f"✅ Edge case handled: {len(case)} characters")
        except Exception as e:
            print(f"❌ Edge case failed: {e}")

    print("🎉 All tests completed!")

if __name__ == "__main__":
    run_comprehensive_test()
```

---

## 🔍 Troubleshooting Common Issues

### Model Loading Issues

```python
# If Gemma model doesn't load, use these alternatives:
alternative_models = [
    "microsoft/DialoGPT-medium",  # Smaller, more reliable
    "distilbert-base-uncased",    # Fast classification
    "roberta-base"                # Good for text classification
]

# For CPU-only systems:
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased",
    device=-1  # Force CPU
)
```

### Memory Issues

```python
# Reduce batch size
def process_in_chunks(reviews, chunk_size=10):
    results = []
    for i in range(0, len(reviews), chunk_size):
        chunk = reviews[i:i+chunk_size]
        chunk_results = classifier.classify_batch(chunk)
        results.extend(chunk_results)

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
```

### Prompt Engineering Tips

```python
# If model gives inconsistent results, try:
# 1. More specific prompts
# 2. Few-shot examples
# 3. Chain-of-thought reasoning
# 4. Temperature adjustment (lower = more consistent)

improved_prompt = """
Let's think step by step.

First, read this review carefully: "{review_text}"

Now, check if this review contains any promotional content:
- Does it mention websites, phone numbers, or special offers?
- Is it trying to sell something or promote a business?

Based on this analysis, is this review an advertisement?
Answer: YES or NO

Answer:"""
```

This technical guide provides everything you need to implement your solution step by step. Remember to start simple and iterate quickly!
