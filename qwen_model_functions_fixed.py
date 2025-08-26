"""
Qwen Model Functions - Complete Standalone Solution for DeepSeek R1 Thinking Model
COMPLETELY INDEPENDENT - NO DEPENDENCIES ON f1_solution.py!
"""

import json
import re
import pandas as pd
import numpy as np
import gzip
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import openai
import warnings

warnings.filterwarnings("ignore")


class QwenReviewClassifier:
    """
    Complete standalone review classifier with DeepSeek R1 thinking model support.
    """

    def __init__(
        self,
        use_ml_models=True,
        lm_studio_url="http://localhost:1234/v1",
        model_name="deepseek-r1-distill-qwen-7b",
    ):
        self.use_ml_models = use_ml_models
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.client = None

        if use_ml_models:
            self._initialize_lm_studio()

        # Keyword patterns for rule-based classification
        self.promotional_keywords = [
            "discount",
            "sale",
            "promo",
            "deal",
            "offer",
            "coupon",
            "free",
            "visit our",
            "check out",
            "follow us",
            "website",
            "call us",
        ]

        self.irrelevant_keywords = [
            "weather",
            "politics",
            "my car",
            "my phone",
            "broke down",
            "parking lot",
            "traffic",
            "died",
            "lost",
        ]

        self.fake_rant_keywords = [
            "never been",
            "never visited",
            "heard from",
            "people say",
            "supposedly",
            "apparently",
            "probably",
            "likely",
        ]

    def _initialize_lm_studio(self):
        """Initialize LM Studio connection"""
        try:
            self.client = openai.OpenAI(
                base_url=self.lm_studio_url, api_key="not-needed"
            )
            models = self.client.models.list()
            model_ids = [model.id for model in models.data]

            if self.model_name in model_ids:
                print(f"✅ LM Studio connection established successfully!")
                print(
                    f"🤖 Using {self.model_name} model via LM Studio at {self.lm_studio_url}"
                )
            else:
                print(f"⚠️  Model {self.model_name} not found. Available: {model_ids}")
                self.use_ml_models = False

        except Exception as e:
            print(f"⚠️  LM Studio connection failed: {e}")
            self.use_ml_models = False

    def clean_lm_response(self, response_text: str) -> str:
        """Clean DeepSeek R1 response by removing <think> tags"""
        if not response_text:
            return "0.5"

        # Remove thinking tags
        if "<think>" in response_text.lower():
            # Remove complete thinking blocks
            think_pattern = r"<think>.*?</think>"
            response_text = re.sub(
                think_pattern, "", response_text, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove incomplete thinking blocks
            response_text = re.sub(
                r"<think>.*", "", response_text, flags=re.DOTALL | re.IGNORECASE
            )
            # Remove common thinking patterns
            response_text = re.sub(
                r"Alright,?\s*(so\s*)?(I\s*need\s*to\s*|let me\s*)?analyze.*",
                "",
                response_text,
                flags=re.IGNORECASE,
            )
            response_text = re.sub(
                r"Okay,?\s*(so\s*)?(I\s*need\s*to\s*|let me\s*)?analyze.*",
                "",
                response_text,
                flags=re.IGNORECASE,
            )

        response_text = response_text.strip()
        return response_text if response_text else "0.5"

    def get_ml_score(self, text: str, category: str) -> Tuple[float, float]:
        """Get ML score using DeepSeek R1 model"""
        if not self.use_ml_models or not self.client:
            return 0.5, 0.3

        try:
            if category == "advertisement":
                prompt = f"""Analyze this review for advertising/promotional content.
Review: "{text}"
Score from 0.0 (not advertisement) to 1.0 (clear advertisement):"""
            elif category == "irrelevant":
                prompt = f"""Analyze this review for content irrelevant to the business.
Review: "{text}"  
Score from 0.0 (relevant) to 1.0 (irrelevant):"""
            elif category == "fake_rant":
                prompt = f"""Analyze this review for fake rant indicators (never visited but complaining).
Review: "{text}"
Score from 0.0 (genuine) to 1.0 (fake rant):"""
            else:
                return 0.5, 0.3

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
            )

            response_content = response.choices[0].message.content
            cleaned_response = self.clean_lm_response(response_content)

            # Extract score
            score_match = re.search(r"([0-1]?\.\d+|[01])", cleaned_response)
            if score_match:
                score = float(score_match.group(1))
                return score, 0.8
            else:
                return 0.5, 0.3

        except Exception:
            return 0.5, 0.3

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text"""
        text_lower = text.lower()

        return {
            "length": len(text),
            "word_count": len(text.split()),
            "has_url": bool(re.search(r"https?://|www\.|\.com|\.net|\.org", text)),
            "has_phone": bool(re.search(r"\d{3}[-.]?\d{3}[-.]?\d{4}", text)),
            "promotional_words": sum(
                1 for kw in self.promotional_keywords if kw in text_lower
            ),
            "irrelevant_words": sum(
                1 for kw in self.irrelevant_keywords if kw in text_lower
            ),
            "fake_rant_words": sum(
                1 for kw in self.fake_rant_keywords if kw in text_lower
            ),
            "exclamation_count": text.count("!"),
        }

    def classify_advertisement(
        self, text: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify advertisement"""
        rule_score = 0.0

        if features["has_url"] or features["has_phone"]:
            rule_score += 0.4
        if features["promotional_words"] >= 2:
            rule_score += 0.3
        elif features["promotional_words"] >= 1:
            rule_score += 0.2

        ml_score, ml_confidence = self.get_ml_score(text, "advertisement")

        if self.use_ml_models and ml_confidence > 0.5:
            final_score = (ml_score * 0.7) + (rule_score * 0.3)
            confidence = ml_confidence
        else:
            final_score = rule_score
            confidence = 0.6

        return {
            "is_advertisement": final_score > 0.5,
            "confidence": min(final_score, 1.0),
            "reasoning": "Combined ML and rule-based analysis",
        }

    def classify_irrelevant(
        self, text: str, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Classify irrelevant content"""
        rule_score = 0.0

        if features["irrelevant_words"] >= 2:
            rule_score += 0.4
        elif features["irrelevant_words"] >= 1:
            rule_score += 0.2

        # Check business relevance
        business_words = ["service", "food", "staff", "restaurant", "place"]
        has_business_content = any(word in text.lower() for word in business_words)
        if not has_business_content and features["word_count"] > 10:
            rule_score += 0.3

        ml_score, ml_confidence = self.get_ml_score(text, "irrelevant")

        if self.use_ml_models and ml_confidence > 0.5:
            final_score = (ml_score * 0.7) + (rule_score * 0.3)
            confidence = ml_confidence
        else:
            final_score = rule_score
            confidence = 0.6

        return {
            "is_irrelevant": final_score > 0.5,
            "confidence": min(final_score, 1.0),
            "reasoning": "Combined ML and rule-based analysis",
        }

    def classify_fake_rant(self, text: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify fake rant"""
        rule_score = 0.0

        if features["fake_rant_words"] >= 2:
            rule_score += 0.4
        elif features["fake_rant_words"] >= 1:
            rule_score += 0.2

        # Check for hearsay patterns
        text_lower = text.lower()
        hearsay_patterns = ["never been", "heard from", "people say", "supposedly"]
        if any(pattern in text_lower for pattern in hearsay_patterns):
            rule_score += 0.4

        ml_score, ml_confidence = self.get_ml_score(text, "fake_rant")

        if self.use_ml_models and ml_confidence > 0.5:
            final_score = (ml_score * 0.7) + (rule_score * 0.3)
            confidence = ml_confidence
        else:
            final_score = rule_score
            confidence = 0.6

        return {
            "is_fake_rant": final_score > 0.5,
            "confidence": min(final_score, 1.0),
            "reasoning": "Combined ML and rule-based analysis",
        }

    def classify_review(self, text: str) -> Dict[str, Any]:
        """Main classification method"""
        if not text or len(text.strip()) == 0:
            return {
                "advertisement": {"is_advertisement": False, "confidence": 0.0},
                "irrelevant": {"is_irrelevant": False, "confidence": 0.0},
                "fake_rant": {"is_fake_rant": False, "confidence": 0.0},
            }

        features = self.extract_features(text)

        return {
            "advertisement": self.classify_advertisement(text, features),
            "irrelevant": self.classify_irrelevant(text, features),
            "fake_rant": self.classify_fake_rant(text, features),
            "features": features,
        }


class QwenDataPipeline:
    """Complete standalone data pipeline"""

    def __init__(self, reviews_path: str = None, meta_path: str = None):
        self.reviews_path = reviews_path
        self.meta_path = meta_path
        self.reviews_data = None
        self.meta_data = None

    def load_data(self):
        """Load data from gzip JSON files"""
        if self.reviews_path:
            try:
                with gzip.open(self.reviews_path, "rt") as f:
                    reviews = []
                    for line in f:
                        reviews.append(json.loads(line))
                self.reviews_data = pd.DataFrame(reviews)
                print(f"✅ Loaded {len(self.reviews_data)} reviews")
            except Exception as e:
                print(f"⚠️  Could not load reviews: {e}")
                self.reviews_data = None

        if self.meta_path:
            try:
                with gzip.open(self.meta_path, "rt") as f:
                    meta = []
                    for line in f:
                        meta.append(json.loads(line))
                self.meta_data = pd.DataFrame(meta)
                print(f"✅ Loaded {len(self.meta_data)} business records")
            except Exception as e:
                print(f"⚠️  Could not load meta data: {e}")
                self.meta_data = None

        return self

    def clean_data(self):
        """Clean and prepare data"""
        if self.reviews_data is not None:
            self.reviews_data["text_length"] = self.reviews_data["text"].str.len()
            self.reviews_data["word_count"] = (
                self.reviews_data["text"].str.split().str.len()
            )
            self.reviews_data = self.reviews_data[self.reviews_data["text"].notna()]
            self.reviews_data = self.reviews_data[
                self.reviews_data["text"].str.strip() != ""
            ]
            print(f"✅ Data cleaned. {len(self.reviews_data)} reviews remaining")
        return self

    def create_sample_dataset(self, sample_size: int = 1000) -> pd.DataFrame:
        """Create sample dataset"""
        if self.reviews_data is not None and len(self.reviews_data) > 0:
            sample_size = min(sample_size, len(self.reviews_data))
            return self.reviews_data.sample(n=sample_size, random_state=42)
        else:
            return create_enhanced_sample_data(sample_size)


class QwenF1Evaluator:
    """Enhanced evaluator for DeepSeek R1 thinking model"""

    def __init__(self):
        self.categories = ["advertisement", "irrelevant", "fake_rant"]
        self.parsing_stats = {
            "total_requests": 0,
            "successful_parses": 0,
            "think_tag_responses": 0,
            "fallback_used": 0,
        }

    def evaluate_classifier(
        self, classifier, test_data: pd.DataFrame, sample_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Evaluate classifier with enhanced error handling"""
        print("🔍 Enhanced Qwen F1 Evaluator - Starting evaluation...")

        # Sample data if requested
        if sample_size and len(test_data) > sample_size:
            test_sample = test_data.sample(n=sample_size, random_state=42)
            print(f"📊 Using sample of {sample_size} reviews for evaluation")
        else:
            test_sample = test_data

        # Reset parsing stats
        self.parsing_stats = {
            "total_requests": 0,
            "successful_parses": 0,
            "think_tag_responses": 0,
            "fallback_used": 0,
        }

        # Get predictions
        print(f"📈 Evaluating classifier performance...")
        predictions = []

        total_reviews = len(test_sample)
        for idx, (_, row) in enumerate(test_sample.iterrows(), 1):
            if idx % 50 == 0 or idx == 1:
                print(f"Processing review {idx}/{total_reviews}")

            try:
                result = classifier.classify_review(row["text"])
                predictions.append(result)
            except Exception as e:
                print(f"   ⚠️  Error processing review {idx}: {str(e)[:50]}...")
                predictions.append(self._get_fallback_response())

        # Calculate metrics
        evaluation_results = {}
        overall_scores = {"precision": [], "recall": [], "f1": [], "accuracy": []}

        for category in self.categories:
            # Extract labels
            y_true = test_sample[f"is_{category}"].values
            y_pred = [pred[category][f"is_{category}"] for pred in predictions]

            # Calculate metrics
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            support = sum(y_true)

            evaluation_results[category] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "support": support,
                "y_true": y_true,
                "y_pred": y_pred,
            }

            overall_scores["precision"].append(precision)
            overall_scores["recall"].append(recall)
            overall_scores["f1"].append(f1)
            overall_scores["accuracy"].append(accuracy)

        # Overall metrics
        evaluation_results["overall_precision"] = np.mean(overall_scores["precision"])
        evaluation_results["overall_recall"] = np.mean(overall_scores["recall"])
        evaluation_results["overall_f1"] = np.mean(overall_scores["f1"])
        evaluation_results["overall_accuracy"] = np.mean(overall_scores["accuracy"])

        print(f"\n✅ Evaluation completed!")
        return evaluation_results

    def _get_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response when processing fails"""
        self.parsing_stats["fallback_used"] += 1
        return {
            category: {
                f"is_{category}": False,
                "confidence": 0.1,
                "reasoning": "Fallback response",
            }
            for category in self.categories
        }

    def plot_confusion_matrices(self, evaluation_results: Dict[str, Any]) -> plt.Figure:
        """Plot confusion matrices"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for idx, category in enumerate(self.categories):
            if category in evaluation_results:
                y_true = evaluation_results[category]["y_true"]
                y_pred = evaluation_results[category]["y_pred"]

                cm = confusion_matrix(y_true, y_pred)

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=axes[idx],
                    cbar=True,
                    xticklabels=["Not " + category.title(), category.title()],
                    yticklabels=["Not " + category.title(), category.title()],
                )

                axes[idx].set_title(f"{category.title()} Confusion Matrix")
                axes[idx].set_xlabel("Predicted")
                axes[idx].set_ylabel("Actual")

        plt.tight_layout()
        return fig

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = [
            "\n" + "=" * 60,
            "🏆 QWEN F1 SOLUTION EVALUATION REPORT",
            "=" * 60,
            "",
            f"📊 Overall Performance:",
            f"   • Overall F1 Score: {evaluation_results['overall_f1']:.3f}",
            f"   • Overall Precision: {evaluation_results['overall_precision']:.3f}",
            f"   • Overall Recall: {evaluation_results['overall_recall']:.3f}",
            f"   • Overall Accuracy: {evaluation_results['overall_accuracy']:.3f}",
            "",
            "📈 Category-wise Performance:",
        ]

        for category in self.categories:
            if category in evaluation_results:
                cat_results = evaluation_results[category]
                report_lines.extend(
                    [
                        f"",
                        f"   🎯 {category.title()}:",
                        f"      • F1 Score: {cat_results['f1_score']:.3f}",
                        f"      • Precision: {cat_results['precision']:.3f}",
                        f"      • Recall: {cat_results['recall']:.3f}",
                        f"      • Accuracy: {cat_results['accuracy']:.3f}",
                        f"      • Support: {cat_results['support']}",
                    ]
                )

        report_lines.extend(
            [
                "",
                "✅ Evaluation completed with enhanced DeepSeek R1 support!",
                "=" * 60,
            ]
        )

        return "\n".join(report_lines)

    def classify_batch_with_progress(
        self, classifier, texts: List[str], batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """Classify batch with progress tracking"""
        results = []
        total = len(texts)

        print(f"🚀 Processing {total} texts...")

        for idx, text in enumerate(texts, 1):
            if idx % batch_size == 0 or idx == 1 or idx == total:
                print(f"   Processing {idx}/{total} ({(idx/total)*100:.1f}%)")

            try:
                result = classifier.classify_review(text)
                results.append(result)
            except Exception as e:
                print(f"   ⚠️  Error processing text {idx}: {str(e)[:50]}...")
                results.append(self._get_fallback_response())

        return results


def create_enhanced_sample_data(sample_size: int = 500) -> pd.DataFrame:
    """Create enhanced sample dataset for testing"""
    print(f"🎯 Creating enhanced sample dataset ({sample_size} samples)...")

    # Diverse review templates
    clean_reviews = [
        "Great food and excellent service! Highly recommended.",
        "Nice atmosphere, friendly staff, good value for money.",
        "Delicious meals and quick service. Will come back again.",
        "Family-friendly restaurant with tasty food options.",
        "Good location, clean facilities, reasonable prices.",
    ]

    advertisement_reviews = [
        "Visit our website at www.restaurant.com for amazing deals!",
        "Call 555-1234 for reservations and special offers!",
        "Check out our Facebook page for daily discounts!",
        "Free delivery on orders over $25! Order online now!",
        "Follow us on Instagram @restaurant for exclusive coupons!",
    ]

    irrelevant_reviews = [
        "My car broke down in the parking lot. Weather was terrible.",
        "I lost my phone here last week. Politics are crazy these days.",
        "The traffic was bad when I visited. My dog got sick.",
        "Construction noise was loud. My neighbor moved away recently.",
        "Had to wait for a taxi. The news today is very concerning.",
    ]

    fake_rant_reviews = [
        "Never been here but heard it's overpriced and terrible service.",
        "My friend said the food is awful. Probably not worth visiting.",
        "People told me to avoid this place. Seems like a waste of money.",
        "Haven't visited yet but reviews online are mostly negative.",
        "Was planning to go but decided against it based on rumors.",
    ]

    # Generate samples with proper distribution
    samples = []
    categories = [
        (
            clean_reviews,
            {
                "is_advertisement": False,
                "is_irrelevant": False,
                "is_fake_rant": False,
            },
        ),
        (
            advertisement_reviews,
            {"is_advertisement": True, "is_irrelevant": False, "is_fake_rant": False},
        ),
        (
            irrelevant_reviews,
            {"is_advertisement": False, "is_irrelevant": True, "is_fake_rant": False},
        ),
        (
            fake_rant_reviews,
            {"is_advertisement": False, "is_irrelevant": False, "is_fake_rant": True},
        ),
    ]

    # Distribute samples (60% clean, 40% violations)
    samples_per_category = [
        int(sample_size * 0.6),  # Clean: 60%
        int(sample_size * 0.15),  # Advertisement: 15%
        int(sample_size * 0.15),  # Irrelevant: 15%
        int(sample_size * 0.1),  # Fake rant: 10%
    ]

    for (templates, labels), count in zip(categories, samples_per_category):
        for i in range(count):
            template = templates[i % len(templates)]
            # Add variation
            if i > 0:
                variations = [
                    f"{template} Really enjoyed the experience.",
                    f"Honestly, {template.lower()}",
                    f"{template} Would definitely recommend.",
                    template,
                ]
                text = variations[i % len(variations)]
            else:
                text = template

            sample = {
                "text": text,
                "rating": np.random.choice([1, 2, 3, 4, 5]),
                "text_length": len(text),
                "word_count": len(text.split()),
                **labels,
            }
            samples.append(sample)

    # Create DataFrame and shuffle
    df = pd.DataFrame(samples)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"✅ Created enhanced dataset with {len(df)} samples")
    print(f"   Distribution:")
    for label in ["is_advertisement", "is_irrelevant", "is_fake_rant"]:
        count = df[label].sum()
        percentage = (count / len(df)) * 100
        print(f"   • {label.replace('is_', '').title()}: {count} ({percentage:.1f}%)")

    return df


def create_qwen_evaluator():
    """Create a Qwen F1 Evaluator instance"""
    return QwenF1Evaluator()


def quick_evaluation_test(classifier, sample_size: int = 100):
    """Run quick evaluation test"""
    print(f"🚀 Running quick evaluation test ({sample_size} samples)...")

    # Create test data
    test_data = create_enhanced_sample_data(sample_size)

    # Create evaluator
    evaluator = create_qwen_evaluator()

    # Run evaluation
    results = evaluator.evaluate_classifier(classifier, test_data)

    print(f"\n📊 Quick Test Results:")
    print(f"   Overall F1: {results['overall_f1']:.3f}")

    return results


# Export main classes and functions
__all__ = [
    "QwenReviewClassifier",
    "QwenDataPipeline",
    "QwenF1Evaluator",
    "create_enhanced_sample_data",
    "create_qwen_evaluator",
    "quick_evaluation_test",
]

if __name__ == "__main__":
    print("🚀 Qwen Model Functions loaded successfully!")
    print("Available functions:")
    print("  • QwenReviewClassifier() - Complete standalone classifier")
    print("  • QwenDataPipeline() - Data pipeline")
    print("  • QwenF1Evaluator() - Enhanced evaluator")
    print("  • create_enhanced_sample_data() - Generate test data")
    print("  • quick_evaluation_test() - Fast evaluation")
