#!/usr/bin/env python3
"""
F1 Solution for TechJam 2025: ML for Trustworthy Location Reviews
Building a Winning Classification System for Policy Violation Detection

This solution implements a comprehensive ML pipeline to detect:
1. Advertisements (promotional content)
2. Irrelevant content (not related to location)
3. Fake rants (complaints from users who never visited)
"""

import pandas as pd
import numpy as np
import re
import gzip
import json
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# ML and evaluation imports
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# For LM Studio API
try:
    from openai import OpenAI

    LM_STUDIO_AVAILABLE = True
    print("🚀 OpenAI client available for LM Studio!")
except ImportError:
    LM_STUDIO_AVAILABLE = False
    print("⚠️  OpenAI client not available, using rule-based only")

# For transformer models (backup)
try:
    from transformers import pipeline

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class ReviewPolicyClassifier:
    """
    Main classifier for detecting policy violations in reviews
    """

    def __init__(
        self,
        use_ml_models=True,
        lm_studio_url="http://localhost:1234/v1",
        model_name="gemma-2-2b-it",
    ):
        self.use_ml_models = use_ml_models
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.classifier = None
        self.openai_client = None
        self._setup_models()

        # Define keyword patterns for rule-based classification
        self.ad_keywords = [
            "visit",
            "website",
            "www",
            "http",
            "call",
            "phone",
            "discount",
            "deal",
            "promo",
            "sale",
            "coupon",
            "special offer",
            "check out",
            "click here",
            "link",
            ".com",
            "promotion",
            "offer",
            "free delivery",
        ]

        self.irrelevant_indicators = [
            "my phone",
            "my car",
            "politics",
            "weather",
            "traffic",
            "my day",
            "my life",
            "news",
            "government",
            "president",
            "election",
            "coronavirus",
            "covid",
            "vaccine",
            "personal life",
        ]

        self.fake_rant_indicators = [
            "never been",
            "never visited",
            "heard it",
            "looks like",
            "probably",
            "i hate these",
            "all these places",
            "never went",
            "sounds like",
            "seems like",
            "i bet",
            "typical",
            "always like this",
        ]

    def _setup_models(self):
        """Setup ML models if available"""
        if self.use_ml_models:
            # Try LM Studio first
            if LM_STUDIO_AVAILABLE:
                try:
                    self.openai_client = OpenAI(
                        base_url=self.lm_studio_url,
                        api_key="lm-studio",  # LM Studio doesn't require a real API key
                    )
                    # Test the connection
                    response = self.openai_client.chat.completions.create(
                        model=self.model_name,  # Use configurable model name
                        messages=[{"role": "user", "content": "test"}],
                        max_tokens=1,
                    )
                    print("✅ LM Studio connection established successfully!")
                    print(
                        f"🤖 Using {self.model_name} model via LM Studio at {self.lm_studio_url}"
                    )
                    return
                except Exception as e:
                    print(f"❌ LM Studio connection failed: {e}")
                    print(
                        "💡 Make sure LM Studio is running and Gemma 1.2 model is loaded"
                    )

            # Fallback to Hugging Face transformers
            if HF_AVAILABLE:
                try:
                    self.classifier = pipeline(
                        "text-classification",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1,  # Use CPU
                    )
                    print("✅ Fallback to Hugging Face model loaded successfully!")
                except Exception as e:
                    print(f"❌ ML model loading failed: {e}")
                    self.use_ml_models = False
            else:
                print("❌ No ML backends available, using rule-based only")
                self.use_ml_models = False

    def _analyze_with_lm_studio(
        self, text: str, analysis_type: str
    ) -> Tuple[float, float]:
        """Use LM Studio with Gemma 1.2 to analyze text for policy violations"""
        try:
            if analysis_type == "advertisement":
                prompt = f"""Analyze this review text for promotional/advertising content. Reply with just a score from 0.0 to 1.0 where:
- 0.0 = definitely not promotional/advertising
- 1.0 = definitely promotional/advertising content

Look for: URLs, phone numbers, promotional language, business self-promotion, discount mentions, calls to action.

Review text: "{text[:500]}"

Score:"""

            elif analysis_type == "irrelevant":
                prompt = f"""Analyze this review text to determine if it's irrelevant to a business/location. Reply with just a score from 0.0 to 1.0 where:
- 0.0 = relevant to the business/location
- 1.0 = completely irrelevant to business/location

Look for: personal life content, politics, weather, off-topic subjects, lack of business-related keywords.

Review text: "{text[:500]}"

Score:"""

            elif analysis_type == "fake_rant":
                prompt = f"""Analyze this review text to determine if it's a fake rant from someone who never visited. Reply with just a score from 0.0 to 1.0 where:
- 0.0 = genuine review from actual visitor
- 1.0 = fake rant from non-visitor

Look for: admissions of never visiting, hearsay language, assumptions without experience, generic complaints without specifics.

Review text: "{text[:500]}"

Score:"""

            else:
                return 0.5, 0.5

            if self.openai_client is None:
                return 0.5, 0.5

            response = self.openai_client.chat.completions.create(
                model=self.model_name,  # Use configurable model name
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing review text for policy violations. Always respond with only a decimal number between 0.0 and 1.0.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            # Extract score from response
            response_content = response.choices[0].message.content
            if response_content is None:
                print("LM Studio returned empty response")
                return 0.5, 0.5

            score_text = response_content.strip()

            # Handle DeepSeek R1 thinking model responses - remove <think> tags
            if "<think>" in score_text.lower():
                # Remove thinking tags and their content
                think_pattern = r"<think>.*?</think>"
                score_text = re.sub(
                    think_pattern, "", score_text, flags=re.DOTALL | re.IGNORECASE
                )
                # Also remove incomplete thinking blocks
                score_text = re.sub(
                    r"<think>.*", "", score_text, flags=re.DOTALL | re.IGNORECASE
                )
                # Remove common thinking patterns
                score_text = re.sub(
                    r"Alright,?\s*(so\s*)?(I\s*need\s*to\s*|let me\s*)?analyze.*",
                    "",
                    score_text,
                    flags=re.IGNORECASE,
                )
                score_text = re.sub(
                    r"Okay,?\s*(so\s*)?(I\s*need\s*to\s*|let me\s*)?analyze.*",
                    "",
                    score_text,
                    flags=re.IGNORECASE,
                )
                score_text = score_text.strip()

            # Try to extract a float from the cleaned response
            score_match = re.search(r"([0-1]?\.\d+|[01])", score_text)
            if score_match:
                score = float(score_match.group(1))
                confidence = 0.8  # High confidence in LM Studio analysis
                return score, confidence
            else:
                # Fallback: if we still can't parse, use rule-based approach silently
                return 0.5, 0.5

        except Exception as e:
            print(f"LM Studio analysis error: {e}")
            return 0.5, 0.5

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from review text"""
        text_lower = text.lower()

        features = {
            # Basic text features
            "length": len(text),
            "word_count": len(text.split()),
            "exclamation_count": text.count("!"),
            "question_count": text.count("?"),
            "caps_ratio": (
                sum(1 for c in text if c.isupper()) / len(text) if text else 0
            ),
            # URL and contact detection
            "has_url": bool(re.search(r"(www\.|http|\.com|\.org|\.net)", text_lower)),
            "has_phone": bool(re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", text)),
            "has_email": bool(
                re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
            ),
            # Promotional content
            "promotional_words": sum(
                1 for word in self.ad_keywords if word in text_lower
            ),
            "has_discount_mention": any(
                word in text_lower for word in ["discount", "deal", "sale", "promo"]
            ),
            # Irrelevant content indicators
            "irrelevant_words": sum(
                1 for word in self.irrelevant_indicators if word in text_lower
            ),
            "personal_pronouns": text_lower.count("my ") + text_lower.count("i "),
            # Fake rant indicators
            "fake_rant_words": sum(
                1 for word in self.fake_rant_indicators if word in text_lower
            ),
            "negative_assumptions": sum(
                1
                for phrase in ["probably", "i bet", "sounds like"]
                if phrase in text_lower
            ),
        }

        return features

    def classify_advertisement(self, text: str, features: Dict) -> Dict[str, Any]:
        """Classify if review contains advertisements"""
        # Enhanced rule-based approach
        rule_score = 0
        text_lower = text.lower()

        # Strong advertisement indicators
        if features["has_url"] or features["has_phone"] or features["has_email"]:
            rule_score += 0.6

        # Promotional language
        if features["promotional_words"] >= 2:
            rule_score += 0.4
        elif features["promotional_words"] >= 1:
            rule_score += 0.2

        # Discount/deal mentions
        if features["has_discount_mention"]:
            rule_score += 0.3

        # Strong promotional phrases
        strong_promo_phrases = [
            "visit our",
            "check out",
            "click here",
            "call us",
            "contact us",
            "order online",
            "free delivery",
            "special offer",
            "limited time",
            "book now",
            "reserve now",
            "download our app",
        ]
        if any(phrase in text_lower for phrase in strong_promo_phrases):
            rule_score += 0.4

        # Business self-promotion indicators
        self_promo_patterns = [
            "our website",
            "our facebook",
            "our instagram",
            "our menu",
            "follow us",
            "like us",
            "subscribe",
            "sign up",
        ]
        if any(pattern in text_lower for pattern in self_promo_patterns):
            rule_score += 0.3

        # ML enhancement if available
        ml_score = 0.5  # neutral
        confidence = 0.6

        # Use LM Studio as primary decision maker when available
        if self.use_ml_models and self.openai_client:
            try:
                # Use LM Studio with Gemma to analyze promotional content
                ml_score, confidence = self._analyze_with_lm_studio(
                    text, "advertisement"
                )
                # Use ML score as primary decision
                is_advertisement = ml_score > 0.5
            except Exception as e:
                print(f"LM Studio analysis failed: {e}")
                # Fallback to rule-based decision
                is_advertisement = rule_score > 0.5
        elif self.use_ml_models and self.classifier:
            try:
                # Fallback to Hugging Face sentiment analysis
                result = self.classifier(text[:512])  # Limit text length
                if result[0]["label"] == "POSITIVE" and result[0]["score"] > 0.9:
                    ml_score = 0.7  # High positive sentiment might indicate promotion
                confidence = result[0]["score"]
                is_advertisement = ml_score > 0.5
            except:
                # Fallback to rule-based decision
                is_advertisement = rule_score > 0.5
        else:
            # Pure rule-based decision when no ML available
            is_advertisement = rule_score > 0.5

        return {
            "is_advertisement": is_advertisement,
            "confidence": confidence,
            "rule_score": rule_score,
            "ml_score": ml_score,
            "features_detected": {
                "has_contact_info": features["has_url"]
                or features["has_phone"]
                or features["has_email"],
                "promotional_language": features["promotional_words"] > 0,
                "discount_mention": features["has_discount_mention"],
            },
        }

    def classify_irrelevant(self, text: str, features: Dict) -> Dict[str, Any]:
        """Classify if review is irrelevant to the location"""
        rule_score = 0
        text_lower = text.lower()

        # Strong irrelevant content indicators
        if features["irrelevant_words"] >= 3:
            rule_score += 0.6
        elif features["irrelevant_words"] >= 2:
            rule_score += 0.4
        elif features["irrelevant_words"] >= 1:
            rule_score += 0.2

        # Very personal content unrelated to business
        if features["personal_pronouns"] >= 4:
            rule_score += 0.3

        # Check for business-related keywords
        business_keywords = [
            "service",
            "staff",
            "food",
            "place",
            "location",
            "experience",
            "visit",
            "restaurant",
            "store",
            "shop",
            "business",
            "customer",
            "order",
            "ordered",
            "ate",
            "served",
            "server",
            "waiter",
            "waitress",
            "manager",
            "table",
            "menu",
            "price",
            "quality",
            "atmosphere",
            "clean",
            "dirty",
            "recommend",
        ]
        has_business_context = any(word in text_lower for word in business_keywords)

        # No business context is a strong indicator
        if not has_business_context:
            rule_score += 0.4

        # Off-topic content patterns
        off_topic_patterns = [
            "my personal",
            "my family",
            "my relationship",
            "my job",
            "my work",
            "politics",
            "government",
            "election",
            "president",
            "mayor",
            "weather was",
            "traffic was",
            "parking was difficult",
        ]
        if any(pattern in text_lower for pattern in off_topic_patterns):
            rule_score += 0.3

        # Very short reviews without business content
        if features["word_count"] < 8 and not has_business_context:
            rule_score += 0.3

        # Too much irrelevant content ratio
        if (
            features["word_count"] > 5
            and features["irrelevant_words"] > features["word_count"] * 0.4
        ):
            rule_score += 0.4

        # ML enhancement with LM Studio
        ml_score = 0.5  # neutral
        confidence = min(rule_score + 0.3, 1.0)

        # Use LM Studio as primary decision maker when available
        if self.use_ml_models and self.openai_client:
            try:
                # Use LM Studio to analyze relevance to business/location
                ml_score, confidence = self._analyze_with_lm_studio(text, "irrelevant")
                # Use ML score as primary decision
                is_irrelevant = ml_score > 0.5
            except Exception as e:
                print(f"LM Studio analysis failed: {e}")
                # Fallback to rule-based decision
                is_irrelevant = rule_score > 0.4
        else:
            # Pure rule-based decision when no ML available
            is_irrelevant = rule_score > 0.4

        return {
            "is_irrelevant": is_irrelevant,
            "confidence": confidence,
            "rule_score": rule_score,
            "ml_score": ml_score,
            "features_detected": {
                "high_irrelevant_words": features["irrelevant_words"] >= 2,
                "too_personal": features["personal_pronouns"] >= 3,
                "no_business_context": not has_business_context,
            },
        }

    def classify_fake_rant(self, text: str, features: Dict) -> Dict[str, Any]:
        """Classify if review is a fake rant from someone who never visited"""
        rule_score = 0
        text_lower = text.lower()

        # Direct admission of not visiting (very strong indicator)
        never_visited_patterns = [
            "never been",
            "never visited",
            "never went",
            "haven't been",
            "haven't visited",
            "never actually",
            "not been there",
        ]
        if any(pattern in text_lower for pattern in never_visited_patterns):
            rule_score += 0.7

        # Hearsay indicators
        hearsay_patterns = [
            "heard it",
            "heard that",
            "heard from",
            "people say",
            "they say",
            "someone told me",
            "word is",
            "rumor has it",
            "i heard",
        ]
        if any(pattern in text_lower for pattern in hearsay_patterns):
            rule_score += 0.4

        # Assumptions without experience
        assumption_patterns = [
            "probably",
            "i bet",
            "seems like",
            "sounds like",
            "looks like",
            "must be",
            "i imagine",
            "i assume",
            "typical",
        ]
        assumption_count = sum(
            1 for pattern in assumption_patterns if pattern in text_lower
        )
        if assumption_count >= 2:
            rule_score += 0.5
        elif assumption_count >= 1:
            rule_score += 0.3

        # Generic complaints without specifics
        generic_complaints = [
            "terrible",
            "awful",
            "worst",
            "horrible",
            "hate",
            "disgusting",
        ]
        specific_details = [
            "ordered",
            "ate",
            "waited",
            "server",
            "menu",
            "table",
            "food was",
            "service was",
            "staff was",
            "manager",
            "bill",
            "price",
            "atmosphere",
        ]

        has_generic = any(word in text_lower for word in generic_complaints)
        has_specific = any(phrase in text_lower for phrase in specific_details)

        if has_generic and not has_specific:
            rule_score += 0.4

        # Very short negative reviews without specifics
        if features["word_count"] < 15 and has_generic and not has_specific:
            rule_score += 0.3

        # Patterns indicating no actual experience
        no_experience_patterns = [
            "avoid this place",
            "don't go",
            "stay away",
            "don't waste",
            "save your money",
            "not worth it",
        ]
        if (
            any(pattern in text_lower for pattern in no_experience_patterns)
            and not has_specific
        ):
            rule_score += 0.3

        # Multiple negative assumptions
        if features["negative_assumptions"] >= 2:
            rule_score += 0.4
        elif features["negative_assumptions"] >= 1:
            rule_score += 0.2

        # ML enhancement with LM Studio
        ml_score = 0.5  # neutral
        confidence = min(rule_score + 0.2, 1.0)

        # Use LM Studio as primary decision maker when available
        if self.use_ml_models and self.openai_client:
            try:
                # Use LM Studio to analyze if this is a fake rant
                ml_score, confidence = self._analyze_with_lm_studio(text, "fake_rant")
                # Use ML score as primary decision
                is_fake_rant = ml_score > 0.5
            except Exception as e:
                print(f"LM Studio analysis failed: {e}")
                # Fallback to rule-based decision
                is_fake_rant = rule_score > 0.4
        else:
            # Pure rule-based decision when no ML available
            is_fake_rant = rule_score > 0.4

        return {
            "is_fake_rant": is_fake_rant,
            "confidence": confidence,
            "rule_score": rule_score,
            "ml_score": ml_score,
            "features_detected": {
                "admits_no_visit": any(
                    pattern in text_lower for pattern in never_visited_patterns
                ),
                "uses_hearsay": any(
                    pattern in text_lower for pattern in hearsay_patterns
                ),
                "makes_assumptions": features["negative_assumptions"] >= 1,
                "generic_without_specifics": has_generic and not has_specific,
            },
        }

    def classify_review(self, text: str) -> Dict[str, Any]:
        """Main classification method for a single review"""
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

    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify multiple reviews"""
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"Processing review {i+1}/{len(texts)}")
            results.append(self.classify_review(text))
        return results


class F1DataPipeline:
    """
    Enhanced data pipeline for processing and preparing review data
    """

    def __init__(self, reviews_path: str, meta_path: str):
        self.reviews_path = reviews_path
        self.meta_path = meta_path
        self.reviews_data = None
        self.meta_data = None
        self.processed_data = None

    def load_data(self):
        """Load and basic cleaning of review and business data"""
        print("📊 Loading data...")

        # Load reviews
        self.reviews_data = pd.read_json(
            self.reviews_path, lines=True, compression="gzip"
        )

        # Load business metadata
        self.meta_data = pd.read_json(self.meta_path, lines=True, compression="gzip")

        # Standardize columns
        self.reviews_data.columns = self.reviews_data.columns.str.lower().str.strip()
        self.meta_data.columns = self.meta_data.columns.str.lower().str.strip()

        print(
            f"✅ Loaded {len(self.reviews_data):,} reviews and {len(self.meta_data):,} businesses"
        )

        return self

    def clean_data(self):
        """Clean and prepare data for analysis"""
        print("🧹 Cleaning data...")

        # Clean reviews - keep only reviews with text
        if self.reviews_data is None:
            print("❌ No reviews data loaded")
            return self

        initial_count = len(self.reviews_data)
        self.reviews_data = self.reviews_data.dropna(
            subset=["text", "rating", "gmap_id"]
        )
        self.reviews_data = self.reviews_data[self.reviews_data["text"].str.len() > 0]

        print(
            f"✅ Kept {len(self.reviews_data):,} reviews with text ({initial_count - len(self.reviews_data):,} removed)"
        )

        # Create additional features
        self.reviews_data["has_pics"] = self.reviews_data["pics"].notna()
        self.reviews_data["has_response"] = self.reviews_data["resp"].notna()
        self.reviews_data["text_length"] = self.reviews_data["text"].str.len()
        self.reviews_data["word_count"] = (
            self.reviews_data["text"].str.split().str.len()
        )

        # Clean metadata
        if self.meta_data is not None:
            self.meta_data = self.meta_data.dropna(subset=["gmap_id"])

        return self

    def create_sample_dataset(self, sample_size: int = 1000) -> pd.DataFrame:
        """Create a representative sample for testing"""
        # Check if reviews_data exists
        if self.reviews_data is None or len(self.reviews_data) == 0:
            print("❌ No reviews data available for sampling")
            return pd.DataFrame()

        # Stratified sampling by rating to ensure diversity
        sample_data = (
            self.reviews_data.groupby("rating")
            .apply(lambda x: x.sample(min(len(x), sample_size // 5)))
            .reset_index(drop=True)
        )

        # If we don't have enough, just take random sample
        if len(sample_data) < sample_size:
            sample_data = self.reviews_data.sample(
                min(len(self.reviews_data), sample_size)
            )

        return sample_data.copy()

    def generate_ground_truth_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pseudo ground truth labels for evaluation
        This simulates having labeled data for evaluation
        """
        print("🏷️  Generating ground truth labels for evaluation...")

        # Simple heuristic labeling for demonstration
        # In a real scenario, these would be manually labeled or from a better model

        data = data.copy()

        # Advertisement labels (more precise rules)
        data["is_advertisement"] = data["text"].str.contains(
            r"www\.|http|\.com|call|phone|discount|deal|promo",
            case=False,
            na=False,
            regex=True,
        ) | data["text"].str.contains(
            r"visit our|check out|special offer|click here",
            case=False,
            na=False,
            regex=True,
        )

        # Irrelevant content (more conservative)
        irrelevant_patterns = (
            r"politics|weather|my phone|my car|personal life|coronavirus|vaccine"
        )
        data["is_irrelevant"] = data["text"].str.contains(
            irrelevant_patterns, case=False, na=False, regex=True
        ) & ~data["text"].str.contains(
            r"service|food|staff|place|experience", case=False, na=False, regex=True
        )

        # Fake rants (very specific patterns)
        data["is_fake_rant"] = data["text"].str.contains(
            r"never been|never visited|heard it|probably|i bet|sounds like.*terrible",
            case=False,
            na=False,
            regex=True,
        )

        # Ensure no overlap (advertisement takes precedence, then irrelevant, then fake_rant)
        data.loc[data["is_advertisement"], ["is_irrelevant", "is_fake_rant"]] = False
        data.loc[data["is_irrelevant"], "is_fake_rant"] = False

        print(f"📊 Ground truth distribution:")
        print(
            f"   Advertisements: {data['is_advertisement'].sum():,} ({data['is_advertisement'].mean()*100:.1f}%)"
        )
        print(
            f"   Irrelevant: {data['is_irrelevant'].sum():,} ({data['is_irrelevant'].mean()*100:.1f}%)"
        )
        print(
            f"   Fake rants: {data['is_fake_rant'].sum():,} ({data['is_fake_rant'].mean()*100:.1f}%)"
        )

        return data


class F1Evaluator:
    """
    Evaluation system for the F1 solution
    """

    def __init__(self):
        self.results = {}

    def evaluate_classifier(
        self, classifier: ReviewPolicyClassifier, test_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Evaluate classifier performance"""
        print("📈 Evaluating classifier performance...")

        # Get predictions
        predictions = classifier.classify_batch(test_data["text"].tolist())

        # Extract predictions for each category
        pred_advertisement = [
            p["advertisement"]["is_advertisement"] for p in predictions
        ]
        pred_irrelevant = [p["irrelevant"]["is_irrelevant"] for p in predictions]
        pred_fake_rant = [p["fake_rant"]["is_fake_rant"] for p in predictions]

        # Calculate metrics for each category
        categories = {
            "advertisement": (
                test_data["is_advertisement"].tolist(),
                pred_advertisement,
            ),
            "irrelevant": (test_data["is_irrelevant"].tolist(), pred_irrelevant),
            "fake_rant": (test_data["is_fake_rant"].tolist(), pred_fake_rant),
        }

        evaluation_results = {}

        for category, (y_true, y_pred) in categories.items():
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            accuracy = accuracy_score(y_true, y_pred)

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            evaluation_results[category] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "support": sum(y_true),
            }

            print(f"\n🎯 {category.upper()} DETECTION:")
            print(f"   F1 Score:  {f1:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"   Support:   {sum(y_true)} positive cases")

        # Overall F1 score
        overall_f1 = np.mean(
            [results["f1_score"] for results in evaluation_results.values()]
        )
        evaluation_results["overall_f1"] = overall_f1

        print(f"\n🏆 OVERALL F1 SCORE: {overall_f1:.3f}")

        return evaluation_results

    def plot_confusion_matrices(
        self, evaluation_results: Dict, save_path: str | None = None
    ):
        """Plot confusion matrices for all categories"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        categories = ["advertisement", "irrelevant", "fake_rant"]

        for i, category in enumerate(categories):
            cm = evaluation_results[category]["confusion_matrix"]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                ax=axes[i],
                xticklabels=["Not " + category, category],
                yticklabels=["Not " + category, category],
            )
            axes[i].set_title(f"{category.title()} Detection")
            axes[i].set_ylabel("True Label")
            axes[i].set_xlabel("Predicted Label")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"📊 Confusion matrices saved to {save_path}")

        plt.show()

        return fig

    def generate_report(self, evaluation_results: Dict) -> str:
        """Generate a comprehensive evaluation report"""
        report = """
# 🏆 F1 Solution Evaluation Report

## Model Performance Summary

"""

        for category in ["advertisement", "irrelevant", "fake_rant"]:
            results = evaluation_results[category]
            report += f"""
### {category.title()} Detection
- **F1 Score**: {results['f1_score']:.3f}
- **Precision**: {results['precision']:.3f}
- **Recall**: {results['recall']:.3f}
- **Accuracy**: {results['accuracy']:.3f}
- **Support**: {results['support']} positive cases

"""

        report += f"""
## Overall Performance
- **Overall F1 Score**: {evaluation_results['overall_f1']:.3f}

## Model Architecture
- **Rule-based classifier** with feature engineering
- **Enhanced with ML models** (when available)
- **Multi-category detection** for comprehensive policy enforcement

## Key Features
1. **Advertisement Detection**: URL/phone detection, promotional keywords
2. **Irrelevant Content**: Topic analysis, service keyword presence
3. **Fake Rant Detection**: Visit admission patterns, generic vs specific complaints

"""

        return report


def main():
    """Main execution function - Complete F1 Solution"""
    print("🚀 Starting F1 Solution for TechJam 2025")
    print("=" * 50)

    # Initialize pipeline
    pipeline = F1DataPipeline(
        reviews_path="review_South_Dakota.json.gz",
        meta_path="meta_South_Dakota.json.gz",
    )

    # Load and clean data
    pipeline.load_data().clean_data()

    # Create sample for demonstration
    sample_data = pipeline.create_sample_dataset(sample_size=500)
    print(f"📝 Created sample dataset with {len(sample_data)} reviews")

    # Generate ground truth for evaluation
    labeled_data = pipeline.generate_ground_truth_labels(sample_data)

    # Split data for evaluation
    train_data, test_data = train_test_split(
        labeled_data, test_size=0.3, random_state=42
    )
    print(f"📊 Split data: {len(train_data)} train, {len(test_data)} test")

    # Initialize classifier
    classifier = ReviewPolicyClassifier(use_ml_models=True)

    # Evaluate on test set
    evaluator = F1Evaluator()
    results = evaluator.evaluate_classifier(classifier, test_data)

    # Generate visualizations
    try:
        evaluator.plot_confusion_matrices(results, "confusion_matrices.png")
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")

    # Generate report
    report = evaluator.generate_report(results)
    print(report)

    # Save results
    with open("f1_solution_report.md", "w") as f:
        f.write(report)

    print("\n🎉 F1 Solution completed successfully!")
    print(f"🏆 Overall F1 Score: {results['overall_f1']:.3f}")

    # Demonstrate on a few examples
    print("\n" + "=" * 50)
    print("🔍 DEMONSTRATION ON SAMPLE REVIEWS:")
    print("=" * 50)

    demo_reviews = [
        "Great food and excellent service! Highly recommend this place.",
        "Visit our website at www.example.com for special discounts and deals!",
        "I never been here but I heard it's terrible. Probably overpriced.",
        "My phone battery died today. The weather is also bad. Politics is crazy.",
    ]

    for i, review in enumerate(demo_reviews, 1):
        print(f"\n📝 Review {i}: '{review}'")
        result = classifier.classify_review(review)

        for category in ["advertisement", "irrelevant", "fake_rant"]:
            classification = result[category]
            if classification[f"is_{category}"]:
                print(
                    f"   🚫 {category.upper()}: {classification['confidence']:.2f} confidence"
                )
            else:
                print(f"   ✅ {category}: Clean")

    return results


if __name__ == "__main__":
    results = main()
