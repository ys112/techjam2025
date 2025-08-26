#!/usr/bin/env python3
"""
Demo script for Fast Embedding Classifier
High-performance review classification using SentenceTransformers + XGBoost
"""

import sys
import os

sys.path.append("/Users/lunlun/Downloads/Github/techjam2025")

from qwen_model_functions import (
    FastEmbeddingClassifier,
    create_sample_dataset,
    evaluate_results,
    plot_results,
)
import pandas as pd
import numpy as np


def main():
    print("🚀 Fast Embedding Classifier Demo")
    print("=" * 50)

    # Create sample dataset
    print("\n1️⃣ Creating sample dataset...")
    df = create_sample_dataset(size=1000)

    print(f"✅ Dataset created: {df.shape}")
    print(f"   - Spam: {df['is_spam'].sum()} ({df['is_spam'].mean():.1%})")
    print(
        f"   - Ads: {df['is_advertisements'].sum()} ({df['is_advertisements'].mean():.1%})"
    )
    print(
        f"   - Rants: {df['is_rant_without_visit'].sum()} ({df['is_rant_without_visit'].mean():.1%})"
    )

    # Initialize classifier
    print("\n2️⃣ Initializing FastEmbeddingClassifier...")
    classifier = FastEmbeddingClassifier(embedding_model="all-MiniLM-L6-v2")

    # Prepare features
    print("\n3️⃣ Preparing features...")
    X = classifier.prepare_features(df)

    # Prepare targets
    y_dict = {
        "is_spam": df["is_spam"].values.astype(int),
        "is_advertisements": df["is_advertisements"].values.astype(int),
        "is_rant_without_visit": df["is_rant_without_visit"].values.astype(int),
    }

    print(f"✅ Features prepared: {X.shape}")

    # Train classifiers
    print("\n4️⃣ Training XGBoost classifiers...")
    results = classifier.train_classifiers(X, y_dict, test_size=0.2)

    # Evaluate results
    print("\n5️⃣ Results:")
    evaluate_results(results)

    # Test on examples
    print("\n6️⃣ Testing on examples:")
    test_examples = [
        {
            "text": "Great food and service! Highly recommend.",
            "rating": 5,
            "category": "Restaurant",
            "state": "CA",
            "pics": True,
            "resp": False,
            "avg_rating": 4.5,
            "num_of_reviews": 150,
            "price_level": 2,
            "user_id": "test1",
            "user_name": "Test User",
            "time": 1600000000,
            "gmap_id": "test",
            "biz_name": "Test Biz",
            "description": "Test",
            "hours": "9-5",
            "address": "Test St",
        },
        {
            "text": "OMG BEST PLACE EVER!!! AMAZING!!!",
            "rating": 5,
            "category": "Restaurant",
            "state": "NY",
            "pics": False,
            "resp": False,
            "avg_rating": 4.0,
            "num_of_reviews": 80,
            "price_level": 1,
            "user_id": "test2",
            "user_name": "Test User 2",
            "time": 1600000000,
            "gmap_id": "test2",
            "biz_name": "Test Biz 2",
            "description": "Test",
            "hours": "9-5",
            "address": "Test St",
        },
        {
            "text": "Visit our website at www.restaurant.com for deals!",
            "rating": 5,
            "category": "Restaurant",
            "state": "TX",
            "pics": False,
            "resp": True,
            "avg_rating": 4.3,
            "num_of_reviews": 200,
            "price_level": 2,
            "user_id": "test3",
            "user_name": "Test User 3",
            "time": 1600000000,
            "gmap_id": "test3",
            "biz_name": "Test Biz 3",
            "description": "Test",
            "hours": "9-5",
            "address": "Test St",
        },
    ]

    test_df = pd.DataFrame(test_examples)
    predictions = classifier.predict(test_df)

    for i, example in enumerate(test_examples):
        print(f"\n   Example {i+1}: '{example['text'][:50]}...'")
        detected = []
        if predictions["is_spam"][i]:
            detected.append("Spam")
        if predictions["is_advertisements"][i]:
            detected.append("Advertisement")
        if predictions["is_rant_without_visit"][i]:
            detected.append("Rant")

        print(f"   Detected: {detected if detected else ['Clean']}")

    print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    main()
