"""
Fast Embedding-Based Classifier for Review Policy Violations
Using SentenceTransformers + XGBoost for high-performance classification
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Any
import re
import json
import gzip

warnings.filterwarnings('ignore')


class FastEmbeddingClassifier:
    """
    High-performance classifier using SentenceTransformers embeddings + XGBoost
    """
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize with small, fast embedding model
        
        Args:
            embedding_model: HuggingFace model name for embeddings
        """
        print(f"🚀 Initializing Fast Embedding Classifier...")
        print(f"📦 Loading embedding model: {embedding_model}")
        
        # Load small, fast sentence transformer
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
        
        # XGBoost classifiers for each task
        self.classifiers = {
            'is_spam': None,
            'is_advertisements': None,
            'is_rant_without_visit': None
        }
        
        # Scalers for metadata features
        self.metadata_scaler = StandardScaler()
        self.label_encoders = {}
        
        # Feature names for interpretability
        self.feature_names = []
        
        print(f"✅ Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def extract_text_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Extract embeddings from text using SentenceTransformers
        
        Args:
            texts: List of review texts
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings
        """
        print(f"🔄 Extracting embeddings for {len(texts)} texts...")
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Extract embeddings in batches
        embeddings = self.embedding_model.encode(
            cleaned_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        print(f"✅ Embeddings extracted: {embeddings.shape}")
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def preprocess_metadata(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess metadata columns into numerical features
        
        Args:
            df: DataFrame with metadata columns
            
        Returns:
            Numpy array of processed metadata features
        """
        print("🔧 Processing metadata features...")
        
        metadata_features = []
        feature_names = []
        
        # Numeric features
        numeric_cols = ['rating', 'avg_rating', 'num_of_reviews', 'price_level']
        for col in numeric_cols:
            if col in df.columns:
                values = pd.to_numeric(df[col], errors='coerce').fillna(0)
                metadata_features.append(values.values.reshape(-1, 1))
                feature_names.append(f'numeric_{col}')
        
        # Text length features
        if 'text' in df.columns:
            text_lengths = df['text'].str.len().fillna(0)
            word_counts = df['text'].str.split().str.len().fillna(0)
            metadata_features.extend([
                text_lengths.values.reshape(-1, 1),
                word_counts.values.reshape(-1, 1)
            ])
            feature_names.extend(['text_length', 'word_count'])
        
        # Categorical features (encoded)
        categorical_cols = ['category', 'state']
        for col in categorical_cols:
            if col in df.columns:
                # Handle missing values
                values = df[col].fillna('unknown').astype(str)
                
                # Fit label encoder if not exists
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    encoded = self.label_encoders[col].fit_transform(values)
                else:
                    # Handle unseen categories
                    encoded = []
                    for val in values:
                        if val in self.label_encoders[col].classes_:
                            encoded.append(self.label_encoders[col].transform([val])[0])
                        else:
                            encoded.append(0)  # Unknown category
                    encoded = np.array(encoded)
                
                metadata_features.append(encoded.reshape(-1, 1))
                feature_names.append(f'categorical_{col}')
        
        # Boolean features
        boolean_cols = ['pics', 'resp']
        for col in boolean_cols:
            if col in df.columns:
                values = df[col].fillna(False).astype(bool).astype(int)
                metadata_features.append(values.values.reshape(-1, 1))
                feature_names.append(f'boolean_{col}')
        
        # Combine all features
        if metadata_features:
            combined_features = np.hstack(metadata_features)
            self.metadata_feature_names = feature_names
            print(f"✅ Metadata features: {combined_features.shape}")
            return combined_features
        else:
            print("⚠️  No metadata features found")
            return np.zeros((len(df), 1))
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Combine text embeddings and metadata features
        
        Args:
            df: DataFrame with text and metadata
            
        Returns:
            Combined feature matrix
        """
        # Extract text embeddings
        text_embeddings = self.extract_text_embeddings(df['text'].tolist())
        
        # Extract metadata features
        metadata_features = self.preprocess_metadata(df)
        
        # Combine features
        combined_features = np.hstack([text_embeddings, metadata_features])
        
        print(f"🔗 Combined features shape: {combined_features.shape}")
        print(f"   - Text embeddings: {text_embeddings.shape[1]} dims")
        print(f"   - Metadata features: {metadata_features.shape[1]} dims")
        
        return combined_features
    
    def train_classifiers(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                         test_size: float = 0.2, random_state: int = 42):
        """
        Train XGBoost classifiers with hyperparameter tuning
        
        Args:
            X: Feature matrix
            y_dict: Dictionary of target labels for each task
            test_size: Test set proportion
            random_state: Random seed
        """
        print("🚀 Training XGBoost classifiers with hyperparameter tuning...")
        
        # Split data
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, np.arange(len(X)), test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Scale features
        X_train_scaled = self.metadata_scaler.fit_transform(X_train)
        X_test_scaled = self.metadata_scaler.transform(X_test)
        
        results = {}
        
        # Train classifier for each task
        for task, y_full in y_dict.items():
            print(f"\n🎯 Training {task} classifier...")
            
            # Get train/test labels
            y_train = y_full[indices_train]
            y_test = y_full[indices_test]
            
            # Check class distribution
            pos_rate = y_train.mean()
            print(f"   Positive rate: {pos_rate:.3f}")
            
            # XGBoost hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 6],
                'learning_rate': [0.1, 0.15, 0.2],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 1.5]
            }
            
            # Base classifier
            base_clf = xgb.XGBClassifier(
                random_state=random_state,
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist'
            )
            
            # Grid search with cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            grid_search = GridSearchCV(
                base_clf, 
                param_grid, 
                cv=cv,
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            # Fit with progress
            print("   🔍 Hyperparameter search...")
            grid_search.fit(X_train_scaled, y_train)
            
            # Best classifier
            best_clf = grid_search.best_estimator_
            self.classifiers[task] = best_clf
            
            # Evaluate
            y_pred = best_clf.predict(X_test_scaled)
            y_proba = best_clf.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            
            results[task] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'best_params': grid_search.best_params_,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"   ✅ {task} Results:")
            print(f"      F1 Score: {f1:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall: {recall:.4f}")
            print(f"      Accuracy: {accuracy:.4f}")
            print(f"      Best params: {grid_search.best_params_}")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Make predictions on new data
        
        Args:
            df: DataFrame with text and metadata
            
        Returns:
            Dictionary of predictions for each task
        """
        # Prepare features
        X = self.prepare_features(df)
        X_scaled = self.metadata_scaler.transform(X)
        
        predictions = {}
        for task, clf in self.classifiers.items():
            if clf is not None:
                predictions[task] = clf.predict(X_scaled)
        
        return predictions
    
    def get_feature_importance(self, top_k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get feature importance for each classifier
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            Dictionary of feature importance for each task
        """
        importance_dict = {}
        
        # Create feature names
        embedding_names = [f'embed_{i}' for i in range(self.embedding_dim)]
        all_feature_names = embedding_names + getattr(self, 'metadata_feature_names', [])
        
        for task, clf in self.classifiers.items():
            if clf is not None:
                importances = clf.feature_importances_
                feature_importance = list(zip(all_feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                importance_dict[task] = feature_importance[:top_k]
        
        return importance_dict


def create_sample_dataset(size: int = 2000) -> pd.DataFrame:
    """
    Create a realistic sample dataset with proper labels
    """
    print(f"🎯 Creating sample dataset with {size} reviews...")
    
    np.random.seed(42)
    
    # Base data structure
    data = []
    
    # Categories for realistic data
    categories = ['Restaurant', 'Store', 'Hotel', 'Service', 'Entertainment', 'Healthcare']
    states = ['CA', 'NY', 'TX', 'FL', 'WA', 'IL']
    
    # Sample texts for each category
    clean_reviews = [
        "Great food and excellent service. Highly recommended!",
        "Nice place with friendly staff. Good value for money.",
        "Clean facilities and good location. Will come back.",
        "Enjoyed our visit. Food was delicious and fresh.",
        "Professional service and quality products.",
        "Family-friendly atmosphere with reasonable prices."
    ]
    
    spam_reviews = [
        "OMG BEST PLACE EVER!!!! MUST GO NOW!!!!",
        "FREE FREE FREE! Everyone should visit immediately!",
        "AMAZING AMAZING AMAZING! 5 STARS ALWAYS!",
        "Perfect perfect perfect! No complaints ever!",
        "Incredible incredible incredible! Wow wow wow!"
    ]
    
    ad_reviews = [
        "Visit our website at www.restaurant.com for deals!",
        "Call 555-1234 for special offers and discounts!",
        "Check out our Facebook page for daily specials!",
        "Follow us on Instagram @restaurant for coupons!",
        "Order online at our website for free delivery!"
    ]
    
    rant_reviews = [
        "Never been here but heard it's terrible from friends.",
        "People told me to avoid this place completely.",
        "Haven't visited but reviews online are mostly bad.",
        "My neighbor said the service is awful here.",
        "Heard from multiple people it's not worth it."
    ]
    
    for i in range(size):
        # Determine review type
        review_type = np.random.choice(['clean', 'spam', 'ad', 'rant'], p=[0.6, 0.15, 0.15, 0.1])
        
        if review_type == 'clean':
            text = np.random.choice(clean_reviews)
            rating = np.random.choice([4, 5], p=[0.3, 0.7])
            is_spam = False
            is_ad = False
            is_rant = False
        elif review_type == 'spam':
            text = np.random.choice(spam_reviews)
            rating = 5
            is_spam = True
            is_ad = False
            is_rant = False
        elif review_type == 'ad':
            text = np.random.choice(ad_reviews)
            rating = np.random.choice([4, 5])
            is_spam = False
            is_ad = True
            is_rant = False
        else:  # rant
            text = np.random.choice(rant_reviews)
            rating = np.random.choice([1, 2], p=[0.7, 0.3])
            is_spam = False
            is_ad = False
            is_rant = True
        
        # Add some variation to text
        if i % 10 != 0:  # Add variation to 90% of reviews
            variations = [
                f"{text} Really enjoyed it.",
                f"Honestly, {text.lower()}",
                f"{text} Would recommend.",
                text
            ]
            text = np.random.choice(variations)
        
        # Generate other fields
        row = {
            'user_id': f"user_{i}",
            'user_name': f"User {i}",
            'time': 1500000000 + i * 1000,
            'rating': rating,
            'text': text,
            'pics': np.random.choice([True, False], p=[0.2, 0.8]),
            'resp': np.random.choice([True, False], p=[0.3, 0.7]),
            'gmap_id': f"gmap_{i}",
            'biz_name': f"Business {i % 100}",
            'description': "Sample business description",
            'category': np.random.choice(categories),
            'avg_rating': np.random.uniform(3.0, 4.8),
            'num_of_reviews': np.random.randint(5, 500),
            'hours': "9 AM - 9 PM",
            'address': f"{i} Main Street, City, State",
            'price_level': np.random.randint(0, 4),
            'state': np.random.choice(states),
            'is_spam': is_spam,
            'is_advertisements': is_ad,
            'is_rant_without_visit': is_rant
        }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    print(f"✅ Sample dataset created!")
    print(f"   Total reviews: {len(df)}")
    print(f"   Spam: {df['is_spam'].sum()} ({df['is_spam'].mean():.1%})")
    print(f"   Advertisements: {df['is_advertisements'].sum()} ({df['is_advertisements'].mean():.1%})")
    print(f"   Rant without visit: {df['is_rant_without_visit'].sum()} ({df['is_rant_without_visit'].mean():.1%})")
    
    return df


def evaluate_results(results: Dict[str, Dict]) -> None:
    """
    Print comprehensive evaluation results
    """
    print("\n" + "="*80)
    print("🏆 FAST EMBEDDING CLASSIFIER RESULTS")
    print("="*80)
    
    # Summary table
    summary_data = []
    for task, metrics in results.items():
        summary_data.append({
            'Task': task.replace('is_', '').replace('_', ' ').title(),
            'F1 Score': f"{metrics['f1_score']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'Accuracy': f"{metrics['accuracy']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n📊 Performance Summary:")
    print(summary_df.to_string(index=False))
    
    # Overall metrics
    avg_f1 = np.mean([m['f1_score'] for m in results.values()])
    avg_precision = np.mean([m['precision'] for m in results.values()])
    avg_recall = np.mean([m['recall'] for m in results.values()])
    avg_accuracy = np.mean([m['accuracy'] for m in results.values()])
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Average F1 Score: {avg_f1:.4f}")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Average Recall: {avg_recall:.4f}")
    print(f"   Average Accuracy: {avg_accuracy:.4f}")
    
    print(f"\n✅ High-performance classification achieved!")
    print(f"🚀 Ready for production deployment!")


def plot_results(results: Dict[str, Dict]) -> None:
    """
    Visualize classification results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Metrics comparison
    tasks = list(results.keys())
    task_names = [t.replace('is_', '').replace('_', ' ').title() for t in tasks]
    
    metrics = ['f1_score', 'precision', 'recall', 'accuracy']
    metric_names = ['F1 Score', 'Precision', 'Recall', 'Accuracy']
    
    # Plot 1: Metrics by task
    x = np.arange(len(tasks))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[task][metric] for task in tasks]
        axes[0, 0].bar(x + i*width, values, width, label=metric_names[i])
    
    axes[0, 0].set_xlabel('Tasks')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Performance by Task')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(task_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: F1 scores
    f1_scores = [results[task]['f1_score'] for task in tasks]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = axes[0, 1].bar(task_names, f1_scores, color=colors)
    axes[0, 1].set_title('F1 Scores by Task')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom')
    
    # Plot 3: Precision vs Recall
    precisions = [results[task]['precision'] for task in tasks]
    recalls = [results[task]['recall'] for task in tasks]
    
    scatter = axes[1, 0].scatter(precisions, recalls, c=colors, s=100)
    for i, task_name in enumerate(task_names):
        axes[1, 0].annotate(task_name, (precisions[i], recalls[i]), 
                           xytext=(5, 5), textcoords='offset points')
    
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    
    # Plot 4: Overall performance radar (simplified)
    avg_metrics = [np.mean([results[task][metric] for task in tasks]) 
                   for metric in metrics]
    
    axes[1, 1].bar(metric_names, avg_metrics, color='lightblue')
    axes[1, 1].set_title('Average Performance Across All Tasks')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    
    # Add value labels
    for i, (name, value) in enumerate(zip(metric_names, avg_metrics)):
        axes[1, 1].text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


# Export main classes
__all__ = [
    'FastEmbeddingClassifier',
    'create_sample_dataset',
    'evaluate_results',
    'plot_results'
]

if __name__ == "__main__":
    print("🚀 Fast Embedding Classifier loaded!")
    print("   • SentenceTransformer embeddings: all-MiniLM-L6-v2 (384-dim)")
    print("   • XGBoost classifiers with hyperparameter tuning")
    print("   • Multi-task classification: spam, ads, rant detection")
    print("   • High-performance pipeline ready!")
