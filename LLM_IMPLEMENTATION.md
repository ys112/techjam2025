# LLM-Based Review Classification Implementation

This implementation updates the TechJam 2025 starter notebook to use real language models for review classification instead of simple rule-based approaches.

## Key Features

### 🤖 Real Language Model Integration
- **Primary Model**: DistilBERT for sentiment analysis combined with heuristics
- **Alternative Models**: Support for text generation models (T5, GPT-2, etc.)
- **Fallback System**: Rule-based classification when models fail to load

### 🎯 Policy Violation Detection
The system classifies reviews for three types of violations:

1. **Advertisement**: Reviews containing promotional content or links
2. **Irrelevant Content**: Reviews not related to the business location  
3. **Fake Rants**: Complaints from users who likely never visited

### 🔧 Technical Implementation

#### Model Architecture
```python
class ReviewClassifier:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        # Supports multiple model types with automatic fallback
        
    def classify_review(self, text: str) -> dict:
        # Returns: {'advertisement': bool, 'irrelevant': bool, 'fake_rant': bool}
        
    def classify_batch(self, texts: list) -> list:
        # Batch processing with progress indicators
```

#### Prompt Engineering
For text generation models, the system uses carefully crafted prompts:

```
Task: Determine if this review contains advertisements or promotional content.

Examples of ADVERTISEMENTS:
- "Great food! Visit www.discount-deals.com for coupons!"
- "Call 555-1234 for catering services!"

Examples of NOT ADVERTISEMENTS:
- "The food was delicious and service was great"
- "Terrible experience, would not recommend"

Review to analyze: "{review_text}"

Is this review an ADVERTISEMENT? Answer only YES or NO:
```

#### Hybrid Classification Approach
For BERT-style models, the system combines:
- **Sentiment Analysis**: Base model predictions
- **Heuristic Rules**: Keyword matching for specific violations
- **Contextual Logic**: Different strategies per violation type

## Installation & Usage

### Prerequisites
```bash
pip install transformers torch pandas numpy matplotlib seaborn scikit-learn
```

### Running the Notebook
1. Open `TechJam_2025_Starter_Notebook.ipynb`
2. Install required packages in the first cell
3. Run all cells sequentially

### Testing the Implementation
```bash
python test_llm_classifier.py
```

## Model Comparison Results

The notebook includes comparison of different models:

| Model | Type | Speed | Accuracy | Resource Usage |
|-------|------|-------|----------|----------------|
| DistilBERT | Classification | Fast | Good | Low |
| T5-Small | Text-to-Text | Medium | Better | Medium |
| DialoGPT | Generation | Slow | Best | High |

## Performance Metrics

The system evaluates performance using:
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual violations
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Fallback Strategy

If Hugging Face models fail to load, the system automatically falls back to rule-based classification using keyword matching:

```python
ad_keywords = ['visit', 'website', 'www', 'http', 'call', 'phone', 'discount']
irrelevant_keywords = ['my phone', 'my car', 'politics', 'weather', 'traffic']
fake_rant_keywords = ['never been', 'heard it', 'looks like', 'probably']
```

## Next Steps for Production

1. **Fine-tuning**: Train models on domain-specific review data
2. **Ensemble Methods**: Combine multiple model predictions
3. **Real-time Processing**: Optimize for production workloads
4. **A/B Testing**: Compare model performance on live data

## Troubleshooting

### Common Issues

**Model Loading Fails**
- Check internet connection for model downloads
- Ensure sufficient memory (4GB+ recommended)
- Try smaller models like DistilBERT first

**Slow Performance**
- Use CPU-optimized models (`device=-1`)
- Implement batch processing
- Consider model quantization

**Poor Accuracy**
- Adjust keyword lists for your domain
- Experiment with different base models
- Implement ensemble voting

## Contributing

To improve the classifier:
1. Add more training examples
2. Expand keyword lists
3. Test additional model architectures
4. Implement active learning for continuous improvement

## License

This implementation follows the original TechJam 2025 project license.