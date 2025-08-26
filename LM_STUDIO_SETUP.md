# LM Studio + Gemma 1.2 Setup Guide

## 🚀 Quick Setup Instructions

### 1. Install LM Studio

1. Download LM Studio from https://lmstudio.ai/
2. Install and run LM Studio
3. Go to the "Discover" tab and search for "Gemma"
4. Download a Gemma 1.2 model (recommended: `gemma-2-2b-it` for speed)

### 2. Start LM Studio Server

1. In LM Studio, go to "Local Server" tab
2. Load your downloaded Gemma model
3. Start the server (default: http://localhost:1234)
4. Note the model name shown in the interface

### 3. Update Your Code

```python
# Initialize classifier with your specific model name
classifier = ReviewPolicyClassifier(
    use_ml_models=True,
    lm_studio_url="http://localhost:1234/v1",  # Default LM Studio URL
    model_name="google/gemma-3-12b"  # Replace with your actual model name
)
```

### 4. Install Dependencies

```bash
pip install openai
```

### 5. Run the Enhanced Solution

```bash
python f1_solution_lmstudio.py
```

## 🔧 Configuration Options

### Custom LM Studio Configuration

```python
classifier = ReviewPolicyClassifier(
    use_ml_models=True,
    lm_studio_url="http://localhost:1234/v1",  # Change port if needed
    model_name="your-actual-model-name"        # Check LM Studio interface
)
```

### Common Model Names

- `gemma-2-2b-it` - Smaller, faster
- `gemma-2-9b-it` - Larger, more accurate
- Check LM Studio interface for exact name

### Troubleshooting

**Connection Failed?**

- Ensure LM Studio is running
- Check the server URL in LM Studio interface
- Verify the model is loaded and active
- Check firewall settings

**Model Name Issues?**

- Copy the exact model name from LM Studio interface
- Model names are case-sensitive

**Performance Issues?**

- Use smaller models (2B instead of 9B)
- Reduce the sample size in main() function
- The system automatically falls back to rule-based if LM Studio fails

## 📊 Expected Performance Improvements

With Gemma 1.2, you should see:

- Better nuanced understanding of promotional language
- Improved detection of subtle irrelevant content
- More accurate fake rant identification
- Higher overall F1 scores compared to rule-based only

## 🔄 Fallback Behavior

The system gracefully falls back:

1. **Primary**: LM Studio + Gemma 1.2 (70% weight)
2. **Secondary**: Rule-based features (30% weight)
3. **Fallback**: Hugging Face transformers (if LM Studio unavailable)
4. **Final fallback**: Pure rule-based classification

This ensures your solution works even if LM Studio is not available.
