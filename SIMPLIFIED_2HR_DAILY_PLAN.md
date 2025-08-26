# 📅 Simplified 2-Hour Daily Plan (3 Days Total - 72-Hour Challenge)

## 🎯 Overview

A focused 3-day approach with 2 hours per day, aligned with the 72-hour challenge workflow. Each day corresponds to a major milestone while maintaining manageable daily commitments.

---

## Day 1 (Aug 25): Data Pipeline Setup & Initial Modeling (2 hours)

### Hour 1: Environment & Data Collection

- [ ] **15 min**: Set up Google Colab or local Python environment
- [ ] **30 min**: Download and explore Google Reviews dataset (primary data source)
- [ ] **15 min**: Install required packages (transformers, pandas, scikit-learn, matplotlib)

### Hour 2: Prompt Engineering & Prototype Pipeline

- [ ] **30 min**: Design prompts for each policy violation category:
  - Advertisement detection
  - Irrelevant content identification
  - Rants without visit detection
- [ ] **30 min**: Build basic Hugging Face classification pipeline (Gemini, Qwen, or DistilBERT)

**End of Day 1 Goal**: Data pipeline ready, prototype classification system with HuggingFace models

---

## Day 2 (Aug 26): Evaluation & Label Refinement (2 hours)

### Hour 1: Result Analysis & Model Evaluation

- [ ] **30 min**: Run inference on collected reviews using fine-tuned models or few-shot LLMs
- [ ] **30 min**: Calculate precision, recall, and F1-score for each violation type

### Hour 2: Handle Missing Labels & Iterate

- [ ] **30 min**: Generate pseudo-labels using advanced LLM (GPT-4o) or manual annotation subset
- [ ] **30 min**: Refine prompts, adjust model thresholds, and improve labeling strategies

**End of Day 2 Goal**: Evaluated model performance, refined classification approach with improved accuracy

---

## Day 3 (Aug 27): Complete Deliverables for Submission (2 hours)

### Hour 1: Final Model & Documentation

- [ ] **30 min**: Finalize best-performing model and create deployment pipeline
- [ ] **30 min**: Create comprehensive README and technical documentation

### Hour 2: Demo & Submission Preparation

- [ ] **30 min**: Create interactive demo (Streamlit app or Jupyter notebook)
- [ ] **15 min**: Record demo video (5-7 minutes)
- [ ] **15 min**: Complete final submission package and submit to Devpost

**End of Day 3 Goal**: Complete submission with all deliverables ready for presentation

---

## 🎯 Daily Success Checkpoints

### Minimum Viable Progress

- **Day 1**: Data pipeline operational, HuggingFace prototype built with prompt engineering
- **Day 2**: Model evaluation complete, improved labeling strategy implemented
- **Day 3**: All deliverables complete and submitted

### 🚨 If Running Behind

- **Day 1**: Focus on basic HuggingFace pipeline over complex prompt engineering
- **Day 2**: Use manual annotation subset instead of advanced LLM pseudo-labeling
- **Day 3**: Prioritize core deliverables over demo polish

## 🛠️ Tools & Technologies Stack

### Core Technologies
- **Programming**: Python (primary)
- **NLP Libraries**: HuggingFace Transformers, spaCy, NLTK
- **ML Frameworks**: scikit-learn, PyTorch
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, Plotly
- **Models**: Gemini, Qwen, DistilBERT, GPT-4o (for pseudo-labeling)

## 💡 Daily Tips

### Day 1: Pipeline Foundation
- Focus on data collection from Google Reviews dataset
- Design clear prompts for each policy violation category
- Build functional HuggingFace prototype (fine-tuned or few-shot)

### Day 2: Evaluation & Refinement
- Compare model predictions against available ground truth
- Use GPT-4o for pseudo-labeling if ground truth missing
- Iterate on prompts and thresholds based on evaluation metrics

### Day 3: Deliverables & Submission
- Finalize best-performing approach
- Create comprehensive documentation
- Ensure demo showcases key capabilities

## 🏆 Success Strategy

### Core Principles

1. **Structured Workflow**: Follow the 72-hour challenge methodology
2. **Data-Driven Approach**: Focus on Google Reviews dataset as primary source
3. **Evaluation-Centric**: Measure precision, recall, F1-score for each violation type
4. **Iterative Refinement**: Use Day 2 evaluation to improve Day 1 prototype

### Competitive Advantage

- **Methodical Approach**: Structured 3-day workflow aligns with challenge requirements
- **Technical Depth**: Comprehensive use of HuggingFace, prompt engineering, and evaluation metrics
- **Quality Focus**: Dedicated evaluation day ensures robust performance measurement
- **Complete Deliverables**: Final day ensures all submission requirements met

### Key Success Factors

- **Day 1**: Solid foundation with functional pipeline and initial modeling
- **Day 2**: Rigorous evaluation and refinement based on metrics
- **Day 3**: Professional delivery with complete documentation and demo

Remember: This 72-hour challenge is structured for systematic progress. 2 hours of focused work per day for 3 days = 6 total concentrated hours, perfectly aligned with the challenge workflow for a competitive solution!
