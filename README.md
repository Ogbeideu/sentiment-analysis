# Hotel Review Sentiment Analysis

Machine learning project that classifies hotel reviews as "happy" or "not happy" using NLP techniques. Achieves **88.2% accuracy** with logistic regression and TF-IDF vectorization.

## Dataset
- **38,932 hotel reviews** with sentiment labels
- **Class distribution**: 68% positive, 32% negative reviews
- **Features**: Review text, user device/browser data

## Technology Stack
- **Python** - pandas, scikit-learn, matplotlib
- **ML Pipeline** - TF-IDF → Logistic Regression
- **Environment** - Google Colab/Jupyter Notebook

## Performance
| Metric | Score |
|--------|-------|
| Accuracy | 88.24% |
| Precision | 88.91% |
| Recall | 88.24% |
| ROC-AUC | 85.50% |

## Quick Start

```python
# Load and preprocess data
df = pd.read_csv('hotel_review_data.csv')
df['cleaned_text'] = df.Description.apply(text_preprocessing)

# Train model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

model.fit(X_train, y_train)

# Predict
result = model.predict(["Great hotel with excellent service"])
# Output: ['happy']
```

## Key Features
- **Text preprocessing** with regex cleaning
- **TF-IDF vectorization** for feature extraction
- **Pipeline architecture** for reproducible workflow
- **Comprehensive evaluation** with multiple metrics

## Project Structure
```
sentiment-analysis/
├── data/hotel_review_data.csv
├── notebooks/sentiment_analysis.ipynb
├── README.md
└── requirements.txt
```

## Installation
```bash
pip install pandas matplotlib scikit-learn
```

## Usage
1. Upload dataset to Google Drive or local directory
2. Run the Jupyter notebook for full analysis
3. Use trained model for new predictions

## Contact
**Developer**: Ogbeide Uwagboe  
**Email**: silvaworld@yahoo.com 
**LinkedIn**: www.linkedin.com/in/ogbeide-uwagboe

---
*NLP sentiment analysis for hospitality industry insights*
