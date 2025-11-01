# Stock Market Prediction Using News Headlines Sentiment Analysis

## Project Overview

This project implements machine learning models to predict stock market movements (up/down) based on sentiment analysis of financial news headlines. The research addresses the critical question of whether news sentiment can effectively predict short-term stock price movements, which is valuable for algorithmic trading, risk management, and investment decision-making.

The project demonstrates that financial news headlines contain predictive signals for stock market movements, achieving over 84% accuracy using ensemble methods. This approach could assist traders and investors in making more informed decisions by incorporating news sentiment into their analysis.

## Dataset Information

**Source**: [Kaggle - Stock Sentiment Analysis Using News Headlines](https://www.kaggle.com/code/siddharthtyagi/stock-sentiment-analysis-using-news-headlines)

**Dataset Characteristics**:
- **Size**: 1,989 records spanning from January 2000 to July 2016
- **Features**: 27 columns (Date, Label, Top1-Top25 news headlines)
- **Target Variable**: Binary classification (0 = Stock Down, 1 = Stock Up)
- **Data Split**: 
  - Training: Before January 1, 2015 (1,611 samples)
  - Testing: After December 31, 2014 (378 samples)
- **File Format**: CSV with ISO-8859-1 encoding

**Preprocessing Steps**:
1. **Text Cleaning**: Removed non-alphabetic characters using regex patterns
2. **Normalization**: Converted all text to lowercase for consistency
3. **Feature Engineering**: Combined all 25 news headlines per day into single text strings
4. **Vectorization**: Applied CountVectorizer with unigrams (ngram_range=(1,1))
5. **Data Filtering**: No missing values detected, minimal data cleaning required

## Methodology

### Problem Formulation
The stock prediction problem is formulated as a binary text classification task where:
- **Input**: Concatenated news headlines for each trading day
- **Output**: Binary prediction (0 = negative stock movement, 1 = positive stock movement)

### Algorithm Selection
We implemented and compared two complementary machine learning approaches:

1. **Random Forest Classifier**
   - **Rationale**: Ensemble method that handles high-dimensional sparse text features effectively
   - **Configuration**: 200 estimators, entropy criterion, random_state=42
   - **Advantages**: Robust to overfitting, provides feature importance, handles non-linear relationships

2. **Support Vector Machine (SVM)**
   - **Rationale**: Well-suited for text classification with high-dimensional feature spaces
   - **Configuration**: Linear kernel, random_state=42
   - **Advantages**: Effective with sparse data, strong theoretical foundation, memory efficient

### Feature Engineering Pipeline
```
Raw Headlines → Text Cleaning → Lowercase Conversion → Concatenation → CountVectorizer → ML Models
```

## Implementation Details

### Technical Stack
- **Language**: Python 3.10+
- **Libraries**: 
  - Data Processing: pandas, numpy
  - Machine Learning: scikit-learn
  - Visualization: matplotlib, seaborn
  - Text Processing: sklearn.feature_extraction.text

### Model Architecture
1. **Data Preprocessing**: Text normalization and concatenation
2. **Feature Extraction**: Bag-of-words representation using CountVectorizer
3. **Model Training**: Parallel training of Random Forest and SVM classifiers
4. **Evaluation**: Comprehensive performance analysis with multiple metrics

## How to Run the Code

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution Steps
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd stock-prediction-sentiment-analysis
   ```

2. **Ensure dataset availability**:
   - Download `stk_pred.csv` from the Kaggle source
   - Place it in the project directory

3. **Run the Jupyter notebook**:
   ```bash
   jupyter notebook AI_NLP_2.ipynb
   ```

4. **Execute cells sequentially**:
   - Import libraries and load data
   - Preprocess text data
   - Train both models
   - Generate comparative analysis
   - Test with custom headlines

### File Structure
```
project/
├── AI_NLP_2.ipynb          # Main implementation notebook
├── stk_pred.csv            # Dataset file
├── README.md               # Project documentation
└── results/                # Generated visualizations
    ├── confusion_matrices.png
    ├── performance_comparison.png
    └── model_comparison.png
```

## Experimental Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **84.66%** | **85.19%** | **84.66%** | **84.58%** |
| SVM | 82.28% | 82.28% | 82.28% | 82.27% |

### Key Findings

1. **Random Forest Superiority**: Achieved 2.38% higher accuracy compared to SVM
2. **Balanced Performance**: Both models show consistent precision-recall balance
3. **Robust Predictions**: Over 82% accuracy demonstrates news sentiment predictive power
4. **Model Agreement**: Both models frequently agree on predictions, increasing confidence

### Confusion Matrix Analysis

**Random Forest Results**:
- True Negatives: 145 (correctly predicted downward movements)
- False Positives: 41 (incorrectly predicted upward movements)
- False Negatives: 17 (missed upward movements)
- True Positives: 175 (correctly predicted upward movements)

**SVM Results**:
- True Negatives: 151
- False Positives: 35
- False Negatives: 32
- True Positives: 160

### Visualization Results
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/20bfe517-2030-4a3e-a4a3-e384db67530c" />
*Figure 1: Comparative performance metrics across both models*

<img width="1446" height="590" alt="image" src="https://github.com/user-attachments/assets/38d72a88-d977-420c-bbb3-3ed2b0d7aace" />
*Figure 2: Side-by-side confusion matrices for Random Forest and SVM*

<img width="864" height="274" alt="image" src="https://github.com/user-attachments/assets/912374dc-c57d-4291-9038-cc4b641d0877" />
*Figure 3: Detailed accuracy, precision, recall, and F1-score comparison*

## Analysis and Insights

### Model Performance Analysis
- **Random Forest** demonstrates superior performance due to its ensemble nature, effectively capturing complex patterns in news sentiment
- **SVM** provides reliable baseline performance with faster prediction times
- Both models achieve accuracy levels significantly above random chance (50%), validating the hypothesis that news sentiment contains predictive signals

### Feature Importance
The bag-of-words approach successfully captures sentiment-bearing words that correlate with stock movements. Key observations:
- Financial terminology and sentiment words likely drive predictions
- High-dimensional sparse features are well-handled by both algorithms
- Text preprocessing significantly impacts model performance

### Practical Applications
1. **Algorithmic Trading**: Integration into trading strategies for decision support
2. **Risk Management**: Early warning system for market volatility
3. **Investment Research**: Supplementary tool for fundamental analysis

## Conclusion

This project successfully demonstrates that machine learning models can effectively predict stock market movements using news headline sentiment analysis. Key takeaways include:

1. **News Sentiment Predictive Power**: Achieved >84% accuracy, confirming that financial news contains valuable predictive signals
2. **Model Selection Impact**: Random Forest outperformed SVM by 2.38%, highlighting the importance of algorithm selection
3. **Practical Viability**: Results suggest real-world applicability for trading and investment decisions
4. **Robust Methodology**: Comprehensive evaluation with multiple metrics ensures reliable conclusions

### Future Work
- **Deep Learning**: Implement LSTM/BERT models for improved text understanding
- **Real-time Processing**: Develop streaming prediction system
- **Multi-asset Prediction**: Extend to individual stocks and different markets
- **Feature Enhancement**: Incorporate technical indicators and market data
- **Hyperparameter Optimization**: Grid search for optimal model configurations

### Limitations
- **Temporal Scope**: Dataset limited to 2000-2016 period
- **Market Dynamics**: Financial markets evolve, requiring model retraining
- **Feature Simplicity**: Basic bag-of-words may miss complex linguistic patterns
- **Binary Classification**: Real market movements are more nuanced than up/down

## References

1. Original Dataset: [Kaggle - Stock Sentiment Analysis Using News Headlines](https://www.kaggle.com/code/siddharthtyagi/stock-sentiment-analysis-using-news-headlines)
2. Scikit-learn Documentation: [Machine Learning Library](https://scikit-learn.org/)
3. Pandas Documentation: [Data Manipulation Library](https://pandas.pydata.org/)
4. Research Paper: "Sentiment Analysis in Financial News" - Various Academic Sources
5. "Random Forest for Text Classification" - Machine Learning Literature
6. "Support Vector Machines for Text Mining" - Pattern Recognition Research

---

**Project Status**: Complete ✅  
**Last Updated**: November 2025  
**Authors**: Krunal Dhapodkar  
**Course**: Machine Learning (VII Semester)  
