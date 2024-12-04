
# Predicting Stock Market Movements Using News Headlines

## Overview

This project aims to predict the movements of the Dow Jones Industrial Average (DJIA) using news headlines from Reddit's *r/worldnews* subreddit. By combining natural language processing (NLP) techniques and machine learning models, we explore whether it's possible to forecast stock market trends based on daily news data.

## Motivation

The stock market is influenced by a variety of factors, and news headlines can significantly impact investor sentiment and market trends. This project investigates the predictive power of news data on stock movements, providing insights into the challenges and opportunities in this domain.

## Data

The project uses the following datasets:
- **News Data**: Historical news headlines from *r/worldnews*, ranked by user votes. Each date includes the top 25 headlines.
- **Stock Data**: Daily closing values of the Dow Jones Industrial Average (DJIA), used to label whether the market went up or down.

The combined dataset spans from 2008 to 2016 and is split as follows:
- Training set: Data before January 1, 2015.
- Test set: Data from January 1, 2015, onwards.

## Models

We experimented with three models:
1. **Logistic Regression**:
   - A simple, interpretable baseline model.
   - Features were extracted using TF-IDF vectorization.
2. **Long Short-Term Memory (LSTM)**:
   - A recurrent neural network capable of capturing sequential dependencies in text data.
   - Tokenized input sequences were padded for consistency and fed into an embedding layer.
3. **Bidirectional Encoder Representations from Transformers (BERT)**:
   - A pre-trained transformer model with state-of-the-art NLP capabilities.
   - Fine-tuned on the dataset for binary classification.

## Steps

1. **Preprocessing**:
   - Combined daily headlines into single entries.
   - Cleaned text by removing punctuation, stopwords, and performing lemmatization.
2. **Feature Engineering**:
   - Used TF-IDF for Logistic Regression.
   - Tokenized and padded sequences for LSTM.
   - Encoded text using BERT's tokenizer.
3. **Model Training**:
   - Tuned hyperparameters for all models.
   - Used early stopping and validation AUC as metrics to avoid overfitting.
4. **Evaluation**:
   - Models were evaluated on test data using AUC (Area Under the Receiver Operating Characteristic Curve).
   - Classification reports and confusion matrices provided additional performance insights.

## Challenges

1. **Complexity of the Stock Market**:
   - The stock market is influenced by numerous factors beyond news headlines.
2. **Data Limitations**:
   - Limited dataset size and the indirect relationship between headlines and market movements.
3. **Model Limitations**:
   - Overfitting risks with deep learning models, especially on small datasets.

## Future Work

- Incorporate additional data sources, such as financial indicators and social media sentiment.
- Use advanced feature engineering techniques, like sentiment analysis and topic modeling.
- Experiment with ensemble models to combine the strengths of different approaches.

## How to Run the Project

### Prerequisites
- Python 3.8 or later
- Required libraries: `pandas`, `numpy`, `tensorflow`, `transformers`, `scikit-learn`, `keras-tuner`, `matplotlib`, `seaborn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/stock-market-prediction.git
   cd stock-market-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the Jupyter notebook:
   ```bash
   jupyter notebook stock_market_prediction.ipynb
   ```
2. Follow the cells in the notebook for step-by-step execution.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
