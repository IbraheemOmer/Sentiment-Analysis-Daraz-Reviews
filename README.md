# Sentiment Analysis On Daraz Reviews

This repository contains the code and data for the Sentiment Analysis of reviews from Daraz.pk, a prominent e-commerce site. The primary goal is to develop a model that accurately classifies the sentiment expressed in customer reviews.

## Contents

- `Daraz Reviews.csv`: Labeled data of the reviews scraped and labeled, used for training the model.
- `Sentiment-Analysis-Daraz-Reviews.ipynb`: The Jupyter notebook containing the code for the sentiment analysis.

## Project Workflow

### Data Collection

Reviews were collected from Daraz.pk by a group of 18 students, each responsible for gathering 500 reviews related to specific product categories. The reviews were then labeled as positive, negative, or neutral.

### Data Preprocessing

The text data underwent several preprocessing steps across multiple iterations to improve the model's performance:

1. **First Iteration**: Initial preprocessing included converting text to lowercase, tokenizing, stemming using the LancasterStemmer, and removing stop words. CountVectorizer and LabelEncoder were used for feature extraction and label encoding respectively.

2. **Second Iteration**: Improved preprocessing with adjustments such as using the PorterStemmer, removing punctuation, reducing repeated letters, and removing emojis. TfidfVectorizer was used for feature extraction.

3. **Third Iteration**: Final preprocessing involved converting text to lowercase, using TfidfVectorizer for feature extraction, and employing LabelEncoder for sentiment label encoding.

### Model Development

Several classifiers were implemented and tested, including:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Gradient Boosting
- XGBoost
- AdaBoost
- Logistic Regression
- Support Vector Machine (SVM)
- Multinomial Naive Bayes

Among these, the SVM classifier performed the best, achieving the highest weighted F1 score. Various neural network architectures were also tested but did not outperform the SVM classifier.

### Testing and Results

The dataset was split into training and testing sets, with the SVM classifier achieving the best results. The model's predictions on unseen data provided by the instructor resulted in a weighted F1 score of 86%, indicating reliable performance.

### Limitations and Future Improvements

1. **Imbalanced Dataset**: The dataset had an overrepresentation of positive sentiments.
2. **Lack of Human Validation**: Sentiment labels were not double-checked by external sources.
3. **Roman Urdu Data**: The presence of Roman Urdu posed challenges for sentiment analysis.
4. **Computational Limitations**: Limited resources restricted the use of more complex models.

Future improvements include enhancing the dataset, incorporating human validation, supporting multiple languages, and optimizing models for efficiency.

## Conclusion

This project successfully developed a sentiment analysis model for Daraz.pk reviews using various preprocessing techniques and classifiers, with the SVM classifier emerging as the best performer. Future improvements aim to address the identified limitations to further enhance the model's accuracy and reliability.

---

For more information about Daraz, visit [Daraz.pk](https://www.daraz.pk/).
