import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.text_preprocessing import load_and_split, make_vectorizer


#init the data (cleaned, duplicates removed, split into train/test)
#sentiments: Negative (1-2 stars), Neutral (3), Positive (4-5)
train, test = load_and_split('data/group_sample_seed49.csv')
print(train['sentiment'].value_counts())

#define vocabulary (training reviews only)
vocab = train['Text'].values
y = train['sentiment'].values

#define BOW (keeps "not", "no", "never" and adds two-word phrases like "not good")
bow_vectorizer = make_vectorizer('bow')
X_bow = bow_vectorizer.fit_transform(vocab)

#report BOW
print("BoW Vocabulary Size:", len(bow_vectorizer.vocabulary_))
print("BoW Matrix Shape:", X_bow.shape)

#define TF-IDF
tfidf_vectorizer = make_vectorizer('tfidf')
X_tfidf = tfidf_vectorizer.fit_transform(vocab)

#report TF-IDF
print("TF-IDF Vocabulary Size:", len(tfidf_vectorizer.vocabulary_))
print("TF-IDF Matrix Shape:", X_tfidf.shape)

#logistic regression model
lrm = LogisticRegression(max_iter=1000)

#get features from BOW model
#with 3 classes there is one row of coefficients per class
lrm.fit(X_bow, y)
feature_names = np.array(bow_vectorizer.get_feature_names_out())
positive_coefficients = lrm.coef_[list(lrm.classes_).index('Positive')]
negative_coefficients = lrm.coef_[list(lrm.classes_).index('Negative')]
vocab_words_bow = sorted(bow_vectorizer.vocabulary_)

#most positive features BOW
top_positive_indices = np.argsort(positive_coefficients)[-10:][::-1]
print("Top 10 Positive Features for BOW:")
for idx in top_positive_indices:
    print(f"  {feature_names[idx]}: {positive_coefficients[idx]:.4f}")

#most negative features BOW
top_negative_indices = np.argsort(negative_coefficients)[-10:][::-1]
print("\nTop 10 Negative Features for BOW:")
for idx in top_negative_indices:
    print(f"  {feature_names[idx]}: {negative_coefficients[idx]:.4f}")

#create a bar chart for top positive features in BOW
top_pos_words = feature_names[top_positive_indices]
top_pos_vals = positive_coefficients[top_positive_indices]
plt.figure(figsize=(10, 5))
sns.barplot(x=top_pos_vals, y=top_pos_words, palette='Blues_r')
plt.title('Top 10 Most Informative Features for Positive Sentiment in BOW')
plt.xlabel('Logistic Regression Coefficient')
plt.tight_layout()
plt.savefig('top_positive_features_bow.png')
plt.show()

#get features from tf-idf model
lrm.fit(X_tfidf, y)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
positive_coefficients = lrm.coef_[list(lrm.classes_).index('Positive')]
negative_coefficients = lrm.coef_[list(lrm.classes_).index('Negative')]
vocab_words_tfidf = sorted(tfidf_vectorizer.vocabulary_)

#most positive features TF-IDF
top_positive_indices = np.argsort(positive_coefficients)[-10:][::-1]
print("Top 10 Positive Features for TF-IDF:")
for idx in top_positive_indices:
    print(f"  {feature_names[idx]}: {positive_coefficients[idx]:.4f}")

#most negative features TF-IDF
top_negative_indices = np.argsort(negative_coefficients)[-10:][::-1]
print("\nTop 10 Negative Features for TF-IDF:")
for idx in top_negative_indices:
    print(f"  {feature_names[idx]}: {negative_coefficients[idx]:.4f}")

#create a bar chart for top positive features in TF-IDF
top_pos_words = feature_names[top_positive_indices]
top_pos_vals = positive_coefficients[top_positive_indices]
plt.figure(figsize=(10, 5))
sns.barplot(x=top_pos_vals, y=top_pos_words, palette='Blues_r')
plt.title('Top 10 Most Informative Features for Positive Sentiment in TF-IDF')
plt.xlabel('Logistic Regression Coefficient')
plt.tight_layout()
plt.savefig('top_positive_features_tfidf.png')
plt.show()

output_df = pd.DataFrame({
    'Vocab Words BOW': vocab_words_bow,
    'Vocab Words TF-IDF': vocab_words_tfidf
})
output_df.to_csv('data/processed_reviews.csv', index=False)
