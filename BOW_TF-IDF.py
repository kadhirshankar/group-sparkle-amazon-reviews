import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#init the data
df = pd.read_csv('data/group_sample_seed49.csv')
df = df[df['Score'] != 3]

#sentiments, 1 for 4-5, 0 for 1-2, ignore neutral reviews
df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
print(df['sentiment'].value_counts())

#define vocabulary
vocab = df['Text'].values

#define BOW
bow_vectorizer = CountVectorizer(max_features=10000, stop_words='english')
X_bow = bow_vectorizer.fit_transform(vocab)

#report BOW
print("BoW Vocabulary Size:", len(bow_vectorizer.vocabulary_))
print("BoW Matrix Shape:", X_bow.shape)

#define TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(vocab)

#report TF-IDF
print("TF-IDF Vocabulary Size:", len(tfidf_vectorizer.vocabulary_))
print("TF-IDF Matrix Shape:", X_tfidf.shape)

#logistic regression model
y = df['sentiment'].values
lrm = LogisticRegression(max_iter=1000)

#get features from BOW model
lrm.fit(X_bow, y)
feature_names = np.array(bow_vectorizer.get_feature_names_out())
coefficients = lrm.coef_[0]
vocab_words_bow = list(bow_vectorizer.vocabulary_.keys())

#most positive features BOW
top_positive_indices = np.argsort(coefficients)[-10:][::-1]
print("Top 10 Positive Features for BOW:")
for idx in top_positive_indices:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

#most negative features BOW
top_negative_indices = np.argsort(coefficients)[:10]
print("\nTop 10 Negative Features for BOW:")
for idx in top_negative_indices:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

#create a bar chart for top positive features in BOW
top_pos_words = feature_names[top_positive_indices]
top_pos_vals = coefficients[top_positive_indices]
plt.figure(figsize=(10, 5))
sns.barplot(x=top_pos_vals, y=top_pos_words, palette='Blues_r')
plt.title('Top 10 Most Informative Features for Positive Sentiment in BOW')
plt.xlabel('Logistic Regression Coefficient')
plt.tight_layout()
plt.savefig('top_positive_features.png')
plt.show()

#get features from tf-idf model
lrm.fit(X_tfidf, y)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
coefficients = lrm.coef_[0]
vocab_words_tfidf = list(tfidf_vectorizer.vocabulary_.keys())

#most positive features TF-IDF
top_positive_indices = np.argsort(coefficients)[-10:][::-1]
print("Top 10 Positive Features for TF-IDF:")
for idx in top_positive_indices:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

#most negative features TF-IDF
top_negative_indices = np.argsort(coefficients)[:10]
print("\nTop 10 Negative Features for TF-IDF:")
for idx in top_negative_indices:
    print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")

#create a bar chart for top positive features in TF-IDF
top_pos_words = feature_names[top_positive_indices]
top_pos_vals = coefficients[top_positive_indices]
plt.figure(figsize=(10, 5))
sns.barplot(x=top_pos_vals, y=top_pos_words, palette='Blues_r')
plt.title('Top 10 Most Informative Features for Positive Sentiment in TF-IDF')
plt.xlabel('Logistic Regression Coefficient')
plt.tight_layout()
plt.savefig('top_positive_features.png')
plt.show()

output_df = pd.DataFrame({
    'Vocab Words BOW': vocab_words_bow,
    'Vocab Words TF-IDF': vocab_words_tfidf
})
output_df.to_csv('data/processed_reviews.csv', index=False)