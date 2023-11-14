import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the training data
train_path = "D:/Internships/codesoft/task1/Genre Classification Dataset/train_data.txt"
train_data = pd.read_csv(train_path, sep=':::', names=['Title', 'Genre', 'Description'], engine='python')

print("Training data description:")
print(train_data.describe())

print("\nTraining data information:")
print(train_data.info())

print("\nNumber of missing values in the training data:")
print(train_data.isnull().sum())

# Load the test data
test_path = "D:/Internships/codesoft/task1/Genre Classification Dataset/test_data.txt"
test_data = pd.read_csv(test_path, sep=':::', names=['Id', 'Title', 'Description'], engine='python')
print("\nTest data:")
print(test_data.head())

# Plot the distribution of genres in the training data
plt.figure(figsize=(12, 6))
sns.countplot(data=train_data, y='Genre', order=train_data['Genre'].value_counts().index, palette='Set2')
plt.xlabel('Count', fontsize=12, fontweight='bold')
plt.ylabel('Genre', fontsize=12, fontweight='bold')
plt.title('Distribution of Genres in Training Data', fontsize=14, fontweight='bold')
plt.show()

# Initialize the stemmer and stop words
stemmer = LancasterStemmer()
stop_words = set(stopwords.words('english'))

# Define the clean_text function
def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

# Apply the clean_text function to the 'Description' column in the training and test data
train_data['Cleaned_Description'] = train_data['Description'].apply(clean_text)
test_data['Cleaned_Description'] = test_data['Description'].apply(clean_text)

# Calculate the length of cleaned text
train_data['Description_Length'] = train_data['Cleaned_Description'].apply(len)

# Visualize the distribution of text lengths
plt.figure(figsize=(8, 6))
sns.histplot(data=train_data, x='Description_Length', bins=20, kde=True, color='purple')
plt.xlabel('Length', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Distribution of Description Lengths', fontsize=14, fontweight='bold')
plt.show()

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train = tfidf_vectorizer.fit_transform(train_data['Cleaned_Description'])

# Transform the test data
X_test = tfidf_vectorizer.transform(test_data['Cleaned_Description'])

# Split the data into training and validation sets
X = X_train
y = train_data['Genre']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = classifier.predict(X_val)

# Evaluate the performance of the model
accuracy = accuracy_score(y_val, y_pred)
print("\nValidation Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Use the trained model to make predictions on the test data
y_test_pred = classifier.predict(X_test)

# Save thepredictions to a CSV file
output_path = "predictions.csv"
test_data['Predicted_Genre'] = y_test_pred
test_data[['Id', 'Predicted_Genre']].to_csv(output_path, index=False)

print("\nPredictions saved to:", output_path)