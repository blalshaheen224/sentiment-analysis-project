# 🧠 Sentiment Analysis on Amazon Product Reviews

This project performs sentiment classification on Amazon product reviews using machine learning techniques. It classifies reviews into three categories:

- ✅ Positive
- 😐 Neutral
- ❌ Negative

We use text preprocessing, TF-IDF vectorization, and a Logistic Regression model to predict the sentiment of a review.

---

## 📁 Dataset

We use the publicly available dataset `Reviews.csv`, which contains the following fields:

- `Score`: The rating from 1 to 5
- `Summary`: A short summary of the review
- `Text`: The full text of the review

### 📄 Citation

This is a requirement from the data owner. If you publish articles based on this dataset, please cite the following paper:

> **J. McAuley and J. Leskovec**. *From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews*. WWW, 2013.\
> [View Paper on ACM Digital Library](https://dl.acm.org/doi/10.1145/2488388.2488466)\
> [Dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data)

---

## 📊 Project Workflow

### 1️⃣ Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

We import libraries for data manipulation, visualization, preprocessing, modeling, and evaluation.

---

### 2️⃣ Load the Dataset

```python
df = pd.read_csv('/content/drive/MyDrive/content/Reviews.csv')
```

We load the dataset into a DataFrame.

---

### 3️⃣ Combine Review Fields

```python
df['review'] = df['Summary'].fillna('') + " " + df['Text'].fillna('')
```

We merge the `Summary` and `Text` fields into a single field called `review`. If any values are missing, we fill them with an empty string.

---

### 4️⃣ Map Score to Sentiment

```python
def map_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['Score'].apply(map_sentiment)
```

We convert the numeric scores into sentiment labels:

- 1 or 2 → `negative`
- 3 → `neutral`
- 4 or 5 → `positive`

---

### 5️⃣ Clean the Text

```python
def clean_text(text):
    text = re.sub(r"http\S+|\W+|\d+", ' ', text.lower())
    return text.strip()

df['clean_review'] = df['review'].apply(clean_text)
```

We clean the review text by removing:

- URLs
- Punctuation and non-word characters
- Numbers
- Uppercase letters (convert to lowercase)

---

### 6️⃣ Balance the Classes (Upsampling)

```python
df_majority = df[df['sentiment'] == 'positive']
df_minority_neutral = df[df['sentiment'] == 'neutral']
df_minority_negative = df[df['sentiment'] == 'negative']

neutral_upsampled = resample(df_minority_neutral, replace=True, n_samples=len(df_majority), random_state=42)
negative_upsampled = resample(df_minority_negative, replace=True, n_samples=len(df_majority), random_state=42)

df_balanced = pd.concat([df_majority, neutral_upsampled, negative_upsampled])
```

We fix class imbalance by upsampling the minority classes (`neutral`, `negative`) to match the size of the majority class (`positive`).

---

### 7️⃣ Train-Test Split and TF-IDF Vectorization

```python
X = df_balanced['clean_review']
Y = df_balanced['sentiment']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

vectorizer = TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1, 2))
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
```

We:

- Split the dataset into training and testing sets
- Use TF-IDF vectorization to convert text into numerical features
- Use bigrams and remove English stopwords

---

### 8️⃣ Model Training (Logistic Regression)

```python
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(x_train_vec, y_train)
```

We train a **Logistic Regression** model with:

- `max_iter=1000` to allow more iterations for convergence
- `class_weight='balanced'` to handle any slight imbalance

---

### 9️⃣ Evaluation

```python
y_pred = model.predict(x_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

We print:

- Overall accuracy
- Precision, recall, and F1-score
- Confusion matrix

#### 📈 Model Performance:

```
🙌 Accuracy: 0.9684

📊 Classification Report:
               precision    recall  f1-score   support

    negative       0.97      0.97      0.97     88755
     neutral       0.95      0.99      0.97     88756
    positive       0.98      0.94      0.96     88756

    accuracy                           0.97    266267
   macro avg       0.97      0.97      0.97    266267
weighted avg       0.97      0.97      0.97    266267

🧹 Confusion Matrix:
 [[86489  1144  1122]
  [  348 87913   495]
  [ 2205  3100 83451]]
```
---
note He might answer wrongly.

### 🔒 10️⃣ Save Model and Vectorizer

```python
import joblib
joblib.dump(model, 'sentiment_model_complete_text_summary_logistc.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer_complete_text_summary_logistc.pkl')
```

We save the model and the TF-IDF vectorizer so we can reuse them later without retraining.

---

### 🧪 11️⃣ Interactive Testing

```python
while True:
    test_text = input("Enter a review (type 'stop' to quit): ")
    if test_text.lower() == "stop":
        break
    test_text_clean = clean_text(test_text)
    test_vec = vectorizer.transform([test_text_clean])
    prediction = model.predict(test_vec)[0]
    print(f"Predicted sentiment: {prediction}")
```

This loop lets the user enter a review, which will be cleaned, vectorized, and passed to the trained model to predict the sentiment live.

---

## 🛠️ Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib
- seaborn
- matplotlib
- numpy

You can install all required packages via:

```bash
pip install -r requirements.txt
```

---

## 📌 How to Run

1. Clone the repository
2. Place the `Reviews.csv` in the correct path
3. Run the script in Jupyter or Google Colab
4. Interact with the model via the command line

---

## 💡 Future Improvements

- Add deep learning models (LSTM, BERT)
- Create a Streamlit web app
- Perform real-time scraping and analysis

---

## 🧑‍💻 Author

Built with ❤️ by [Your Name]

