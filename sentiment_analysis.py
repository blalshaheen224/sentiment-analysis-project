# 📚 1. استيراد المكتبات
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
import joblib

# 📥 2. تحميل البيانات
df = pd.read_csv('/content/drive/MyDrive/content/Reviews.csv')

# 🧹 3. التنظيف الأولي
df['review'] = df['Summary'].fillna('') + " " + df['Text'].fillna('')

# 🏷️ 4. تصنيف التقييمات إلى فئات ثلاثية
def map_sentiment(score):
    if score in [1, 2]:
        return 'negative'
    elif score == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['Score'].apply(map_sentiment)

# 🧼 5. تنظيف النصوص
def clean_text(text):
    text = re.sub(r"http\S+|\W+|\d+", ' ', text.lower())
    return text.strip()

df['clean_review'] = df['review'].apply(clean_text)

# ⚖️ 6. معالجة عدم توازن البيانات (Upsampling)
df_majority = df[df['sentiment'] == 'positive']
df_minority_neutral = df[df['sentiment'] == 'neutral']
df_minority_negative = df[df['sentiment'] == 'negative']

neutral_upsampled = resample(df_minority_neutral,
                             replace=True,     # عينة مع التكرار
                             n_samples=len(df_majority),
                             random_state=42)

negative_upsampled = resample(df_minority_negative,
                              replace=True,
                              n_samples=len(df_majority),
                              random_state=42)

df_balanced = pd.concat([df_majority, neutral_upsampled, negative_upsampled])

# ✅ 7. التجهيز للنمذجة
X = df_balanced['clean_review']
Y = df_balanced['sentiment']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

vectorizer = TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1, 2))
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

from sklearn.svm import LinearSVC


model = LogisticRegression(max_iter=1000,class_weight='balanced')
model.fit(x_train_vec, y_train)

# 🎯 9. التقييم
y_pred = model.predict(x_test_vec)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# Save the trained model
joblib.dump(model, 'sentiment_model_complete_text_summary_logistc.pkl')

# Save the trained vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer_complete_text_summary_logistc.pkl')


while True:
    test_text = input("\n📝 اكتب مراجعة لاختبارها (أو stop للخروج): ")
    if test_text.lower() == "stop":
        print("🛑 تم إنهاء الاختبار.")
        break
    test_text_clean = clean_text(test_text)
    test_vec = vectorizer.transform([test_text_clean])
    prediction = model.predict(test_vec)[0]
    print(f"\n✅ التقييم المتوقع: {prediction}")


