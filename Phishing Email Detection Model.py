import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def extract_custom_features(texts):
    """Extract and analyze email features such as URLs, keywords, and text length."""
    features = []
    for text in texts:
        # Feature 1: Count URLs present in the email text
        url_count = len(re.findall(r'http[s]?://|www\.', text))
        # Feature 2: Check for suspicious keywords/triggers
        has_urgent = 1 if re.search(r'\b(urgent|verify|suspend|click|login|immediate)\b', text, re.IGNORECASE) else 0
        # Feature 3: Check for financial/lottery keywords
        has_money = 1 if '$' in text or 'win' in text.lower() else 0
        # Feature 4: Length of text
        text_length = len(text)
        
        features.append([url_count, has_urgent, has_money, text_length])
        
    return np.array(features)

def run_phishing_detection_pipeline():
    print("[*] Initializing Phishing Email Detection Model...")

    # 1. Train on a dataset of phishing and legitimate emails[cite: 5]
    data = {
        "text": [
            "Dear user, your bank account has been locked. Click the secure link http://phishing-site.com immediately to verify your credentials.",
            "Hey Yash, are we still meeting for the project sync tomorrow at 10 AM in the conference room?",
            "URGENT: You have won a lottery of $1,000,000! Claim your prize now by sending your details to http://win-prize.net",
            "Hi team, please find attached the quarterly financial status report for your review.",
            "Security Alert: Someone tried to log into your account from an unrecognized device. Reset your password here: http://secure-login.com",
            "Don't forget to buy groceries on your way back home today: milk, bread, and coffee.",
            "Verify your Apple ID immediately at http://apple-id-verify.com to prevent permanent account deactivation.",
            "Your subscription invoice for Netflix has been successfully processed. Thank you for staying with us."
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Phishing, 0 = Safe (Legitimate)[cite: 5]
    }
    
    df = pd.DataFrame(data)

    # Split dataset into training and testing subsets
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42
    )

    # 2. Extract and analyze email features (URLs, keywords, etc.)[cite: 5]
    print("[*] Extracting and analyzing TF-IDF text tokens and URL/keyword features...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    X_train_custom = extract_custom_features(X_train_text)
    X_test_custom = extract_custom_features(X_test_text)

    # Combine text vectors with custom numerical features
    X_train_final = hstack([X_train_tfidf, X_train_custom])
    X_test_final = hstack([X_test_tfidf, X_test_custom])

    # Train model using Random Forest Classifier
    print("[*] Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_final, y_train)

    # 3. Classify emails as "Phishing" or "Safe"[cite: 5]
    print("[*] Classifying testing emails...")
    y_pred = model.predict(X_test_final)

    # 4. Display accuracy and confusion matrix[cite: 5]
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[+] Model Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"], zero_division=0))

    print("[*] Generating Confusion Matrix visualization plot...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Phishing"])
    
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.show()

    # Live classification test
    print("\n--- Live Sample Email Classification ---")
    test_samples = [
        "Urgent action required: update your password via http://malicious-link.com to secure your profile.",
        "Can we reschedule our project sync call to Thursday afternoon?"
    ]
    
    sample_tfidf = vectorizer.transform(test_samples)
    sample_custom = extract_custom_features(test_samples)
    sample_final = hstack([sample_tfidf, sample_custom])
    
    predictions = model.predict(sample_final)
    for email, pred in zip(test_samples, predictions):
        status = "Phishing 🚨" if pred == 1 else "Safe ✅"
        print(f"Email: '{email[:45]}...' --> Classification: {status}")

if __name__ == "__main__":
    run_phishing_detection_pipeline()
    