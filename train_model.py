"""
VaakBhav — train_model.py
Run this once to produce the three .pkl files in ./models/
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack, csr_matrix

# ── Download NLTK data ────────────────────────────────────────────────────────
for resource in ["stopwords", "vader_lexicon"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass  # will fall back below

# Hardcoded English stopwords as offline fallback
_FALLBACK_STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than','too',
    'very','s','t','can','will','just','don','should','now','d','ll','m','o',
    're','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
    'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won',
    'wouldn',
}

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "output__1_.csv"       # update if your CSV has a different name
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load & validate data ──────────────────────────────────────────────────────
print("📂 Loading dataset…")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    # Try alternate filename
    try:
        df = pd.read_csv("output (1).csv")
    except FileNotFoundError:
        print(f"❌ CSV not found. Place '{CSV_PATH}' in this folder and retry.")
        sys.exit(1)

print(f"   Raw rows: {len(df)}")

# Normalise column names
df.columns = [c.strip().lower() for c in df.columns]

# Detect text + label columns flexibly
text_col  = next((c for c in df.columns if 'text' in c), None)
label_col = next((c for c in df.columns if 'label' in c or 'sentiment' in c), None)

if text_col is None or label_col is None:
    print(f"❌ Could not find text/label columns. Found: {list(df.columns)}")
    sys.exit(1)

df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
print(f"   After dropna: {len(df)}")
print(f"   Label distribution:\n{df['label'].value_counts()}")

# ── NLP setup ─────────────────────────────────────────────────────────────────
HINGLISH_MAP = {
    "accha": "good", "acha": "good", "achha": "good", "acchi": "good",
    "bakwas": "bad", "bekar": "bad", "bura": "bad",
    "mast": "excellent", "kamaal": "excellent",
    "pyaar": "love", "mohabbat": "love",
    "mehnga": "expensive", "sasta": "cheap",
    "faltu": "useless", "ghatiya": "bad",
    "timepass": "boring", "superb": "excellent",
    "bahut": "very", "theek": "okay", "bilkul": "absolutely",
}

stemmer = SnowballStemmer("english")
try:
    stop_words = set(stopwords.words("english"))
except Exception:
    stop_words = _FALLBACK_STOPWORDS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [
        stemmer.stem(HINGLISH_MAP.get(w, w))
        for w in text.split()
        if w not in stop_words
    ]
    return " ".join(words) if words else text

print("\n🔧 Cleaning text…")
df["clean_text"] = df["text"].apply(clean_text)

# ── Label encoding ─────────────────────────────────────────────────────────────
le = LabelEncoder()
# Coerce labels to int if they are numeric strings
try:
    df["label_encoded"] = le.fit_transform(df["label"].astype(int))
except (ValueError, TypeError):
    df["label_encoded"] = le.fit_transform(df["label"])

label_dict = dict(zip(le.transform(le.classes_), le.classes_))
print(f"   Classes: {label_dict}")

# ── VADER scores ──────────────────────────────────────────────────────────────
print("\n🔍 Computing VADER scores…")
try:
    sia = SentimentIntensityAnalyzer()
    df["vader_score"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])
except Exception:
    print("   ⚠ VADER unavailable — using 0.0 as fallback")
    sia = None
    df["vader_score"] = 0.0

# ── Noise filtering ───────────────────────────────────────────────────────────
# Only drop clear mismatches
noisy_idx = df[
    ((df["vader_score"] > 0.8) & (df["label_encoded"] == 0)) |
    ((df["vader_score"] < -0.8) & (df["label_encoded"] == 2))
].index
df = df.drop(noisy_idx).reset_index(drop=True)
print(f"   After noise filtering: {len(df)}")

# ── Light augmentation ────────────────────────────────────────────────────────
augmentation_map = {
    "good": ["nice", "great", "awesome"],
    "bad":  ["worst", "poor", "terrible"],
    "love": ["like", "enjoy"],
    "bor":  ["dull", "slow"],         # stemmed 'boring'
    "excel":["fantastic", "amazing"], # stemmed 'excellent'
}

def augment_text(text):
    words = text.split()
    out = []
    for w in words:
        if w in augmentation_map and np.random.rand() < 0.25:
            out.append(np.random.choice(augmentation_map[w]))
        else:
            out.append(w)
    return " ".join(out)

aug_texts  = [augment_text(t) for t in df["clean_text"]]
aug_labels = df["label_encoded"].tolist()
aug_df     = pd.DataFrame({"clean_text": aug_texts, "label_encoded": aug_labels})
df         = pd.concat([df[["clean_text", "label_encoded"]], aug_df], ignore_index=True)
print(f"📈 Dataset after augmentation: {len(df)}")

# ── Train / test split ────────────────────────────────────────────────────────
X = df["clean_text"]
y = df["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ── Vectorisers ───────────────────────────────────────────────────────────────
print("\n📐 Building feature matrix…")
char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=9000)

X_train_char = char_vectorizer.fit_transform(X_train)
X_test_char  = char_vectorizer.transform(X_test)

# VADER feature
def get_vader(text):
    if sia is None:
        return 0.0
    try:
        return sia.polarity_scores(text)["compound"]
    except Exception:
        return 0.0

vader_train = np.array([get_vader(t) for t in X_train]).reshape(-1, 1)
vader_test  = np.array([get_vader(t) for t in X_test]).reshape(-1, 1)

X_train_vec = hstack([X_train_char, csr_matrix(vader_train)])
X_test_vec  = hstack([X_test_char,  csr_matrix(vader_test)])

print(f"   Feature shape: {X_train_vec.shape}")

# ── Model ─────────────────────────────────────────────────────────────────────
print("\n🏋️  Training VotingClassifier…")
svm = CalibratedClassifierCV(
    LinearSVC(class_weight="balanced", max_iter=3000), method="sigmoid"
)
lr  = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.5)

hybrid_model = VotingClassifier([("svm", svm), ("lr", lr)], voting="soft", n_jobs=-1)
hybrid_model.fit(X_train_vec, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred   = hybrid_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred) * 100
f1       = f1_score(y_test, y_pred, average="weighted") * 100

print(f"\n✅ Accuracy:          {accuracy:.2f}%")
print(f"✅ Weighted F1-score: {f1:.2f}%")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=[str(label_dict.get(i,i)) for i in sorted(label_dict)]))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(hybrid_model,   os.path.join(MODEL_DIR, "sentiment_model.pkl"))
joblib.dump(char_vectorizer,os.path.join(MODEL_DIR, "char_vectorizer.pkl"))
joblib.dump(le,             os.path.join(MODEL_DIR, "label_encoder.pkl"))

print(f"\n🎯 Models saved to ./{MODEL_DIR}/")
print("   → sentiment_model.pkl")
print("   → char_vectorizer.pkl")
print("   → label_encoder.pkl")
print("\nNow run:  python app.py")
