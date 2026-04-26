import os
import re
import json
import logging
import traceback
import nltk
import io
import csv

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── NLTK data ─────────────────────────────────────────────────────────────────
def safe_nltk_download():
    # On cloud platforms, ensure NLTK has a writable data directory
    nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    for r in ["stopwords", "vader_lexicon", "punkt", "punkt_tab"]:
        try:
            nltk.download(r, quiet=True, download_dir=nltk_data_dir)
        except Exception as e:
            logger.warning(f"Could not download NLTK resource '{r}': {e}")

safe_nltk_download()

# ── Globals ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE, "templates")
MODEL_DIR = os.path.join(BASE, "models")

model = char_vectorizer = label_encoder = stemmer = sia = None
stop_words = set()

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
    'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn',
}

_POSITIVE_WORDS = {'good','great','excellent','amazing','wonderful','fantastic','love',
    'best','happy','joy','awesome','superb','perfect','nice','beautiful','pleasant',
    'brilliant','outstanding','positive','success','win','victory'}
_NEGATIVE_WORDS = {'bad','terrible','worst','awful','horrible','hate','sad','poor',
    'failure','loss','wrong','negative','useless','boring','dull','ugly','painful',
    'disaster','problem','issue','error','fault','broken','fail','crash'}

HINGLISH_MAP = {
    "accha":"good","acha":"good","achha":"good","acchi":"good",
    "bakwas":"bad","bekar":"bad","bura":"bad",
    "mast":"excellent","kamaal":"excellent",
    "pyaar":"love","mohabbat":"love",
    "mehnga":"expensive","sasta":"cheap",
    "faltu":"useless","ghatiya":"bad",
    "timepass":"boring","superb":"excellent",
    "bahut":"very","theek":"okay","bilkul":"absolutely",
    "yaar":"friend","nahi":"no","haan":"yes",
    "pasand":"like","behtareen":"best","khaas":"special",
    "zabardast":"amazing","shandar":"wonderful","besharam":"shameless",
    "pagal":"crazy","bewakoof":"stupid","shaitaan":"devil",
    "dil":"heart","jaan":"life","khushi":"happiness","dukh":"sadness",
    "sundar":"beautiful","zyada":"more",
    "kuch":"something","sab":"all","bas":"enough","phir":"then",
    "matlab":"meaning","samajh":"understand","dekh":"see","kar":"do",
}

# LABEL_MAP is derived from label_encoder.classes_ at runtime (see predict())
# Hardcoded fallback only used when label_encoder is unavailable
_FALLBACK_LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}

# ── File size limit: 200MB for 1M rows ───────────────────────────────────────
MAX_FILE_BYTES = 200 * 1024 * 1024   # 200 MB
MAX_ROWS       = 1_000_000
CHUNK_SIZE     = 5_000               # rows per processing chunk
MAX_RESULT_ROWS = 10_000             # return at most 10k results in response

# ── Model loader ──────────────────────────────────────────────────────────────
def load_models():
    global model, char_vectorizer, label_encoder, stemmer, stop_words, sia
    try:
        model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
        vectorizer_path = os.path.join(MODEL_DIR, "char_vectorizer.pkl")
        encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
        if not all(os.path.exists(path) for path in (model_path, vectorizer_path, encoder_path)):
            raise FileNotFoundError("One or more model files are missing")

        import joblib

        model           = joblib.load(model_path)
        char_vectorizer = joblib.load(vectorizer_path)
        label_encoder   = joblib.load(encoder_path)
        logger.info("✅ ML models loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}. Run train_model.py first.")
        model = char_vectorizer = label_encoder = None
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        model = char_vectorizer = label_encoder = None

    try:
        stemmer    = SnowballStemmer("english")
        stop_words = set(stopwords.words("english"))
        sia        = SentimentIntensityAnalyzer()
        logger.info("✅ NLP helpers initialised.")
    except Exception as e:
        logger.warning(f"NLP partial init: {e}")
        try: stemmer = SnowballStemmer("english")
        except: stemmer = None
        try: stop_words = set(stopwords.words("english"))
        except: stop_words = _FALLBACK_STOPWORDS
        try: sia = SentimentIntensityAnalyzer()
        except: sia = None

load_models()

# ── Text helpers ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    try:
        text = str(text).lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        words = []
        for w in text.split():
            if w in stop_words:
                continue
            mapped  = HINGLISH_MAP.get(w, w)
            stemmed = stemmer.stem(mapped) if stemmer else mapped
            words.append(stemmed)
        return " ".join(words) if words else text
    except Exception:
        return str(text)[:500]

def detect_hinglish(text: str):
    try:
        return [{"word": w, "meaning": HINGLISH_MAP[w]}
                for w in str(text).lower().split() if w in HINGLISH_MAP]
    except Exception:
        return []

def truncate_to_words(text: str, limit: int = 500):
    words     = str(text).split()
    truncated = len(words) > limit
    return " ".join(words[:limit]), truncated, len(words)

# ── Core prediction ───────────────────────────────────────────────────────────
def _default_scores(label: str) -> dict:
    return {
        "Negative": 1.0 if label == "Negative" else 0.0,
        "Neutral":  1.0 if label == "Neutral"  else 0.0,
        "Positive": 1.0 if label == "Positive" else 0.0,
    }

def _fallback_predict(text: str) -> dict:
    try:
        clean = clean_text(text)
        score = 0.0
        if sia:
            try: score = float(sia.polarity_scores(clean)["compound"])
            except: pass
        else:
            words     = set(clean.lower().split())
            pos_count = len(words & _POSITIVE_WORDS)
            neg_count = len(words & _NEGATIVE_WORDS)
            if   pos_count > neg_count: score =  0.2
            elif neg_count > pos_count: score = -0.2
        label = "Positive" if score >= 0.05 else "Negative" if score <= -0.05 else "Neutral"
        return {"sentiment": label, "scores": _default_scores(label),
                "vader": round(score, 3), "hinglish": detect_hinglish(text),
                "cleaned_text": clean, "fallback": True}
    except Exception:
        return {"sentiment": "Neutral", "scores": _default_scores("Neutral"),
                "vader": 0.0, "hinglish": [], "cleaned_text": "", "fallback": True}

def predict(text: str) -> dict:
    try:
        if not text or not str(text).strip():
            return {"error": "Empty text", "sentiment": "Neutral",
                    "scores": {"Negative":0.0,"Neutral":1.0,"Positive":0.0},
                    "vader": 0.0, "hinglish": [], "cleaned_text": ""}
        if model is None or char_vectorizer is None:
            return _fallback_predict(text)

        clean = clean_text(text) or "neutral"
        try:
            X_char = char_vectorizer.transform([clean])
        except Exception:
            return _fallback_predict(text)

        try:
            import numpy as np
            import scipy.sparse as sp
            from scipy.sparse import hstack, csr_matrix
        except Exception as e:
            logger.warning(f"ML dependencies unavailable: {e} - using fallback")
            return _fallback_predict(text)

        vader_compound = 0.0
        if sia:
            try: vader_compound = float(sia.polarity_scores(clean)["compound"])
            except: pass

        X_vec = hstack([X_char, csr_matrix(np.array([[vader_compound]]))])

        try:
            expected = int(model.n_features_in_)
            current  = X_vec.shape[1]
            if   current < expected: X_vec = hstack([X_vec, sp.csr_matrix((1, expected - current))])
            elif current > expected: X_vec = X_vec[:, :expected]
        except Exception:
            pass

        pred_num = int(model.predict(X_vec)[0])

        # Use label_encoder to map encoded integer → original label string
        try:
            label = str(label_encoder.inverse_transform([pred_num])[0])
            # Normalise numeric labels (0→Negative, 1→Neutral, 2→Positive)
            label = _FALLBACK_LABEL_MAP.get(int(label), label) if label.lstrip('-').isdigit() else label.capitalize()
        except Exception:
            label = _FALLBACK_LABEL_MAP.get(pred_num, "Neutral")

        try:
            proba  = model.predict_proba(X_vec)[0]
            # Build scores dict keyed by human-readable label using encoder class order
            classes = label_encoder.classes_
            raw_scores = {}
            for i, cls in enumerate(classes):
                if i < len(proba):
                    try:
                        human = _FALLBACK_LABEL_MAP.get(int(cls), str(cls)) if str(cls).lstrip('-').isdigit() else str(cls).capitalize()
                    except Exception:
                        human = str(cls).capitalize()
                    raw_scores[human] = float(round(proba[i], 3))
            # Ensure all three keys always present
            scores = {
                "Negative": raw_scores.get("Negative", 0.0),
                "Neutral":  raw_scores.get("Neutral",  0.0),
                "Positive": raw_scores.get("Positive", 0.0),
            }
            if sum(scores.values()) == 0.0:
                scores = _default_scores(label)
        except Exception:
            scores = _default_scores(label)

        return {"sentiment": label, "scores": scores,
                "vader": round(vader_compound, 3),
                "hinglish": detect_hinglish(text), "cleaned_text": clean}
    except Exception as e:
        logger.error(f"predict() failed: {e}\n{traceback.format_exc()}")
        return _fallback_predict(text)

def predict_batch_texts(texts):
    """Process a list of texts, return list of result dicts."""
    results = []
    for idx, t in enumerate(texts):
        try:
            text, was_truncated, wc = truncate_to_words(str(t), 500)
            r = predict(text)
            r["index"]         = idx
            r["word_count"]    = wc
            r["was_truncated"] = was_truncated
        except Exception as err:
            r = {"index": idx, "sentiment": "Neutral",
                 "scores": _default_scores("Neutral"),
                 "vader": 0.0, "hinglish": [], "cleaned_text": "", "error": str(err)}
        results.append(r)
    return results

def aggregate_stats(results):
    sentiments = [r.get("sentiment", "Neutral") for r in results]
    return {
        "total":    len(results),
        "positive": sentiments.count("Positive"),
        "neutral":  sentiments.count("Neutral"),
        "negative": sentiments.count("Negative"),
    }

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_BYTES + 10 * 1024 * 1024  # file + overhead

@app.errorhandler(400)
def bad_request(e): return jsonify({"error": "Bad request", "details": str(e)}), 400
@app.errorhandler(404)
def not_found(e):   return jsonify({"error": "Endpoint not found"}), 404
@app.errorhandler(413)
def too_large(e):   return jsonify({"error": f"Request too large (max {MAX_FILE_BYTES//1024//1024}MB)"}), 413
@app.errorhandler(500)
def server_error(e):return jsonify({"error": "Internal server error"}), 500
@app.errorhandler(Exception)
def unhandled(e):
    logger.error(f"Unhandled exception: {e}\n{traceback.format_exc()}")
    return jsonify({"error": "Unexpected error", "details": str(e)}), 500

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": model is not None, "nlp": sia is not None})

@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data     = request.get_json(silent=True, force=True) or {}
        raw_text = str(data.get("text", "")).strip()
        if not raw_text:
            return jsonify({"error": "No text provided"}), 400
        text, was_truncated, original_wc = truncate_to_words(raw_text, 500)
        result = predict(text)
        result["word_count"]      = original_wc
        result["was_truncated"]   = was_truncated
        result["words_processed"] = min(original_wc, 500)
        return jsonify(result)
    except Exception as e:
        logger.error(f"/predict error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/word-batch", methods=["POST"])
def word_batch_route():
    """
    Accepts:
      • JSON  { "text": "..." }            — splits into sentences, predicts each
      • JSON  { "texts": [...] }           — list mode up to MAX_ROWS
      • multipart file upload (CSV / TXT) — up to MAX_ROWS rows
    """
    try:
        # ── File upload path ──────────────────────────────────────────────────
        if "file" in request.files:
            f = request.files["file"]
            if not f or f.filename == "":
                return jsonify({"error": "No file selected"}), 400

            fname     = f.filename.lower()
            raw_bytes = f.read(MAX_FILE_BYTES)
            lines     = []

            if fname.endswith(".csv"):
                text_io = io.StringIO(raw_bytes.decode("utf-8", errors="ignore"))
                reader  = csv.reader(text_io)
                rows    = list(reader)
                if rows:
                    header  = [c.lower().strip() for c in rows[0]]
                    col_idx = next((i for i, h in enumerate(header) if "text" in h), 0)
                    for row in rows[1:MAX_ROWS + 1]:
                        if len(row) > col_idx and row[col_idx].strip():
                            lines.append(row[col_idx].strip())
            else:
                decoded = raw_bytes.decode("utf-8", errors="ignore")
                lines   = [l.strip() for l in decoded.splitlines() if l.strip()]
                lines   = lines[:MAX_ROWS]

            if not lines:
                return jsonify({"error": "No text found in file"}), 400

            total_rows = len(lines)
            logger.info(f"File upload: {total_rows} rows from {f.filename}")

            # Process in chunks
            results = []
            for chunk_start in range(0, min(total_rows, MAX_ROWS), CHUNK_SIZE):
                chunk   = lines[chunk_start:chunk_start + CHUNK_SIZE]
                results.extend(predict_batch_texts(chunk))

            # For very large results, trim what we return to browser
            returned = results[:MAX_RESULT_ROWS]
            sentiments = [r.get("sentiment","Neutral") for r in results]
            stats = {
                "total":    total_rows,
                "positive": sentiments.count("Positive"),
                "neutral":  sentiments.count("Neutral"),
                "negative": sentiments.count("Negative"),
                "returned": len(returned),
                "truncated_response": len(results) > MAX_RESULT_ROWS,
            }
            overall = max(["Positive","Neutral","Negative"], key=lambda l: sentiments.count(l))
            return jsonify({
                "results": returned, "stats": stats,
                "overall": overall, "filename": f.filename,
                "source": "file"
            })

        # ── JSON path ─────────────────────────────────────────────────────────
        data = request.get_json(silent=True, force=True) or {}

        # List of texts mode
        if "texts" in data:
            raw_texts = data["texts"]
            if isinstance(raw_texts, str):
                raw_texts = [t.strip() for t in raw_texts.splitlines() if t.strip()]
            if not isinstance(raw_texts, list):
                return jsonify({"error": "'texts' must be a list or newline-separated string"}), 400
            raw_texts = raw_texts[:MAX_ROWS]
            results   = predict_batch_texts(raw_texts)
            stats     = aggregate_stats(results)
            overall   = max(["Positive","Neutral","Negative"],
                            key=lambda l: [r.get("sentiment","Neutral") for r in results].count(l))
            return jsonify({"results": results, "stats": stats, "overall": overall, "source": "list"})

        # Single paragraph / sentence-split mode
        raw_text = str(data.get("text", "")).strip()
        if not raw_text:
            return jsonify({"error": "Provide 'text', 'texts', or upload a file"}), 400

        text, was_truncated, original_wc = truncate_to_words(raw_text, 500)
        sentences = re.split(r"(?<=[.!?।])\s+|(?<=\n)\s*", text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) >= 2]

        results = []
        for idx, sent in enumerate(sentences[:200]):
            try:
                r = predict(sent)
                r["index"] = idx
                r["sentence"] = sent
            except Exception as se:
                r = {"index": idx, "sentence": sent, "sentiment": "Neutral",
                     "scores": _default_scores("Neutral"), "vader": 0.0,
                     "hinglish": [], "error": str(se)}
            results.append(r)

        sentiments = [r.get("sentiment","Neutral") for r in results]
        stats = {
            "total":           len(results),
            "positive":        sentiments.count("Positive"),
            "neutral":         sentiments.count("Neutral"),
            "negative":        sentiments.count("Negative"),
            "original_words":  original_wc,
            "was_truncated":   was_truncated,
        }
        overall = max(["Positive","Neutral","Negative"], key=lambda l: sentiments.count(l))
        return jsonify({"results": results, "stats": stats, "overall": overall, "source": "text"})

    except Exception as e:
        logger.error(f"/word-batch error: {e}\n{traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"
    print(f"\n  VaakBhav — Hinglish Sentiment Analyzer")
    print(f"  http://localhost:{port}\n")
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=debug)