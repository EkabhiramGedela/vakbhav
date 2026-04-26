#!/usr/bin/env bash
# ── VaakBhav build script ──────────────────────────────────────────────
# Runs once during deployment to install deps and train the ML model.
# Called by Railway / Render / Heroku as the build command:
#   pip install -r requirements.txt && bash build.sh
# ──────────────────────────────────────────────────────────────────────

set -e   # abort on any error

echo "=== [1/3] Downloading NLTK data ==="
python - <<'PYEOF'
import nltk
for r in ["stopwords", "vader_lexicon", "punkt", "punkt_tab"]:
    try:
        nltk.download(r, quiet=False)
        print(f"  ✓ {r}")
    except Exception as e:
        print(f"  ⚠ {r}: {e}")
PYEOF

echo ""
echo "=== [2/3] Checking model files ==="
if [ -f "models/sentiment_model.pkl" ] && \
   [ -f "models/char_vectorizer.pkl" ] && \
   [ -f "models/label_encoder.pkl" ]; then
    echo "  ✓ All model .pkl files already present — skipping training"
else
    echo "  ⚙ Model files missing — running train_model.py ..."
    if [ -f "output__1_.csv" ]; then
        python train_model.py
        echo "  ✓ Training complete"
    else
        echo "  ⚠ WARNING: output__1_.csv not found."
        echo "    App will start in VADER-fallback mode (no ML model)."
        echo "    Add output__1_.csv to repo root and redeploy to enable full ML."
        mkdir -p models   # ensure folder exists so app.py doesn't crash
    fi
fi

echo ""
echo "=== [3/3] Build finished ==="
