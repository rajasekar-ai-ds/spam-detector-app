import streamlit as st
import joblib
import json
import re
from pathlib import Path
import requests
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_FILE = BASE / "spam_clf.joblib"
VECT_FILE = BASE / "tfidf_vectorizer.joblib"
ARTIFACTS_FILE = BASE / "training_artifacts.json"
TRUSTED_SENDERS = {
    "airtel", "jio", "vi", "vodafone", "bankicici", "bankhdfc",
    "amazon", "flipkart", "phonepe", "paytm", "google", "youtube" , "sbi" , "hdfc" , "github", "twitter",
    "apple" , "x" , "gpay" , "googlepay" , "ubi", "icici", "microsoft", "jetbrains", "whatsapp", "bsnl"    
}
SUSPICIOUS_KEYWORDS = [
    "win", "won", "claim", "lottery", "prize", "click here", "card details", "$10000" , "click here to claim" ,
    "share your card", "free", "congrats", "reply yes", "you have won", "claim now",
    "click to claim", "claim prize", "urgent", "limited time", "act now" , "credited", "rupees" , "debited" , "won an", "iphone" , "enter the card details"
]

CARD_REQUEST_PATTERNS = ["card details", "share your card", "enter your card", "cvv", "account number", "bank details"]

TRUSTED_THRESHOLD = 0.50
UNKNOWN_THRESHOLD = 0.35

FORCE_SPAM_ON_KW_AND_CARD = True   
FORCE_SPAM_ON_KW_AND_URL = True     
FORCE_SPAM_ON_MULTIPLE_KW = True   
MULTI_KW_COUNT = 2                  

def normalize_text(s: str) -> str:
    """Same normalization used during training."""
    if s is None:
        return ""
    s = str(s)
    s = s.lower()   
    s = re.sub(r'\b\d{6,}\b', ' <PHONE> ', s)
    s = re.sub(r'\b\d+\b', ' <NUM> ', s)   
    s = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', s)   
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def simple_sender_normalize(s: str) -> str:
    if not s:
        return ""
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def is_sender_trusted(sender: str) -> bool:
    s = simple_sender_normalize(sender)
    return s in TRUSTED_SENDERS

def contains_suspicious_keyword(text: str):
    t = text.lower()
    found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in t]
    return len(found) > 0, found

def contains_card_request(text: str):
    t = text.lower()
    found = [p for p in CARD_REQUEST_PATTERNS if p in t]
    return len(found) > 0, found

def contains_url(text: str):
    return bool(re.search(r'https?://|www\.', text, flags=re.I))

def count_suspicious_keywords(text: str):
    t = text.lower()
    return sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in t)


model = None
vectorizer = None
threshold_from_artifacts = None
missing = []

if not MODEL_FILE.exists():
    missing.append(str(MODEL_FILE))
if not VECT_FILE.exists():
    missing.append(str(VECT_FILE))
if not ARTIFACTS_FILE.exists():
    missing.append(str(ARTIFACTS_FILE))

if not missing:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
    try:
        with open(ARTIFACTS_FILE, "r") as f:
            artifacts = json.load(f)
            threshold_from_artifacts = float(artifacts.get("threshold", TRUSTED_THRESHOLD))
    except Exception:
        threshold_from_artifacts = TRUSTED_THRESHOLD
else:
    st.warning("Model artifacts missing: " + ", ".join(missing))
    st.info("Place spam_clf.joblib, tfidf_vectorizer.joblib and training_artifacts.json in the app folder.")

def get_spam_probability(text_norm: str):
    """Return spam probability from model (if available)."""
    if model is None or vectorizer is None:
        return None
    X = vectorizer.transform([text_norm])
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X)[0, 1])
    else:
        try:
            score = float(model.decision_function(X)[0])
            return 1.0 / (1.0 + (2.718281828459045 ** (-score)))
        except Exception:
            return None

def decide_label_with_rules(raw_text: str, sender: str = ""):
    """
    Return dict:
      - label (0 ham, 1 spam)
      - prob (model probability or None)
      - debug (which rules fired, used threshold, etc.)
    """
    norm = normalize_text(raw_text)
    prob = get_spam_probability(norm)
    trusted = is_sender_trusted(sender)
    suspicious_flag, found_keywords = contains_suspicious_keyword(raw_text)
    card_flag, found_card = contains_card_request(raw_text)
    url_flag = contains_url(raw_text)
    kw_count = count_suspicious_keywords(raw_text)

    debug = {
        "trusted_sender": trusted,
        "suspicious_keywords": found_keywords,
        "card_request_phrases": found_card,
        "url_present": url_flag,
        "kw_count": kw_count,
        "model_prob": prob
    }

    used_threshold = threshold_from_artifacts if threshold_from_artifacts is not None else TRUSTED_THRESHOLD
    if trusted:
        used_threshold = max(used_threshold, TRUSTED_THRESHOLD)
    else:
        used_threshold = UNKNOWN_THRESHOLD
    forced_spam = False
    forced_reasons = []

    if not trusted:
        if FORCE_SPAM_ON_KW_AND_CARD and suspicious_flag and card_flag:
            forced_spam = True
            forced_reasons.append("kw_and_card")
        if FORCE_SPAM_ON_KW_AND_URL and suspicious_flag and url_flag:
            forced_spam = True
            forced_reasons.append("kw_and_url")
        if FORCE_SPAM_ON_MULTIPLE_KW and kw_count >= MULTI_KW_COUNT:
            forced_spam = True
            forced_reasons.append("multiple_kw")

        if suspicious_flag and (prob is not None and prob < 0.20):
            forced_spam = True
            forced_reasons.append("suspicious_low_model_prob")
    if forced_spam:
        label = 1
    else:
        if prob is None:
            try:
                X = vectorizer.transform([norm]) if vectorizer is not None else None
                label = int(model.predict(X)[0]) if model is not None else 0
            except Exception:
                label = 0
        else:
            label = 1 if prob >= used_threshold else 0

    debug["used_threshold"] = used_threshold
    debug["forced_spam"] = forced_spam
    debug["forced_reasons"] = forced_reasons
    debug["final_label"] = label
    debug["norm_text"] = norm
    return {"label": label, "prob": prob, "debug": debug}

st.set_page_config(page_title="Spam Classifier", page_icon="📨", layout="wide")
st.title("📨 AI Spam Classifier")
st.subheader("Spam detection using ML (with option for trusted senders)")

st.sidebar.header("MENU📌")
st.sidebar.image("logo_st.jpg", use_container_width=True)
st.sidebar.markdown("""
**Team members👥**
- Rajasekar P  
- Saravanakumar S  
- Raghav R K
""")

st.sidebar.markdown("---")
st.sidebar.markdown("COLLEGE🎓")
st.sidebar.image("pmist_logo.st.jpg", use_container_width=True)
st.sidebar.success("PMIST , Vallam")
st.sidebar.markdown("---")
st.sidebar.markdown("PROJECT🤖")
st.sidebar.info("A web app where you can check if an email/sms is spam or not , we used sender option (for trusted senders like isp, banks etc..)Model is built using Scikit-learn(LogisticRegression) and deployed as a public app using streamlit...")

st.markdown("**Enter message & optional sender.** If sender is trusted (Airtel/Jio/Banks), the app is conservative. Unknown senders get stricter checks.")

sender_input = st.text_input("Sender (optional — e.g., AIRTEL / JIO / SBI / ICICI)", value="")

option = st.radio("Choose input type:", ["📝Enter a Message", "📁Upload a .txt file"],index=None)
raw_text = ""
if option == "📝Enter a Message":
    raw_text = st.text_area("Enter the message you want to check", height=160)
elif option == "📁Upload a .txt file":
    uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded is not None:
        raw_text = uploaded.read().decode("utf-8", errors="ignore")

if st.button("Predict"):
    if not raw_text or raw_text.strip() == "":
        st.warning("Please enter or upload some text to predict.")
    else:
        with st.spinner("Classifying..."):
            res = decide_label_with_rules(raw_text, sender=sender_input)
        label = res["label"]
        prob = res["prob"]
        dbg = res["debug"]

        if dbg["trusted_sender"]:
            st.success(f"Sender '{sender_input}' is trusted ✅")
        else:
            st.warning("Sender not trusted🚫— applying stricter rules")

        if label == 1:
            conf_text = f"{prob*100:.1f}%" if prob is not None else "N/A"
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:#ff4d4d;color:white'>"
                        f"<strong>🚨 SPAM</strong> — Confidence: {conf_text}</div>", unsafe_allow_html=True)
        else:
            conf_text = f"{(100 - prob*100):.1f}%" if prob is not None else "N/A"
            st.markdown(f"<div style='padding:12px;border-radius:8px;background:#2ecc71;color:white'>"
                        f"<strong>✅ NOT SPAM</strong> — Confidence: {conf_text}</div>", unsafe_allow_html=True)





















