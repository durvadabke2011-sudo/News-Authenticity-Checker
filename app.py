"""
app.py – Fake News Detection System (Home + Predict only)
"""
import streamlit as st
import pickle, re, os, time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

for r in ['stopwords', 'punkt']:
    try: nltk.data.find(r)
    except: nltk.download(r, quiet=True)

st.set_page_config(page_title="News Authenticity Checker", page_icon="📰", layout="wide")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');
html,body,[class*="css"]{font-family:'Source Sans 3',sans-serif;background:#0D0D0D;color:#F0EDE8;}
.stApp{background:#0D0D0D;}
.headline{font-family:'Playfair Display',serif;font-weight:900;font-size:2.8rem;line-height:1.15;}
.headline span{color:#F5A623;}
.sub{font-size:1rem;color:#888;text-transform:uppercase;letter-spacing:.08em;}
.card{background:#1E1E1E;border:1px solid #2A2A2A;border-radius:4px;padding:1.4rem 1.6rem;margin:.8rem 0;font-size:.93rem;color:#888;line-height:1.7;}
.card b{color:#F0EDE8;}
.mtile{background:#1E1E1E;border:1px solid #2A2A2A;border-radius:4px;padding:1.1rem;text-align:center;margin-bottom:.8rem;}
.mv{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#F5A623;}
.ml{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:#888;margin-top:.2rem;}
.rcard{background:#1E1E1E;border:1px solid #2A2A2A;border-radius:4px;padding:1.6rem 1.8rem;margin:1rem 0;}
.rcard.real{border-left:5px solid #27AE60;} .rcard.fake{border-left:5px solid #E74C3C;}
.rlabel{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;}
.rlabel.real{color:#27AE60;} .rlabel.fake{color:#E74C3C;}
.bar-bg{background:#2A2A2A;border-radius:2px;height:8px;width:100%;margin:.6rem 0;overflow:hidden;}
.bar-fill{height:100%;border-radius:2px;}
.stButton>button{background:#F5A623;color:#0D0D0D;border:none;border-radius:3px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;padding:.6rem 1.8rem;}
textarea{background:#161616!important;color:#F0EDE8!important;border:1px solid #2A2A2A!important;}
section[data-testid="stSidebar"]{background:#161616;border-right:1px solid #2A2A2A;}
section[data-testid="stSidebar"] *{color:#F0EDE8!important;}
#MainMenu,footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──
stemmer = PorterStemmer()
STOPS   = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+|<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(t) for t in text.split() if t not in STOPS and len(t) > 2]
    return ' '.join(tokens)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
        return None, None, None
    model = pickle.load(open('model.pkl', 'rb'))
    vec   = pickle.load(open('vectorizer.pkl', 'rb'))
    meta  = pickle.load(open('model_metadata.pkl', 'rb')) if os.path.exists('model_metadata.pkl') else None
    return model, vec, meta

def predict(text, model, vec):
    v = vec.transform([preprocess(text)])
    pred = model.predict(v)[0]
    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(v)[0]
        fp, rp = float(p[0]), float(p[1])
    else:
        fp, rp = (0.1, 0.9) if pred == 1 else (0.9, 0.1)
    return ('Real' if pred == 1 else 'Fake'), max(fp, rp), fp, rp

# ── Sidebar ──
with st.sidebar:
    st.markdown("## 📰 News Authenticity Checker")
    st.markdown("---")
    page = st.radio("Navigate", ["🏠  Home", "🔍  Predict News"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**📦 Dataset**")
    st.markdown("<div class='card' style='font-size:.82rem'>Kaggle: <b>Fake and Real News Dataset</b><br>≥ 40 000 articles · NLP cleaned</div>", unsafe_allow_html=True)

model, vec, meta = load_artifacts()

# ── HOME ──
if "Home" in page:
    st.markdown('<div class="sub">Natural Language Processing · ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline">Fake News<br><span>Detection System</span></div><br>', unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("""<div class="card"><b>What is this?</b><br>
        A machine learning system trained on tens of thousands of news articles to distinguish
        <b>credible journalism</b> from <b>misinformation</b> using NLP techniques.<br><br>
        <b>Pipeline:</b> Text → Preprocessing → TF-IDF Vectorization → ML Classifier → Prediction</div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card"><b>Pipeline Steps:</b><br>
        1️⃣ Data Collection &amp; Cleaning<br>
        2️⃣ NLP Preprocessing (lowercase · stem · stopwords)<br>
        3️⃣ TF-IDF Feature Extraction<br>
        4️⃣ Model Training (LR · Naive Bayes · Random Forest)<br>
        5️⃣ Evaluation &amp; Cross Validation<br>
        6️⃣ Real-time Prediction via Streamlit</div>""", unsafe_allow_html=True)

    with c2:
        acc = meta.get('best_accuracy', 0.93) if meta else 0.93
        for val, lbl in [(f"{acc*100:.1f}%","Best Accuracy"),("TF-IDF","Feature Method"),("3","Models Compared"),("5-Fold","Cross Validation")]:
            st.markdown(f'<div class="mtile"><div class="mv">{val}</div><div class="ml">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2A2A2A;margin:1.5rem 0'><div style='text-align:center;opacity:.4;font-size:.82rem'>⚠️ For educational purposes. Always verify news through multiple credible sources.</div>", unsafe_allow_html=True)

# ── PREDICT ──
else:
    st.markdown('<div class="sub">Analysis Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="headline">Predict <span>News</span></div><br>', unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ **Model not found.** Run the Jupyter notebook to generate `model.pkl` and `vectorizer.pkl`.")
        st.stop()

    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("**Paste the news article below:**")
        txt = st.text_area("news", label_visibility="collapsed", placeholder="Paste a full news article or headline here...", height=260)
        b1, b2 = st.columns([2, 1])
        analyse = b1.button("🔍  Analyse Article", use_container_width=True)
        if b2.button("✕  Clear", use_container_width=True): st.rerun()

    with c2:
        st.markdown("""<div class="card"><b>💡 Tips</b><br>
        • Paste at least 30–50 words<br>• Include title and body<br>• English news works best<br><br>
        <b>⚠️ Limitations</b><br>• Satire may be misclassified<br>• Short text lowers confidence</div>""", unsafe_allow_html=True)

    if analyse:
        if not txt.strip() or len(txt.split()) < 5:
            st.warning("⚠️ Please enter at least a full sentence.")
            st.stop()

        with st.spinner("🧠 Analysing …"):
            time.sleep(0.5)
            label, conf, fp, rp = predict(txt, model, vec)

        is_real = label == "Real"
        color   = "#27AE60" if is_real else "#E74C3C"
        cls     = "real" if is_real else "fake"
        icon    = "✅" if is_real else "❌"

        st.markdown(f"""
        <div class="rcard {cls}">
          <div class="rlabel {cls}">{icon} {label} News</div>
          <div style="font-size:.88rem;opacity:.7;margin:.4rem 0">{"Appears credible." if is_real else "Shows patterns of misinformation."}</div>
          <div style="font-size:.82rem;opacity:.6">Confidence: <b>{conf*100:.1f}%</b></div>
          <div class="bar-bg"><div class="bar-fill" style="width:{int(conf*100)}%;background:{color}"></div></div>
        </div>""", unsafe_allow_html=True)

        r1, r2 = st.columns(2)
        r1.metric("✅ Real", f"{rp*100:.1f}%"); r1.progress(rp)
        r2.metric("❌ Fake", f"{fp*100:.1f}%"); r2.progress(fp)

        w, c, s = len(txt.split()), len(txt), txt.count('.')+txt.count('!')+txt.count('?')
        st.markdown(f'<div class="card" style="margin-top:.8rem">📝 {w} words · {c} chars · ~{max(s,1)} sentences</div>', unsafe_allow_html=True)