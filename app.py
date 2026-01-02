import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
from io import BytesIO
import json
import base64
import os
import re
import time
import string
import asyncio
import random
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Dict, Any

# --- OpenAI ---
from openai import OpenAI, AsyncOpenAI

# --- Optional advanced libs (graceful fallback) ---
try:
    import instructor  # type: ignore
    from pydantic import BaseModel, Field  # type: ignore
    INSTRUCTOR_OK = True
except Exception:
    INSTRUCTOR_OK = False
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    BM25_OK = True
except Exception:
    BM25_OK = False
    BM25Okapi = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    import torch  # type: ignore
    SEM_OK = True
except Exception:
    SEM_OK = False
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    torch = None  # type: ignore

try:
    from rapidfuzz import fuzz  # type: ignore
    FUZZ_OK = True
except Exception:
    FUZZ_OK = False
    fuzz = None  # type: ignore

# OPTIONAL: Camelot per PDF nativi
try:
    import camelot  # type: ignore
    CAMELOT_OK = True
except Exception:
    CAMELOT_OK = False

# =========================================================
# CONFIG "BEST" (no UI knobs)
# =========================================================
AI_MODEL_VISION = "gpt-4o"
AI_MODEL_REASON = "gpt-4o"

# Vision: process ALL pages if used
VISION_FORCE = False            # True = usa sempre Vision (costoso). False = AUTO: pdfplumber -> fallback Vision
VISION_CONCURRENCY = 5          # parallel calls (moderate)
VISION_DPI_SCALE = 2.0          # render scale for images

# Estrazione pdfplumber
ROW_CLUSTER_TOL = 2.8
SPLIT_FALLBACK_RATIO = 0.52

# Filtri conti movimentabili
MIN_CODE_LEN = 4                # conti "foglia" di solito >= 4
DROP_GROUP_PREFIX = True        # se un codice è prefisso di un altro, lo considero raggruppamento
DROP_TOTALI_E_HEADER = True

# Classificazione: pesi e soglie (fisse)
TOPK_RETRIEVAL = 18
DECISION_SCORE_OK = 0.86
DECISION_GAP_OK = 0.10

W_SEM = 0.60
W_BM25 = 0.22
W_KW = 0.18

# Penalità mismatch macro (solo se macro è affidabile)
PENALTY_TIPO = 0.10
PENALTY_BUCKET = 0.12

# Bonus match PDC bucket
BONUS_PDC_BUCKET = 0.06

# =========================================================
# UI / STILE
# =========================================================
st.set_page_config(
    page_title="Revilaw AuditEv Hierarchy (Quality)",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

st.markdown(r"""
<style>
    /* Dashboard cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #e6e6e6;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }

    /* Sidebar panel emphasis */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 100%);
        border-right: 1px solid #e6e6e6;
    }

    .ctl-box {
        background: #ffffff;
        border: 1px solid #dfe7f3;
        border-left: 6px solid #003366;
        padding: 12px 12px 8px 12px;
        border-radius: 12px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .ctl-title{
        font-weight: 800;
        color: #003366;
        margin-bottom: 6px;
    }
    .ctl-sub{
        color: #51607a;
        font-size: 0.88rem;
        margin-top: -2px;
        margin-bottom: 8px;
    }

    div.stButton > button:first-child {
        background-color: #003366;
        color: white;
        font-weight: 800;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        width: 100%;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #004080;
    }

    h1,h2,h3 {
        color: #003366;
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS MODELLI CLASSIFICAZIONE
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

CLASSIFICATION_CANDIDATES = {
    "Bilancio ordinario": [
        BASE_DIR / "Modello_classificazione_Ordinario_con_keywords.xlsx",
        BASE_DIR / "Modello classificazione Ordinario_pulito per test.xlsx",
    ],
    "Bilancio abbreviato": [
        BASE_DIR / "Modello_classificazione_Abbreviato_con_keywords.xlsx",
        BASE_DIR / "Modello classificazione Abbreviato_pulito per test.xlsx",
    ],
}

def pick_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None

# =========================================================
# API KEY
# =========================================================
def get_api_key() -> Optional[str]:
    try:
        return st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None

api_key = get_api_key() or os.getenv("OPENAI_API_KEY")

# =========================================================
# MODELLI DATI (Vision)
# =========================================================
if INSTRUCTOR_OK:
    class RigaVision(BaseModel):
        codice: str = Field("", description="Codice del conto (vuoto se assente)")
        descrizione: str = Field(..., description="Descrizione del conto")
        sezione: Literal["ATTIVO","PASSIVO","COSTI","RICAVI","SP","CE","UNKNOWN"] = Field("UNKNOWN")
        importo: float = Field(..., description="Importo (con segno se presente)")
        is_raggruppamento: bool = Field(False, description="True se titolo/subtotale/totale")

    class VisionEstratto(BaseModel):
        righe: List[RigaVision]
else:
    RigaVision = None
    VisionEstratto = None

# =========================================================
# SESSION STATE
# =========================================================
def ss_init():
    defaults = {
        "data_processed": None,
        "combo_options": [],
        "map_code_desc": {},
        "map_code_natura": {},
        "map_code_tipo": {},
        "map_code_bucket": {},
        "logs": [],
        "raw_rows": [],
        "estrazione_metodo": "N/D",
        "debug_counts": {},
        "risultato_pdf": None,
        "risultato_pdf_desc": None,
        "warnings": [],
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k]=v

ss_init()

def log(step: str, msg: str, level: str="INFO") -> None:
    st.session_state["logs"].append({
        "time": time.strftime("%H:%M:%S"),
        "level": level,
        "step": step,
        "msg": msg
    })

# =========================================================
# ASYNC SAFE RUNNER
# =========================================================
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    return asyncio.run(coro)

# =========================================================
# UTILS
# =========================================================
IMPORTO_RE = r"[-]?\(?\d{1,3}(?:\.\d{3})*(?:,\d{2})\)?"

PAT_FOOTER = re.compile(r"(Operatore:|Pagina\s+\d+\s+di\s+\d+|DK SET|SITUAZIONE CONTABILE)", re.IGNORECASE)

PAT_RAGGR = re.compile(
    r"^(totale|somma|subtotale|saldo|risultato|utile|perdita|"
    r"attivo|passivo|patrimonio|conto economico|stato patrimoniale)\b",
    re.IGNORECASE
)

RIS_PATTERNS = [
    re.compile(r"perdita\s*d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"utile\s*d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"risultato\s*d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"risultato\s+netto", re.IGNORECASE),
    re.compile(r"utile\s+del\s+periodo", re.IGNORECASE),
    re.compile(r"perdita\s+del\s+periodo", re.IGNORECASE),
    re.compile(r"utile\s*\(.*perdita.*\)\s*d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"utile\s*/\s*perdita\s*d[' ]?esercizio", re.IGNORECASE),
]

def normalize_desc(desc: str) -> str:
    return re.sub(r"\s+", " ", (desc or "").strip())

def simple_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return text.translate(str.maketrans("", "", string.punctuation)).split()

def normalizza_codice(codice: Any) -> str:
    if not codice:
        return ""
    return re.sub(r"[^0-9A-Za-z]", "", str(codice))

def parse_importo(x: Any) -> float:
    if x is None:
        return 0.0
    s = str(x).strip().replace("€", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0

def is_probably_header(text: str) -> bool:
    if not text:
        return False
    up = text.upper().strip()
    if PAT_FOOTER.search(up):
        return True
    if "STATO PATRIMONIALE" in up or "CONTO ECONOMICO" in up:
        return True
    if "ATTIVITA" in up or "ATTIVO" in up or "PASSIVITA" in up or "PASSIVO" in up:
        return True
    if ("COSTI" in up and "RICAVI" in up) or ("VALORE DELLA PRODUZIONE" in up):
        return True
    if PAT_RAGGR.match(text.strip()):
        return True
    if up == up.upper() and len(up) <= 35 and not re.match(r"^\d", up):
        return True
    return False

def e_raggruppamento(r: Dict[str, Any]) -> bool:
    if bool(r.get("is_raggruppamento", False)):
        return True
    desc = (r.get("descrizione") or "").strip()
    cod = (r.get("codice") or "").strip()
    if PAT_RAGGR.match(desc):
        return True
    if cod == "" and (len(desc) <= 40) and (desc.upper() == desc):
        return True
    low = desc.lower()
    if any(x in low for x in ["totale", "subtotale", "somma", "saldo", "risultato", "utile", "perdita"]) and cod == "":
        return True
    return False

def inferisci_tipo_bucket(natura_str: str) -> Tuple[str, Optional[str]]:
    s = (natura_str or "").strip().upper()
    if s == "A":
        return "SP", "ATTIVO"
    if s == "P":
        return "SP", "PASSIVO"
    if s == "C":
        return "CE", "COSTI"
    if s == "R":
        return "CE", "RICAVI"

    tipo = None
    bucket = None

    if any(x in s for x in ["CE", "ECON", "RICAV", "COST", "ONERI", "PROVENT"]):
        tipo = "CE"
    if any(x in s for x in ["SP", "PATR", "ATTIV", "PASSIV"]):
        tipo = "SP"

    if "SP-A" in s or ("SP" in s and "ATT" in s):
        bucket = "ATTIVO"; tipo = tipo or "SP"
    elif "SP-P" in s or ("SP" in s and "PASS" in s):
        bucket = "PASSIVO"; tipo = tipo or "SP"
    elif "CE-C" in s or ("CE" in s and ("COST" in s or "ONER" in s)):
        bucket = "COSTI"; tipo = tipo or "CE"
    elif "CE-R" in s or ("CE" in s and ("RICAV" in s or "PROVENT" in s)):
        bucket = "RICAVI"; tipo = tipo or "CE"

    if tipo not in ("SP", "CE"):
        tipo = "SP"
    return tipo, bucket

def bucket_to_tipo(bucket: Optional[str]) -> Optional[str]:
    if bucket in ("ATTIVO","PASSIVO"):
        return "SP"
    if bucket in ("COSTI","RICAVI"):
        return "CE"
    return None

def expected_natura_da_bucket(bucket: Optional[str]) -> Optional[str]:
    # Natura "attesa" per il saldo: ATTIVO/COSTI -> DARE; PASSIVO/RICAVI -> AVERE
    if bucket in ("ATTIVO", "COSTI"):
        return "DARE"
    if bucket in ("PASSIVO", "RICAVI"):
        return "AVERE"
    return None

# =========================================================
# PDC HELPERS
# =========================================================
def estrai_pdc_da_pdf(file_obj) -> dict:
    map_pdc = {}
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                match = re.search(r"^(\d[\dA-Za-z\.\-]*)\s+(.+)", line.strip())
                if match:
                    code_raw = match.group(1)
                    desc = match.group(2).strip()
                    clean_code = normalizza_codice(code_raw)
                    if len(clean_code) >= 2:
                        map_pdc[clean_code] = desc
    return map_pdc

def analizza_pdc_universale(file_obj) -> dict:
    pdc_norm = {}
    if file_obj.name.lower().endswith(".pdf"):
        return estrai_pdc_da_pdf(file_obj)

    df = None
    try:
        if file_obj.name.lower().endswith(".xlsx"):
            df = pd.read_excel(file_obj, dtype=str)
        else:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, sep=None, engine="python", dtype=str)
    except Exception:
        df = None

    if df is not None and not df.empty:
        df = df.astype(str)
        for _, r in df.iterrows():
            clean_code = normalizza_codice(r.iloc[0])
            desc = str(r.iloc[-1]).strip()
            if clean_code:
                pdc_norm[clean_code] = desc
    return pdc_norm

def build_pdc_from_rows(rows: List[dict]) -> dict:
    # Usa i codici e descrizioni presenti nel bilancio come PDC minimo (sempre disponibile)
    out = {}
    for r in rows:
        code = normalizza_codice(r.get("codice",""))
        desc = normalize_desc(str(r.get("descrizione","")))
        if code and desc and len(code) >= 2:
            # preferisci descrizioni più lunghe
            if (code not in out) or (len(desc) > len(out[code])):
                out[code] = desc
    return out

def trova_padre_gerarchico(codice_raw: str, pdc_norm_map: dict) -> Tuple[Optional[str], Optional[str], int]:
    """
    Ritorna (cod_padre, desc_padre, profondita_tagli) dove profondita_tagli=0 se match esatto.
    """
    if not codice_raw or not pdc_norm_map:
        return None, None, 999
    clean = normalizza_codice(codice_raw)
    if clean in pdc_norm_map:
        return clean, pdc_norm_map[clean], 0
    cut = 0
    while len(clean) > 1:
        clean = clean[:-1]
        cut += 1
        if clean in pdc_norm_map:
            return clean, f"(Gruppo Padre) {pdc_norm_map[clean]}", cut
    return None, None, 999

def infer_macro_da_testo_pdc(pdc_text: str) -> Tuple[Optional[str], Optional[str], int]:
    """
    (tipo, bucket, conf 0-100) da descrizione PDC.
    """
    if not pdc_text:
        return None, None, 0
    t = pdc_text.lower()

    # segnali forti
    if any(k in t for k in ["ricavi", "vendite", "corrispettivi", "fatturato", "proventi", "interessi attivi", "plusval"]):
        return "CE", "RICAVI", 85
    if any(k in t for k in ["costi", "acquisti", "servizi", "personale", "ammort", "oneri", "spese", "canoni", "consulenz", "utenze", "imposte", "tasse", "interessi passivi"]):
        return "CE", "COSTI", 80
    if any(k in t for k in ["cassa", "banca", "crediti", "immobilizz", "rimanenz", "iva credito", "ratei attivi", "depositi", "anticipi"]):
        return "SP", "ATTIVO", 78
    if any(k in t for k in ["debiti", "mutuo", "finanzi", "erario", "tfr", "fondo", "iva debito", "ratei passivi", "patrimonio netto", "capitale", "riserva", "utili portati a nuovo"]):
        return "SP", "PASSIVO", 78

    return None, None, 0

def infer_tipo_da_codice_pdc(codice_raw: str) -> Tuple[Optional[str], int]:
    """
    Heuristica: molti PDC hanno classi 6-9 per CE.
    """
    c = normalizza_codice(codice_raw)
    if not c:
        return None, 0
    # prendi primo carattere numerico
    m = re.search(r"\d", c)
    if not m:
        return None, 0
    d = c[m.start()]
    if d in ("6","7","8","9"):
        return "CE", 70
    if d in ("0","1","2","3","4","5"):
        return "SP", 55
    return None, 0

def infer_macro_da_desc(desc: str) -> Tuple[Optional[str], Optional[str], int]:
    d = (desc or "").lower()
    if any(k in d for k in ["ricavi", "vendite", "corrispettivi", "proventi", "fatturato", "interessi attivi"]):
        return "CE","RICAVI",70
    if any(k in d for k in ["acquisti", "servizi", "personale", "ammort", "oneri", "spese", "canoni", "consulenz", "utenze"]):
        return "CE","COSTI",65
    if any(k in d for k in ["cassa", "banca", "crediti", "immobilizz", "rimanenz", "iva credito", "ratei attivi"]):
        return "SP","ATTIVO",60
    if any(k in d for k in ["debiti", "fornitori", "mutuo", "finanzi", "erario", "tfr", "fondo", "iva debito", "ratei passivi", "patrimonio netto", "capitale", "riserva"]):
        return "SP","PASSIVO",60
    return None,None,0

# =========================================================
# CLASSIFICAZIONE INDEX (BM25 + SEM + KEYWORDS)
# =========================================================
@st.cache_resource
def carica_modello_semantico():
    if not SEM_OK:
        return None
    try:
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None

semantic_model = carica_modello_semantico()

@st.cache_data
def inizializza_motori_ricerca(df_classificazione: pd.DataFrame) -> Optional[dict]:
    if df_classificazione is None or df_classificazione.empty:
        return None

    df = df_classificazione.copy()
    df.columns = [c.strip() for c in df.columns]

    # solo classificabili
    if "Classificabile" in df.columns:
        df = df[df["Classificabile"].astype(str).str.lower().str.strip() == "x"].copy()

    col_cod = next((c for c in df.columns if c.lower().startswith("cod")), df.columns[0])
    col_desc = next((c for c in df.columns if "descr" in c.lower()), df.columns[1])
    col_natura = next((c for c in df.columns if "natura" in c.lower() or "tipo" in c.lower()), None)
    col_kw = next((c for c in df.columns if "keyword" in c.lower()), None)

    codici: List[str] = []
    desc_sem: List[str] = []
    toks: List[List[str]] = []

    map_desc: Dict[str,str] = {}
    map_natura: Dict[str,str] = {}
    map_tipo: Dict[str,str] = {}
    map_bucket: Dict[str,Optional[str]] = {}
    map_kw_phr: Dict[str,List[str]] = {}
    map_kw_tokens: Dict[str,set] = {}

    combo: List[str] = []

    # inverted index: token -> set(idx)
    kw_inv: Dict[str,set] = {}

    idx_attivo: List[int] = []
    idx_passivo: List[int] = []
    idx_costi: List[int] = []
    idx_ricavi: List[int] = []
    idx_sp: List[int] = []
    idx_ce: List[int] = []

    for pos, (_, row) in enumerate(df.iterrows()):
        code = str(row.get(col_cod, "")).strip()
        desc = normalize_desc(str(row.get(col_desc, "")).strip())
        nat = str(row.get(col_natura, "")).strip() if col_natura else ""
        kw_raw = str(row.get(col_kw, "")).strip() if col_kw else ""

        tipo, bucket = inferisci_tipo_bucket(nat)

        # testo per embedding/BM25: descrizione + keywords (se presenti)
        join_text = desc
        kw_phrases: List[str] = []
        if kw_raw and kw_raw.lower() not in ("nan","none"):
            kw_phrases = [normalize_desc(x) for x in re.split(r"[;\n]+", kw_raw) if normalize_desc(x)]
            # aggiungi keywords al testo semantico (ma con separatore)
            join_text = desc + " | " + " | ".join(kw_phrases[:20])

        codici.append(code)
        desc_sem.append(join_text)
        toks.append(simple_tokenize(join_text))

        map_desc[code] = desc
        map_natura[code] = nat
        map_tipo[code] = tipo
        map_bucket[code] = bucket
        map_kw_phr[code] = kw_phrases

        kw_tokens_set = set()
        for phr in kw_phrases:
            for tok in simple_tokenize(phr):
                if len(tok) >= 3:
                    kw_tokens_set.add(tok)
                    kw_inv.setdefault(tok, set()).add(pos)
        map_kw_tokens[code] = kw_tokens_set

        combo.append(f"{code} | {desc[:60]}")

        # indici bucket
        if bucket == "ATTIVO":
            idx_attivo.append(pos); idx_sp.append(pos)
        elif bucket == "PASSIVO":
            idx_passivo.append(pos); idx_sp.append(pos)
        elif bucket == "COSTI":
            idx_costi.append(pos); idx_ce.append(pos)
        elif bucket == "RICAVI":
            idx_ricavi.append(pos); idx_ce.append(pos)
        else:
            # fallback
            idx_sp.append(pos)

    emb = None
    if SEM_OK and semantic_model is not None:
        try:
            emb = semantic_model.encode(desc_sem, convert_to_tensor=True)
        except Exception:
            emb = None

    bm25 = None
    if BM25_OK:
        try:
            bm25 = BM25Okapi(toks)
        except Exception:
            bm25 = None

    return {
        "codici": codici,
        "desc_sem": desc_sem,
        "emb": emb,
        "bm25": bm25,
        "map_desc": map_desc,
        "map_natura": map_natura,
        "map_tipo": map_tipo,
        "map_bucket": map_bucket,
        "map_kw_phr": map_kw_phr,
        "map_kw_tokens": map_kw_tokens,
        "kw_inv": kw_inv,
        "combo": sorted(list(set(combo))),
        "idx_attivo": idx_attivo,
        "idx_passivo": idx_passivo,
        "idx_costi": idx_costi,
        "idx_ricavi": idx_ricavi,
        "idx_sp": idx_sp,
        "idx_ce": idx_ce,
        "has_sem": emb is not None,
        "has_bm25": bm25 is not None
    }

def kw_candidate_indices(query: str, db: dict, limit: int = 40) -> List[int]:
    toks = [t for t in simple_tokenize(query) if len(t) >= 3]
    counter: Dict[int,int] = {}
    for t in toks:
        for idx in db["kw_inv"].get(t, set()):
            counter[idx] = counter.get(idx, 0) + 1
    if not counter:
        return []
    best = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [i for i,_ in best]

def kw_score(query: str, code: str, db: dict) -> float:
    # 0..1
    q_tokens = set([t for t in simple_tokenize(query) if len(t) >= 3])
    kw_tokens = db["map_kw_tokens"].get(code, set()) or set()
    if not kw_tokens:
        # piccolo scoring su descrizione
        desc = (db["map_desc"].get(code,"") or "").lower()
        if not desc:
            return 0.0
        if any(t in desc for t in q_tokens):
            return 0.18
        return 0.0

    inter = len(q_tokens.intersection(kw_tokens))
    tok_score = inter / max(1, min(len(q_tokens), 10))

    # fuzzy max su frasi keyword (solo se rapidfuzz disponibile)
    fuzz_score = 0.0
    if FUZZ_OK:
        best = 0
        for phr in db["map_kw_phr"].get(code, [])[:15]:
            if not phr:
                continue
            v = fuzz.partial_ratio(query.lower(), phr.lower())
            if v > best:
                best = v
        fuzz_score = best / 100.0

    return max(tok_score * 0.55 + fuzz_score * 0.45, 0.0)

def ricerca_ibrida(query: str, db: dict, allowed_indices: List[int]) -> List[dict]:
    """
    Ritorna candidati con score (0..1 circa). Unisce SEM, BM25 e keyword hits.
    """
    allowed = set(allowed_indices) if allowed_indices else set(range(len(db["codici"])))

    cands: Dict[int, Dict[str, Any]] = {}

    # --- BM25 ---
    if db.get("has_bm25") and db["bm25"] is not None:
        toks = simple_tokenize(query)
        scores = db["bm25"].get_scores(toks)
        # normalizza
        mx = max(scores) if len(scores) else 1.0
        if mx <= 0:
            mx = 1.0
        for i, s in enumerate(scores):
            if i not in allowed:
                continue
            if s <= 0:
                continue
            sc = float(s) / float(mx)
            if i not in cands:
                cands[i] = {"i": i, "bm25": sc, "sem": 0.0, "kw": 0.0}
            else:
                cands[i]["bm25"] = max(cands[i]["bm25"], sc)

    # --- SEM ---
    if db.get("has_sem") and db["emb"] is not None and semantic_model is not None:
        try:
            q_emb = semantic_model.encode(query, convert_to_tensor=True)
            sem_sc = util.cos_sim(q_emb, db["emb"])[0]
            # maschera allowed
            if torch is not None:
                mask = torch.zeros_like(sem_sc)
                if allowed:
                    idx_t = torch.tensor(list(allowed), dtype=torch.long)
                    mask[idx_t] = 1.0
                    sem_sc = sem_sc * mask
                topk = min(TOPK_RETRIEVAL, sem_sc.shape[0])
                vals, idxs = torch.topk(sem_sc, k=topk)
                for v, i in zip(vals, idxs):
                    ii = int(i)
                    if ii not in allowed:
                        continue
                    sc = float(v)
                    # cos_sim può essere negativo, clamp
                    if sc < 0:
                        continue
                    if ii not in cands:
                        cands[ii] = {"i": ii, "bm25": 0.0, "sem": sc, "kw": 0.0}
                    else:
                        cands[ii]["sem"] = max(cands[ii]["sem"], sc)
        except Exception:
            pass

    # --- KW candidates (ensure recall) ---
    for ii in kw_candidate_indices(query, db, limit=50):
        if ii not in allowed:
            continue
        if ii not in cands:
            cands[ii] = {"i": ii, "bm25": 0.0, "sem": 0.0, "kw": 0.0}

    # compute kw for each candidate
    out = []
    for ii, d in cands.items():
        code = db["codici"][ii]
        d["kw"] = kw_score(query, code, db)
        out.append(d)

    # final sort later (after macro penalty/bonus)
    return out

def decide_macro_for_row(r: dict, pdc_map: dict) -> Tuple[Optional[str], Optional[str], int, Dict[str, Any]]:
    """
    Decide macro (tipo, bucket, conf) combinando:
    - sezione PDF (se c'è) -> fortissima
    - PDC desc/gerarchia
    - codice PDC (prima cifra)
    - descrizione
    """
    evidence: Dict[str, Any] = {}
    sezione = (r.get("sezione") or "").upper().strip()
    desc = (r.get("descrizione") or "")
    cod = (r.get("codice") or "")

    # 1) PDF sezione
    if sezione in ("ATTIVO","PASSIVO","COSTI","RICAVI"):
        tipo = bucket_to_tipo(sezione)
        evidence["pdf_sezione"] = sezione
        return tipo, sezione, 95, evidence

    # 2) PDC match (self or parent)
    cod_padre, desc_padre, cut = trova_padre_gerarchico(cod, pdc_map)
    pdc_text = ""
    if cod_padre and desc_padre:
        pdc_text = desc_padre
    evidence["pdc_text"] = pdc_text

    t_pdc, b_pdc, conf_pdc = infer_macro_da_testo_pdc(pdc_text)
    if conf_pdc >= 78:
        evidence["pdc_infer"] = {"tipo": t_pdc, "bucket": b_pdc, "conf": conf_pdc}
        return t_pdc, b_pdc, conf_pdc, evidence

    # 3) codice PDC (prima cifra)
    t_code, conf_code = infer_tipo_da_codice_pdc(cod)
    if conf_code >= 60:
        evidence["code_infer"] = {"tipo": t_code, "conf": conf_code}
        # bucket ancora incerto: prova dalla descrizione
        t_d, b_d, c_d = infer_macro_da_desc(desc)
        if b_d and t_d == t_code and c_d >= 60:
            return t_code, b_d, min(75, conf_code + 10), evidence
        return t_code, None, conf_code, evidence

    # 4) descrizione
    t_d, b_d, c_d = infer_macro_da_desc(desc)
    if c_d >= 60:
        evidence["desc_infer"] = {"tipo": t_d, "bucket": b_d, "conf": c_d}
        return t_d, b_d, c_d, evidence

    return None, None, 0, evidence

def allowed_indices_for_macro(db: dict, tipo: Optional[str], bucket: Optional[str], conf: int) -> Tuple[List[int], float]:
    """
    Ritorna allowed_indices e macro_strength (0..1) per penalità/bonus.
    """
    strength = 0.0
    if conf >= 90:
        strength = 1.0
    elif conf >= 80:
        strength = 0.8
    elif conf >= 65:
        strength = 0.6
    else:
        strength = 0.0

    if bucket == "ATTIVO":
        return db["idx_attivo"], strength
    if bucket == "PASSIVO":
        return db["idx_passivo"], strength
    if bucket == "COSTI":
        return db["idx_costi"], strength
    if bucket == "RICAVI":
        return db["idx_ricavi"], strength

    if tipo == "SP":
        return db["idx_sp"], strength * 0.7
    if tipo == "CE":
        return db["idx_ce"], strength * 0.7

    return list(range(len(db["codici"]))), 0.0

def rerank_candidates(query: str, base_cands: List[dict], db: dict,
                      macro_tipo: Optional[str], macro_bucket: Optional[str],
                      macro_strength: float,
                      pdc_bucket: Optional[str], pdc_conf: int) -> List[dict]:
    out = []
    for c in base_cands:
        i = c["i"]
        code = db["codici"][i]
        tipo = db["map_tipo"].get(code, "SP")
        bucket = db["map_bucket"].get(code, None)

        sem = float(c.get("sem", 0.0))
        bm = float(c.get("bm25", 0.0))
        kw = float(c.get("kw", 0.0))

        # normalize sem: cos_sim ~0..1
        sem_n = max(0.0, min(1.0, sem))
        bm_n = max(0.0, min(1.0, bm))
        kw_n = max(0.0, min(1.0, kw))

        score = W_SEM * sem_n + W_BM25 * bm_n + W_KW * kw_n

        # penalties if macro is confident
        if macro_strength > 0:
            if macro_tipo and tipo != macro_tipo:
                score -= PENALTY_TIPO * macro_strength
            if macro_bucket and bucket and bucket != macro_bucket:
                score -= PENALTY_BUCKET * macro_strength
            if macro_bucket and bucket is None:
                score -= (PENALTY_BUCKET * 0.4) * macro_strength

        # bonus: pdc bucket align
        if pdc_bucket and pdc_conf >= 80 and bucket == pdc_bucket:
            score += BONUS_PDC_BUCKET

        out.append({
            "codice": code,
            "descrizione": db["map_desc"].get(code, ""),
            "tipo": tipo,
            "bucket": bucket,
            "score": float(score),
            "sem": sem_n, "bm25": bm_n, "kw": kw_n
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:TOPK_RETRIEVAL]

def decisione_senza_ai(cands: List[dict]) -> bool:
    if not cands:
        return False
    if cands[0]["score"] >= DECISION_SCORE_OK:
        if len(cands) == 1:
            return True
        if (cands[0]["score"] - cands[1]["score"]) >= DECISION_GAP_OK:
            return True
    return False

def scegli_fallback(cands: List[dict], db: dict) -> Tuple[str, str]:
    if cands:
        return cands[0]["codice"], "Fallback: scelto top candidato (score)"
    first = next(iter(db["map_desc"].keys()), "")
    return first, "Fallback: nessun candidato, scelto primo codice disponibile"

def ragionatore_ai(client: OpenAI, row: dict, macro: dict, cands: List[dict], db: dict) -> Tuple[str, str]:
    """
    AI sceglie SOLO tra candidati, usando contesto PDC/sezione.
    """
    desc = row.get("descrizione","")
    cod = row.get("codice","")
    imp = row.get("importo_originale", 0.0)
    sez = row.get("sezione","")

    cand_only = [c["codice"] for c in cands[:12]]
    safe = []
    for c in cands[:12]:
        safe.append({
            "codice": c["codice"],
            "score": round(float(c["score"]), 4),
            "tipo": c.get("tipo","SP"),
            "bucket": c.get("bucket", None),
            "descrizione": (c.get("descrizione","") or "")[:90],
            "kw": round(float(c.get("kw",0.0)),3),
            "sem": round(float(c.get("sem",0.0)),3),
            "bm25": round(float(c.get("bm25",0.0)),3),
        })

    vincolo = macro.get("vincolo","SOFT")
    macro_tipo = macro.get("tipo",None)
    macro_bucket = macro.get("bucket",None)
    pdc_text = macro.get("pdc_text","")

    prompt = f"""
Sei un Revisore Contabile.
Devi selezionare il codice CORRETTO esclusivamente tra i candidati forniti.
Non inventare codici.

RIGA: "{desc}" (Cod PDC: {cod}) | Importo: {imp} | Sezione PDF: {sez}
CONTESTO PDC: {pdc_text}

VINCOLO MACRO: {vincolo}
- tipo atteso: {macro_tipo}
- bucket atteso: {macro_bucket}

CANDIDATI (scegli uno di questi): {json.dumps(cand_only)}
DETTAGLI CANDIDATI: {json.dumps(safe, ensure_ascii=False)}

REGOLE:
1) Se VINCOLO MACRO è FORTE, NON scegliere candidati di tipo/bucket incoerenti (salvo che non esistano alternative sensate).
2) Se la descrizione contiene "F.do amm" o "Fondo amm" in Stato Patrimoniale, è una rettifica dell'attivo: NON selezionare voci di ammortamento del Conto Economico.
3) Se incerto, scegli il candidato con score maggiore.

JSON OUTPUT:
{{"codice_scelto":"<uno dei candidati>", "ragionamento":"max 2 frasi"}}
""".strip()

    try:
        res = client.chat.completions.create(
            model=AI_MODEL_REASON,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            response_format={"type":"json_object"}
        )
        d = json.loads(res.choices[0].message.content)
        code = (d.get("codice_scelto") or "").strip()
        reason = (d.get("ragionamento") or "").strip()
        if code not in cand_only:
            return "", "AI ha restituito codice non in lista (fallback)"
        return code, reason
    except Exception as e:
        return "", f"AI error: {e}"

# =========================================================
# ESTRAZIONE PDF (pdfplumber) — scalare o sezioni contrapposte
# =========================================================
def cluster_words_by_row(words: List[dict], tol: float = ROW_CLUSTER_TOL) -> List[List[dict]]:
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: List[List[dict]] = []
    cur = [words[0]]
    last_top = float(words[0]["top"])
    for w in words[1:]:
        if abs(float(w["top"]) - last_top) <= tol:
            cur.append(w)
        else:
            rows.append(cur)
            cur = [w]
            last_top = float(w["top"])
    rows.append(cur)
    return rows

def row_to_text(row_words: List[dict]) -> str:
    row_words = sorted(row_words, key=lambda w: w["x0"])
    return normalize_desc(" ".join([w.get("text","") for w in row_words]).strip())

def compute_split_x(page) -> float:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words:
        return float(page.width) * SPLIT_FALLBACK_RATIO

    w = float(page.width)
    amount_x0 = []
    code_x0 = []
    for wd in words:
        tx = wd.get("text","")
        if re.fullmatch(IMPORTO_RE, tx):
            amount_x0.append(float(wd["x0"]))
        if re.fullmatch(r"\d[\dA-Za-z\.\-]*", tx) and len(tx) >= 3:
            code_x0.append(float(wd["x0"]))

    if not amount_x0 or not code_x0:
        return w * SPLIT_FALLBACK_RATIO

    left_amts = [x for x in amount_x0 if x < w*0.72]
    if not left_amts:
        return w * SPLIT_FALLBACK_RATIO

    max_left_amt = max(left_amts)
    right_codes = [x for x in code_x0 if x > max_left_amt + 5]
    if not right_codes:
        return min(w*0.78, max_left_amt + 25)

    min_right_code = min(right_codes)
    split = (max_left_amt + min_right_code) / 2.0
    return max(w*0.35, min(w*0.82, split))

def parse_account_text(text: str) -> Optional[Tuple[str,str,float]]:
    """
    Più robusta: prende l'ULTIMO importo a fine stringa (anche se ci sono numeri prima).
    """
    t = (text or "").strip()
    if not t:
        return None
    m_amt = re.search(rf"({IMPORTO_RE})\s*$", t)
    if not m_amt:
        return None
    amt_str = m_amt.group(1)
    amt = parse_importo(amt_str)
    head = t[:m_amt.start()].strip()
    m = re.match(r"^(\d[\dA-Za-z\.\-]*)\s+(.+)$", head)
    if not m:
        return None
    code = m.group(1).strip()
    desc = normalize_desc(m.group(2).strip())
    return code, desc, amt

def update_ctx_from_headers(left: str, right: str, ctx: dict) -> None:
    up = (left + " " + right).upper()
    if "STATO PATRIMONIALE" in up:
        ctx["mode"] = "SP"; ctx["left_label"] = "ATTIVO"; ctx["right_label"] = "PASSIVO"
    if "CONTO ECONOMICO" in up:
        ctx["mode"] = "CE"; ctx["left_label"] = "COSTI"; ctx["right_label"] = "RICAVI"
    if ("ATTIVITA" in left.upper()) or ("ATTIVO" in left.upper()):
        ctx["mode"] = ctx.get("mode") or "SP"; ctx["left_label"] = "ATTIVO"
    if ("PASSIVITA" in right.upper()) or ("PASSIVO" in right.upper()):
        ctx["mode"] = ctx.get("mode") or "SP"; ctx["right_label"] = "PASSIVO"
    if ("COSTI" in left.upper()) and not re.match(r"^\d", left.strip()):
        ctx["mode"] = ctx.get("mode") or "CE"; ctx["left_label"] = "COSTI"
    if ("RICAVI" in right.upper()) and not re.match(r"^\d", right.strip()):
        ctx["mode"] = ctx.get("mode") or "CE"; ctx["right_label"] = "RICAVI"

def scan_risultato_in_testo(page_text: str) -> Optional[Tuple[float,str]]:
    if not page_text:
        return None
    lines = [normalize_desc(x) for x in page_text.split("\n") if normalize_desc(x)]
    best = None
    best_score = -1
    for ln in lines:
        sc = 0
        for p in RIS_PATTERNS:
            if p.search(ln):
                sc += 3
        if sc == 0:
            continue
        nums = re.findall(IMPORTO_RE, ln)
        if not nums:
            continue
        val = abs(parse_importo(nums[-1]))
        if val < 1e-9:
            continue
        low = ln.lower()
        if "perdita" in low:
            val = -val; sc += 1
        elif "utile" in low:
            val = val; sc += 1
        if sc > best_score:
            best_score = sc
            best = (val, ln)
    return best

def estrai_righe_pdfplumber(pdf_bytes: bytes) -> Tuple[List[dict], dict]:
    out: List[dict] = []
    dbg = {"pagine": 0, "righe": 0, "split_x": [], "righe_misc": 0}
    ctx = {"mode": None, "left_label": None, "right_label": None}

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        dbg["pagine"] = len(pdf.pages)
        for pidx, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            up = page_text.upper()

            # page-level hints (reset if found)
            if "STATO PATRIMONIALE" in up:
                ctx["mode"]="SP"; ctx["left_label"]="ATTIVO"; ctx["right_label"]="PASSIVO"
            if "CONTO ECONOMICO" in up:
                ctx["mode"]="CE"; ctx["left_label"]="COSTI"; ctx["right_label"]="RICAVI"
            if ("ATTIVITA" in up or "ATTIVO" in up) and ("PASSIVITA" in up or "PASSIVO" in up) and ctx.get("mode") is None:
                ctx["mode"]="SP"; ctx["left_label"]="ATTIVO"; ctx["right_label"]="PASSIVO"
            if ("COSTI" in up) and ("RICAVI" in up) and ctx.get("mode") is None:
                ctx["mode"]="CE"; ctx["left_label"]="COSTI"; ctx["right_label"]="RICAVI"

            # scan risultato d'esercizio in page text
            res = scan_risultato_in_testo(page_text)
            if res:
                val, ln = res
                out.append({
                    "codice": "",
                    "descrizione": ln,
                    "importo_originale": val,
                    "sezione": ctx.get("mode") or "UNKNOWN",
                    "pagina": pidx+1,
                    "col": "F",
                    "raw": ln,
                    "source": "plumber",
                    "is_raggruppamento": True,
                    "is_misc": True,
                })
                dbg["righe_misc"] += 1

            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                continue

            split_x = compute_split_x(page)
            dbg["split_x"].append(round(split_x, 2))

            rows = cluster_words_by_row(words, tol=ROW_CLUSTER_TOL)
            last_left_idx = None
            last_right_idx = None
            last_full_idx = None

            for row_words in rows:
                full_text = row_to_text(row_words)
                if not full_text:
                    continue
                if PAT_FOOTER.search(full_text):
                    continue

                left_words = [w for w in row_words if float(w["x0"]) < split_x]
                right_words = [w for w in row_words if float(w["x0"]) >= split_x]
                left_text = row_to_text(left_words) if left_words else ""
                right_text = row_to_text(right_words) if right_words else ""

                pl = parse_account_text(left_text) if left_text else None
                pr = parse_account_text(right_text) if right_text else None

                # header/context update
                if not (pl or pr):
                    if (left_text and is_probably_header(left_text)) or (right_text and is_probably_header(right_text)):
                        update_ctx_from_headers(left_text, right_text, ctx)

                # continuation lines (no amount, no code)
                if left_text and (pl is None) and not re.search(IMPORTO_RE, left_text) and last_left_idx is not None and not re.match(r"^\d", left_text.strip()):
                    out[last_left_idx]["descrizione"] = normalize_desc(out[last_left_idx]["descrizione"] + " " + left_text)
                if right_text and (pr is None) and not re.search(IMPORTO_RE, right_text) and last_right_idx is not None and not re.match(r"^\d", right_text.strip()):
                    out[last_right_idx]["descrizione"] = normalize_desc(out[last_right_idx]["descrizione"] + " " + right_text)

                # scalare: se non parse left/right ma parse full
                if not pl and not pr:
                    pf = parse_account_text(full_text)
                    if pf:
                        code, desc, amt = pf
                        out.append({
                            "codice": code,
                            "descrizione": desc,
                            "importo_originale": amt,
                            "sezione": ctx.get("mode") or "UNKNOWN",
                            "pagina": pidx+1,
                            "col": "F",
                            "raw": full_text,
                            "source": "plumber",
                            "is_raggruppamento": False
                        })
                        last_full_idx = len(out)-1
                        continue

                if pl:
                    code, desc, amt = pl
                    out.append({
                        "codice": code,
                        "descrizione": desc,
                        "importo_originale": amt,
                        "sezione": ctx.get("left_label") or ctx.get("mode") or "UNKNOWN",
                        "pagina": pidx+1,
                        "col": "L",
                        "raw": left_text,
                        "source": "plumber",
                        "is_raggruppamento": False
                    })
                    last_left_idx = len(out)-1

                if pr:
                    code, desc, amt = pr
                    out.append({
                        "codice": code,
                        "descrizione": desc,
                        "importo_originale": amt,
                        "sezione": ctx.get("right_label") or ctx.get("mode") or "UNKNOWN",
                        "pagina": pidx+1,
                        "col": "R",
                        "raw": right_text,
                        "source": "plumber",
                        "is_raggruppamento": False
                    })
                    last_right_idx = len(out)-1

    dbg["righe"] = len(out)
    return out, dbg

# =========================================================
# ESTRAZIONE CAMEL0T (opzionale)
# =========================================================
def estrai_righe_camelot(pdf_bytes: bytes) -> Tuple[List[dict], dict]:
    import tempfile
    if not CAMELOT_OK:
        return [], {"enabled": False, "error": "camelot non disponibile"}

    rows: List[dict] = []
    dbg = {"enabled": True, "tables": 0, "rows": 0, "flavor": "lattice"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
        dbg["tables"] = len(tables)
        for t in tables:
            df = t.df.copy()
            df = df.replace("", None).dropna(how="all")
            if df.empty:
                continue
            for _, r in df.iterrows():
                line = " ".join([str(x) for x in r.values if x is not None]).strip()
                line = normalize_desc(line)
                if not line or len(line) < 3:
                    continue
                nums = re.findall(IMPORTO_RE, line)
                if not nums:
                    continue
                imp = parse_importo(nums[-1])

                m = re.match(r"^(\d[\dA-Za-z\.\-]*)\s+(.+)$", re.sub(rf"\s+{IMPORTO_RE}\s*$","",line))
                if m:
                    codice_raw = m.group(1).strip()
                    descr = normalize_desc(m.group(2).strip())
                else:
                    codice_raw = ""
                    descr = line

                rows.append({
                    "codice": codice_raw,
                    "descrizione": descr,
                    "importo_originale": imp,
                    "sezione": "UNKNOWN",
                    "pagina": None,
                    "col": "F",
                    "raw": line,
                    "source": "camelot",
                    "is_raggruppamento": e_raggruppamento({"codice":codice_raw,"descrizione":descr})
                })
        dbg["rows"] = len(rows)
        return rows, dbg
    except Exception as e:
        return [], {"enabled": True, "error": str(e)}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# =========================================================
# ESTRAZIONE VISION (fallback per PDF scansionati / layout ostici)
# =========================================================
def crop_page_to_base64(page: fitz.Page) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(VISION_DPI_SCALE, VISION_DPI_SCALE))
    return base64.b64encode(pix.tobytes("jpg", jpg_quality=80)).decode("utf-8")

async def retry_async(fn, attempts: int = 4, base: float = 0.7):
    last = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:
            last = e
            sleep = base * (2 ** i) + random.uniform(0, 0.25)
            await asyncio.sleep(sleep)
    raise last

async def vision_page_extract(aclient, img_b64: str, sem: asyncio.Semaphore) -> List[dict]:
    if not INSTRUCTOR_OK:
        return []

    prompt = """
Analizza una pagina di bilancio (stato patrimoniale e/o conto economico).
Il bilancio può essere:
- SCALARE (una colonna)
- A SEZIONI CONTRAPPOSTE (due colonne: sinistra/destra)

OBIETTIVO:
Estrarre SOLO righe contabili (conti) con importo. Se una riga contiene DUE conti (sx+dx), devi restituire DUE righe separate.

OUTPUT per ogni riga:
- codice: se presente, altrimenti ""
- descrizione: testo del conto (senza importi)
- sezione: scegli una tra ATTIVO, PASSIVO, COSTI, RICAVI, SP, CE, UNKNOWN
- importo: numero (con segno negativo se indicato con '-' o parentesi)
- is_raggruppamento: True se è titolo/subtotale/totale (es. "TOTALE", "IMMOBILIZZAZIONI", "COSTI DELLA PRODUZIONE")

REGOLE:
- NON inventare righe.
- NON unire conti diversi.
- Se vedi "Utile/Perdita d'esercizio" estraila (anche senza codice) come riga con importo e is_raggruppamento=True.
""".strip()

    async with sem:
        async def call():
            return await aclient.chat.completions.create(
                model=AI_MODEL_VISION,
                response_model=VisionEstratto,
                max_tokens=2200,
                messages=[{
                    "role":"user",
                    "content":[
                        {"type":"text","text":prompt},
                        {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }]
            )

        try:
            resp = await retry_async(call)
            return [r.model_dump() for r in resp.righe]  # type: ignore
        except Exception:
            return []

async def estrai_tutto_vision(pdf_bytes: bytes) -> Tuple[List[dict], dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sem = asyncio.Semaphore(VISION_CONCURRENCY)
    aclient = instructor.from_openai(AsyncOpenAI(api_key=api_key))  # type: ignore

    tasks = []
    prog = st.progress(0)
    status = st.empty()

    for i, p in enumerate(doc):
        status.text(f"Vision: pagina {i+1}/{len(doc)}...")
        img_b64 = crop_page_to_base64(p)
        tasks.append(vision_page_extract(aclient, img_b64, sem))
        prog.progress((i+1)/len(doc))

    results = await asyncio.gather(*tasks)
    flat = [x for sub in results for x in sub]

    dbg = {"pagine": len(doc), "righe": len(flat), "concurrency": VISION_CONCURRENCY}
    return flat, dbg

# =========================================================
# NORMALIZZAZIONE + DEDUP + MOVIMENTABILI
# =========================================================
def normalizza_righe(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        desc = normalize_desc(str(r.get("descrizione","")))
        if not desc or len(desc) < 2:
            continue
        out.append({
            "codice": str(r.get("codice","")).strip(),
            "descrizione": desc,
            "importo_originale": float(parse_importo(r.get("importo_originale", r.get("importo", 0)))),
            "sezione": str(r.get("sezione","UNKNOWN")).upper().strip(),
            "pagina": r.get("pagina", None),
            "col": r.get("col", "F"),
            "raw": str(r.get("raw","")),
            "source": str(r.get("source","unknown")),
            "is_raggruppamento": bool(r.get("is_raggruppamento", False)),
            "is_misc": bool(r.get("is_misc", False)),
        })
    return out

def deduplica(rows: List[dict]) -> List[dict]:
    if not rows:
        return []
    out: List[dict] = []
    for r in rows:
        is_dup = False
        for o in out[-250:]:
            same_sec = (r.get("sezione") == o.get("sezione"))
            same_code = normalizza_codice(r.get("codice","")) and (normalizza_codice(r.get("codice","")) == normalizza_codice(o.get("codice","")))
            close_amt = abs(float(r.get("importo_originale",0.0)) - float(o.get("importo_originale",0.0))) < 0.5
            if FUZZ_OK:
                sim_desc = fuzz.token_sort_ratio(r["descrizione"].lower(), o["descrizione"].lower()) >= 94
            else:
                sim_desc = (r["descrizione"].lower() == o["descrizione"].lower())
            if same_sec and close_amt and (same_code or sim_desc):
                is_dup = True
                break
        if not is_dup:
            out.append(r)
    return out

def filter_movimentabili(rows: List[dict]) -> Tuple[List[dict], dict]:
    dbg = {"input": len(rows), "drop_raggr": 0, "drop_short": 0, "drop_group": 0}
    if not rows:
        return [], dbg

    # rimuovi raggruppamenti espliciti (ma tieni misc risultato per debug)
    tmp = []
    for r in rows:
        if r.get("is_misc"):
            tmp.append(r); continue
        if DROP_TOTALI_E_HEADER and e_raggruppamento(r):
            dbg["drop_raggr"] += 1
            continue
        tmp.append(r)

    # codici normalizzati presenti
    codes = [normalizza_codice(r.get("codice","")) for r in tmp if normalizza_codice(r.get("codice",""))]
    code_set = set(codes)

    group_set = set()
    if DROP_GROUP_PREFIX:
        for c in code_set:
            for k in range(2, len(c)):
                pref = c[:k]
                if pref in code_set:
                    group_set.add(pref)

    out = []
    for r in tmp:
        c = normalizza_codice(r.get("codice",""))
        if not c:
            # senza codice: di solito sono titoli; tieni solo se è risultato d'esercizio (misc)
            continue
        if len(c) < MIN_CODE_LEN:
            dbg["drop_short"] += 1
            continue
        if c in group_set:
            dbg["drop_group"] += 1
            continue
        out.append(r)

    dbg["output"] = len(out)
    return out, dbg

# =========================================================
# RISULTATO D'ESERCIZIO (robusto)
# =========================================================
def trova_risultato_esercizio(rows_all: List[dict]) -> Tuple[Optional[float], Optional[str]]:
    best_score = -1
    best_val = None
    best_desc = None

    for r in rows_all:
        desc = (r.get("descrizione") or "").strip()
        if not desc:
            continue
        score = 0
        for p in RIS_PATTERNS:
            if p.search(desc):
                score += 3
        if score == 0:
            continue

        val_raw = float(r.get("importo_originale") or 0.0)
        val_abs = abs(val_raw)
        if val_abs < 1e-9:
            continue

        dlow = desc.lower()
        if "perdita" in dlow:
            val = -val_abs; score += 1
        elif "utile" in dlow:
            val = val_abs; score += 1
        else:
            # fallback: usa segno già presente
            val = val_raw

        if score > best_score:
            best_score = score
            best_val = val
            best_desc = desc

    return best_val, best_desc

# =========================================================
# ANOMALIE + CONTI SOSPETTI
# =========================================================
def analizza_anomalie(df: pd.DataFrame) -> List[str]:
    anomalie = []
    for _, r in df.iterrows():
        desc = str(r.get("Descrizione", "")).lower()
        saldo = float(r.get("Saldo finale", 0.0))
        tipo = r.get("Tipo", "")

        if "cassa" in desc and saldo < 0:
            anomalie.append(f"⚠️ **Cassa negativa:** {r['Descrizione']} ({saldo:,.2f}€)")
        if "iva" in desc and "credito" in desc and saldo < 0:
            anomalie.append(f"⚠️ **IVA a credito con segno anomalo:** {r['Descrizione']} ({saldo:,.2f}€)")
        if "iva" in desc and "debito" in desc and saldo > 0:
            anomalie.append(f"⚠️ **IVA a debito con segno anomalo:** {r['Descrizione']} ({saldo:,.2f}€)")
        if "fornitor" in desc and saldo > 0:
            anomalie.append(f"❓ **Fornitori con saldo positivo:** {r['Descrizione']} ({saldo:,.2f}€)")
        if "client" in desc and saldo < 0:
            anomalie.append(f"❓ **Clienti con saldo negativo:** {r['Descrizione']} ({saldo:,.2f}€)")
        if tipo == "CE" and saldo > 0 and all(x not in desc for x in ["ricav", "provent", "plusval", "rimbor", "contribut"]):
            anomalie.append(f"❓ **CE positivo anomalo:** {r['Descrizione']} ({saldo:,.2f}€)")
    return anomalie

def conti_sospetti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evidenzia conti dove:
    - sezione del PDF (macro) e tipo/bucket classificazione non combaciano
    - confidence bassa
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def flag_row(r):
        flags=[]
        sec = str(r.get("Sezione PDF","")).upper()
        bucket = str(r.get("Bucket",""))
        tipo = str(r.get("Tipo",""))
        conf = float(r.get("Confidence",0.0))
        if sec in ("ATTIVO","PASSIVO","COSTI","RICAVI"):
            if sec != bucket:
                flags.append("Macro mismatch")
        if conf < 0.68:
            flags.append("Conf bassa")
        if "f.do amm" in str(r.get("Descrizione","")).lower() and tipo=="CE":
            flags.append("F.do amm in CE?")
        return ", ".join(flags)

    dfx = df.copy()
    dfx["Flags"] = dfx.apply(flag_row, axis=1)
    dfx = dfx[dfx["Flags"].astype(str).str.len() > 0]
    return dfx.sort_values(by=["Confidence"], ascending=True)

# =========================================================
# SIDEBAR UI
# =========================================================
banner_img = "Revilaw S.p.A..jpg"
if os.path.exists(banner_img):
    st.image(banner_img, use_container_width=True)
else:
    st.markdown("## 🧬 Revilaw AuditEv Hierarchy (Quality)")

with st.sidebar:
    st.markdown("### 🎛️ Pannello Controllo")

    st.markdown("<div class='ctl-box'><div class='ctl-title'>1) Bilancio (PDF)</div><div class='ctl-sub'>Carica il PDF da analizzare</div>", unsafe_allow_html=True)
    f_pdf = st.file_uploader("Bilancio (PDF)", type="pdf", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ctl-box'><div class='ctl-title'>2) Tipo bilancio</div><div class='ctl-sub'>Seleziona il modello ordinario/abbreviato</div>", unsafe_allow_html=True)
    bilancio_tipo = st.selectbox("Tipo bilancio", ["Bilancio ordinario","Bilancio abbreviato"], label_visibility="collapsed")
    cls_path = pick_existing(CLASSIFICATION_CANDIDATES.get(bilancio_tipo, []))
    st.caption(f"📌 Classificazione: {cls_path.name if cls_path else 'NON TROVATA'}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='ctl-box'><div class='ctl-title'>3) Piano dei conti (opzionale)</div><div class='ctl-sub'>PDF / XLSX / CSV. Se manca, uso il PDC ricavato dal bilancio.</div>", unsafe_allow_html=True)
    f_pdc = st.file_uploader("Piano dei conti", type=["pdf","xlsx","csv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    ready = (f_pdf is not None) and (cls_path is not None) and (api_key is not None) and (len(api_key or "") > 10)
    start = st.button("🚀 AVVIA ELABORAZIONE", disabled=not ready, type="primary")
    if not api_key:
        st.caption("🔴 Configura OPENAI_API_KEY (st.secrets o variabile d'ambiente).")
    if f_pdf is None:
        st.caption("🔴 Carica un PDF.")
    if cls_path is None:
        st.caption("🔴 Modello classificazione non trovato nella cartella.")

# =========================================================
# PIPELINE
# =========================================================
if start:
    st.session_state["data_processed"] = None
    st.session_state["logs"] = []
    st.session_state["raw_rows"] = []
    st.session_state["estrazione_metodo"] = "N/D"
    st.session_state["debug_counts"] = {}
    st.session_state["risultato_pdf"] = None
    st.session_state["risultato_pdf_desc"] = None
    st.session_state["warnings"] = []

    main = st.empty()
    main.info("⚙️ Inizializzazione...")

    try:
        # --- Load classificazione ---
        df_c = pd.read_excel(cls_path, dtype=str)
        db = inizializza_motori_ricerca(df_c)
        if db is None:
            st.error("Errore: non riesco a inizializzare la classificazione.")
            st.stop()

        st.session_state["combo_options"] = db["combo"]
        st.session_state["map_code_desc"] = db["map_desc"]
        st.session_state["map_code_natura"] = db["map_natura"]
        st.session_state["map_code_tipo"] = db["map_tipo"]
        st.session_state["map_code_bucket"] = db["map_bucket"]

        log("CLASSIFICAZIONE", f"{bilancio_tipo}: indicizzate {len(db['codici'])} righe. SEM={db['has_sem']} BM25={db['has_bm25']}")

        # --- Read PDF bytes ---
        f_pdf.seek(0)
        pdf_bytes = f_pdf.read()

        # --- Extraction strategy ---
        rows_all: List[dict] = []
        dbg_extract: Dict[str, Any] = {}

        # Try camelot first (if available)
        if CAMELOT_OK and not VISION_FORCE:
            main.info("📄 Tentativo estrazione tabelle (Camelot)...")
            cam_rows, cam_dbg = estrai_righe_camelot(pdf_bytes)
            if cam_rows and len(cam_rows) >= 40:
                rows_all = cam_rows
                dbg_extract = cam_dbg
                st.session_state["estrazione_metodo"] = "CAMELOT"
                log("CAMELOT", f"OK rows={len(cam_rows)} tables={cam_dbg.get('tables')}")
            else:
                log("CAMELOT", f"Fallback (rows={len(cam_rows)})", level="WARN")

        # pdfplumber extraction
        if not rows_all and not VISION_FORCE:
            main.info("📄 Estrazione testo strutturata (pdfplumber)...")
            pl_rows, pl_dbg = estrai_righe_pdfplumber(pdf_bytes)
            rows_all = pl_rows
            dbg_extract = pl_dbg
            st.session_state["estrazione_metodo"] = "PDFPLUMBER"
            log("PDFPLUMBER", f"rows={len(pl_rows)} misc={pl_dbg.get('righe_misc')}")

        # Vision fallback
        if (VISION_FORCE or (not rows_all) or (len(rows_all) < 25)) and api_key and INSTRUCTOR_OK:
            main.info("🤖 Fallback Vision (tutte le pagine)...")
            vis_rows, vis_dbg = run_async(estrai_tutto_vision(pdf_bytes))
            # normalizza output Vision a schema rows
            norm_vis = []
            for r in vis_rows:
                norm_vis.append({
                    "codice": r.get("codice",""),
                    "descrizione": r.get("descrizione",""),
                    "importo_originale": r.get("importo", r.get("importo_originale",0)),
                    "sezione": r.get("sezione","UNKNOWN"),
                    "pagina": None,
                    "col": "F",
                    "raw": "",
                    "source": "vision",
                    "is_raggruppamento": bool(r.get("is_raggruppamento", False)),
                    "is_misc": False
                })
            if norm_vis:
                rows_all = norm_vis
                dbg_extract = vis_dbg
                st.session_state["estrazione_metodo"] = "VISION"
                log("VISION", f"rows={len(norm_vis)}")
            else:
                log("VISION", "0 righe", level="ERROR")

        if not rows_all:
            st.error("❌ Nessuna riga utile estratta dal PDF (Camelot/pdfplumber/Vision). Controlla 'Registro debug'.")
            st.session_state["debug_counts"] = {"righe_grezze": 0, "norma_grezza": 0, "unico": 0}
            st.stop()

        # Normalize + dedup
        norm = normalizza_righe(rows_all)
        unico = deduplica(norm)

        # Build PDC map: bilancio-derived + optional external
        pdc_map = build_pdc_from_rows(unico)
        if f_pdc:
            main.info("🧬 Indicizzazione PDC caricato...")
            pdc_user = analizza_pdc_universale(f_pdc)
            # merge (prefer user)
            pdc_map.update(pdc_user)
            log("PDC", f"User PDC voci={len(pdc_user)}")
        log("PDC", f"PDC totale (bilancio+user) voci={len(pdc_map)}")

        # risultato d'esercizio (dal RAW incluso misc)
        ris_val, ris_desc = trova_risultato_esercizio(norm)
        st.session_state["risultato_pdf"] = ris_val
        st.session_state["risultato_pdf_desc"] = ris_desc

        # Filtra movimentabili
        mov, dbg_mov = filter_movimentabili(unico)

        st.session_state["raw_rows"] = unico

        st.session_state["debug_counts"] = {
            "righe_grezze": len(rows_all),
            "norma_grezza": len(norm),
            "unico": len(unico),
            "movimentabili": len(mov),
            "estrazione_metodo": st.session_state["estrazione_metodo"],
            "dbg_extract": dbg_extract,
            "dbg_mov": dbg_mov
        }

        if not mov:
            st.error("❌ Nessun conto movimentabile dopo filtri (raggruppamenti/prefissi).")
            st.stop()

        # --- Classificazione ---
        main.info(f"🧠 Classificazione su {len(mov)} conti movimentabili...")
        raw_client = OpenAI(api_key=api_key)

        final_rows = []
        pbar = st.progress(0)
        for i, r in enumerate(mov):
            pbar.progress((i+1)/len(mov))

            cod_raw = str(r.get("codice","")).strip()
            desc = str(r.get("descrizione","")).strip()
            sec = str(r.get("sezione","UNKNOWN")).upper()

            # macro decision (combine pdf/pdc/code/desc)
            tipo_m, bucket_m, conf_m, evidence = decide_macro_for_row(r, pdc_map)
            allowed, strength = allowed_indices_for_macro(db, tipo_m, bucket_m, conf_m)

            # PDC text (gerarchia)
            cod_padre, desc_padre, cut = trova_padre_gerarchico(cod_raw, pdc_map)
            pdc_text = desc_padre or ""
            # PDC macro (for bonus)
            t_pdc, b_pdc, c_pdc = infer_macro_da_testo_pdc(pdc_text)

            # Query: descrizione + PDC (prima) + fallback descrizione
            query = desc
            if pdc_text:
                query = f"{desc} | {pdc_text}"

            base = ricerca_ibrida(query, db, allowed)
            ranked = rerank_candidates(query, base, db, tipo_m, bucket_m, strength, b_pdc, c_pdc)

            # decide
            if decisione_senza_ai(ranked):
                code_sel = ranked[0]["codice"]
                reason = "Scelta deterministica: score alto"
            else:
                macro_pack = {
                    "tipo": tipo_m,
                    "bucket": bucket_m,
                    "conf": conf_m,
                    "vincolo": "FORTE" if conf_m >= 90 else ("MEDIO" if conf_m >= 80 else "SOFT"),
                    "pdc_text": pdc_text
                }
                code_sel, reason = ragionatore_ai(raw_client, r, macro_pack, ranked, db)

            if not code_sel:
                code_sel, fb_reason = scegli_fallback(ranked, db)
                reason = (reason + " | " if reason else "") + fb_reason

            desc_audit = db["map_desc"].get(code_sel, "???")
            natura_classif = db["map_natura"].get(code_sel, "")
            tipo_classif = db["map_tipo"].get(code_sel, "SP")
            bucket_classif = db["map_bucket"].get(code_sel, None)

            # saldo finale: usa importo_originale (segno) ma rendilo coerente con bucket se necessario
            imp_raw = float(r.get("importo_originale",0.0))
            imp_abs = abs(imp_raw)

            # Natura attesa (da classificazione) + Natura calcolata (considerando il segno in PDF)
            nat_attesa = expected_natura_da_bucket(bucket_classif) or "DARE"
            nat_calc = nat_attesa
            if imp_raw < 0:
                nat_calc = "AVERE" if nat_attesa == "DARE" else "DARE"

            saldo_dare = imp_abs if nat_calc == "DARE" else 0.0
            saldo_avere = imp_abs if nat_calc == "AVERE" else 0.0

            # saldo finale: CE = avere-dare, SP = dare-avere
            sf = (saldo_avere - saldo_dare) if tipo_classif == "CE" else (saldo_dare - saldo_avere)

            final_rows.append({
                "Codice": cod_raw,
                "Descrizione": desc,
                "Importo (PDF)": imp_raw,
                "Sezione PDF": sec,
                "Classificazione_Combo": f"{code_sel} | {desc_audit}",
                "Codice Classificazione": code_sel,
                "Natura Classificazione": natura_classif,
                "Tipo": tipo_classif,
                "Bucket": bucket_classif,
                "Natura attesa": nat_attesa,
                "Natura calcolata": nat_calc,
                "Saldo dare": saldo_dare,
                "Saldo avere": saldo_avere,
                "Saldo finale": sf,
                "Inverti Segno": False,
                "Gerarchia PDC": pdc_text[:120] if pdc_text else "N/A",
                "Ragionamento AI": reason,
                "Confidence": float(ranked[0]["score"] if ranked else 0.0),
            })

        df_final = pd.DataFrame(final_rows)
        df_final.sort_values(by=["Codice"], inplace=True)
        st.session_state["data_processed"] = df_final
        main.empty()
        st.rerun()

    except Exception as e:
        st.error(f"Errore Critico: {e}")
        log("ERRORE", str(e), level="ERROR")

# =========================================================
# VISUALIZZAZIONE RISULTATI
# =========================================================
if st.session_state["data_processed"] is not None:
    df = st.session_state["data_processed"]
    combo_options = st.session_state.get("combo_options", [])

    st.markdown("### 📊 Dashboard")

    df_sp = df[df["Tipo"] == "SP"]
    df_ce = df[df["Tipo"] == "CE"]

    tot_attivo = df_sp[df_sp["Saldo finale"] > 0]["Saldo finale"].sum()
    tot_passivo = df_sp[df_sp["Saldo finale"] < 0]["Saldo finale"].sum()
    tot_risultato = df_ce["Saldo finale"].sum()
    tot_sp_netto = df_sp["Saldo finale"].sum()
    diff = tot_sp_netto - tot_risultato

    ris_pdf = st.session_state.get("risultato_pdf", None)
    ris_desc = st.session_state.get("risultato_pdf_desc", None)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("ATTIVO", f"€ {tot_attivo:,.2f}")
    c2.metric("PASSIVO", f"€ {tot_passivo:,.2f}", delta_color="inverse")
    c3.metric("RISULTATO (RICL.)", f"€ {tot_risultato:,.2f}")

    if abs(diff) < 1.0:
        c4.success("✅ QUADRATO")
    else:
        c4.error(f"⚠️ SBILANCIO: € {diff:,.2f}")

    if ris_pdf is not None:
        c5.metric("RISULTATO (PDF)", f"€ {ris_pdf:,.2f}", delta=(ris_desc or "")[:35])
    else:
        c5.metric("RISULTATO (PDF)", "N/D", delta="Non trovato")

    anomalie = analizza_anomalie(df)
    if anomalie:
        with st.expander(f"🚨 {len(anomalie)} Anomalie", expanded=True):
            for a in anomalie:
                st.markdown(a)

    # Conti sospetti
    susp = conti_sospetti(df)
    if not susp.empty:
        with st.expander(f"🕵️ Conti sospetti ({len(susp)})", expanded=False):
            st.dataframe(susp, use_container_width=True, hide_index=True)

    # Debug counts / registro debug
    with st.expander("🧾 Registro debug (conteggi + dettagli)", expanded=False):
        dbg = st.session_state.get("debug_counts", {})
        st.code(json.dumps(dbg, indent=2, ensure_ascii=False))

        logs = st.session_state.get("logs", [])
        if logs:
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
        else:
            st.info("Nessun log.")

    # Preview estrazione
    with st.expander("📄 Preview estrazione (prime righe)", expanded=False):
        raw = st.session_state.get("raw_rows", [])
        if raw:
            st.dataframe(pd.DataFrame(raw).head(30), use_container_width=True, hide_index=True)
        else:
            st.info("Nessuna riga raw salvata.")

    tab_edit, tab_exp = st.tabs(["📝 Revisione", "💾 Export"])

    # ----------------------------
    # TAB EDIT
    # ----------------------------
    with tab_edit:
        # colonne essenziali + natura
        df_v = df[[
            "Codice","Descrizione","Sezione PDF","Importo (PDF)",
            "Saldo finale","Natura Classificazione","Natura attesa","Natura calcolata",
            "Inverti Segno","Classificazione_Combo"
        ]].copy()

        edited = st.data_editor(
            df_v,
            column_order=["Codice","Descrizione","Sezione PDF","Importo (PDF)","Saldo finale","Natura Classificazione","Natura attesa","Inverti Segno","Classificazione_Combo"],
            column_config={
                "Classificazione_Combo": st.column_config.SelectboxColumn("Classificazione", options=combo_options, width="large"),
                "Saldo finale": st.column_config.NumberColumn("Saldo finale", format="%.2f €", disabled=True),
                "Importo (PDF)": st.column_config.NumberColumn("Importo (PDF)", format="%.2f €", disabled=True),
                "Inverti Segno": st.column_config.CheckboxColumn("Inverti Segno"),
                "Codice": st.column_config.TextColumn("Codice conto", disabled=True),
                "Descrizione": st.column_config.TextColumn("Descrizione", disabled=True),
                "Sezione PDF": st.column_config.TextColumn("Sezione PDF", disabled=True),
                "Natura Classificazione": st.column_config.TextColumn("Natura (A/P/C/R)", disabled=True),
                "Natura attesa": st.column_config.TextColumn("Natura attesa (DARE/AVERE)", disabled=True),
            },
            height=650,
            use_container_width=True,
            hide_index=True
        )

        # applica modifiche
        upd = False
        map_tipo = st.session_state.get("map_code_tipo", {})
        map_bucket = st.session_state.get("map_code_bucket", {})
        map_nat = st.session_state.get("map_code_natura", {})

        for i, row in edited.iterrows():
            orig = df.loc[i]
            if (row["Classificazione_Combo"] != orig["Classificazione_Combo"]) or (bool(row["Inverti Segno"]) != bool(orig["Inverti Segno"])):
                upd = True

                combo = str(row["Classificazione_Combo"])
                code = combo.split("|")[0].strip() if "|" in combo else combo.strip()

                tipo = map_tipo.get(code, "SP")
                bucket = map_bucket.get(code, None)
                nat_class = map_nat.get(code, "")

                nat_attesa = expected_natura_da_bucket(bucket) or "DARE"

                imp_raw = float(orig["Importo (PDF)"])
                imp_abs = abs(imp_raw)

                nat_calc = nat_attesa
                if imp_raw < 0:
                    nat_calc = "AVERE" if nat_calc == "DARE" else "DARE"
                if bool(row["Inverti Segno"]):
                    nat_calc = "AVERE" if nat_calc == "DARE" else "DARE"

                sd = imp_abs if nat_calc == "DARE" else 0.0
                sa = imp_abs if nat_calc == "AVERE" else 0.0
                sf = (sa - sd) if tipo == "CE" else (sd - sa)

                df.at[i, "Classificazione_Combo"] = combo
                df.at[i, "Codice Classificazione"] = code
                df.at[i, "Tipo"] = tipo
                df.at[i, "Bucket"] = bucket
                df.at[i, "Natura Classificazione"] = nat_class
                df.at[i, "Natura attesa"] = nat_attesa
                df.at[i, "Natura calcolata"] = nat_calc
                df.at[i, "Saldo dare"] = sd
                df.at[i, "Saldo avere"] = sa
                df.at[i, "Saldo finale"] = sf
                df.at[i, "Inverti Segno"] = bool(row["Inverti Segno"])

        if upd:
            st.session_state["data_processed"] = df
            st.rerun()

    # ----------------------------
    # TAB EXPORT
    # ----------------------------
    with tab_exp:
        out = BytesIO()
        try:
            with pd.ExcelWriter(out, engine="xlsxwriter") as w:
                dx = df.copy()
                dx["Classificazione"] = dx["Classificazione_Combo"].apply(lambda x: x.split("|")[0].strip() if "|" in str(x) else str(x))

                cols = [
                    "Codice", "Descrizione",
                    "Sezione PDF","Importo (PDF)",
                    "Saldo dare", "Saldo avere", "Saldo finale",
                    "Classificazione", "Natura Classificazione","Natura attesa","Natura calcolata",
                    "Tipo", "Bucket", "Gerarchia PDC",
                    "Confidence", "Ragionamento AI"
                ]
                dx2 = dx[cols]
                dx2.to_excel(w, index=False, sheet_name="Bilancio")

                wb = w.book
                ws = w.sheets["Bilancio"]

                fmt_head = wb.add_format({"bold": True, "bg_color": "#003366", "font_color": "white", "border": 1})
                fmt_curr = wb.add_format({"num_format": "#,##0.00 €"})
                fmt_neg = wb.add_format({"num_format": "#,##0.00 €", "font_color": "red"})

                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, len(dx2), len(cols) - 1)

                for c, v in enumerate(cols):
                    ws.write(0, c, v, fmt_head)

                # format currency
                money_cols = [3,4,5,6]  # Importo, dare, avere, finale
                for r in range(1, len(dx2) + 1):
                    for c in money_cols:
                        val = dx2.iloc[r-1, c]
                        try:
                            fval = float(val)
                        except Exception:
                            fval = 0.0
                        ws.write_number(r, c, fval, fmt_neg if fval < 0 else fmt_curr)

                ws.set_column("A:A", 14)
                ws.set_column("B:B", 55)
                ws.set_column("C:C", 12)
                ws.set_column("D:G", 18)
                ws.set_column("H:H", 16)
                ws.set_column("I:J", 14)
                ws.set_column("K:L", 10)
                ws.set_column("M:M", 30)
                ws.set_column("N:N", 10)
                ws.set_column("O:O", 60)

            out.seek(0)
            st.download_button("💾 SCARICA EXCEL", data=out, file_name="Revilaw_Bilancio_QUALITY.xlsx", type="primary")
        except Exception as e:
            st.error(f"Errore export Excel: {e}")

else:
    st.markdown("<div style='text-align: center; color: gray; margin-top: 50px;'><h3>👋 Carica i file per iniziare</h3></div>", unsafe_allow_html=True)
