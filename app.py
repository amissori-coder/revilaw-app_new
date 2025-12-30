import streamlit as st
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from io import BytesIO
import json
import base64
from openai import OpenAI, AsyncOpenAI
import os
import re
import time
import string
import asyncio
import random
from pathlib import Path

# LIBRERIE AVANZATE
import instructor
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch
from rapidfuzz import fuzz

# OPTIONAL: Camelot per PDF nativi
try:
    import camelot  # type: ignore
    CAMEL0T_OK = True
except Exception:
    CAMEL0T_OK = False

# ---------------------------------------------------------
# CONFIGURAZIONE GENERALE & STILE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Revilaw AuditEv Hierarchy (Quality)",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Stile Dashboard */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.1);
    }
    div.stButton > button:first-child {
        background-color: #003366;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        width: 100%;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #004080;
        color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }
    h1, h2, h3 {
        color: #003366;
        font-family: 'Segoe UI', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

SOFTWARE_PROMPTS = {
    "Generico / Non so": "Tabella conti standard.",
    "Zucchetti (Ago/AdHoc/Mago)": "Layout Zucchetti. Codici tipo '2405001'.",
    "TeamSystem (Alyante/Gamma)": "Layout TeamSystem. Codici tipo '10.20.30'.",
    "Datev Koinos": "Layout Datev.",
    "Sistemi (Profis)": "Layout Profis. Codici tipo '1234.5678'."
}

# ---------------------------------------------------------
# CLASSIFICAZIONE: scegli tipo bilancio e carica file locale
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CLASSIFICATION_FILES = {
    "Bilancio ordinario": BASE_DIR / "Modello classificazione Ordinario_pulito per test.xlsx",
    "Bilancio abbreviato": BASE_DIR / "Modello classificazione Abbreviato_pulito per test.xlsx",
}

# =========================================================
# API KEY (SICURA + NO CRASH SE SECRETS MANCANO)
# =========================================================
def get_api_key() -> Optional[str]:
    try:
        # st.secrets può alzare StreamlitSecretNotFoundError se non esiste il file
        return st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        return None

api_key = get_api_key() or os.getenv("OPENAI_API_KEY")

# =========================================================
# MODELLI DATI (per instructor)
# =========================================================
class RigaBilancio(BaseModel):
    codice: str = Field(..., description="Codice numerico del conto (può essere vuoto).")
    descrizione: str = Field(..., description="Descrizione analitica del conto")
    natura: Literal["DARE", "AVERE"]
    importo: float
    is_raggruppamento: bool = Field(..., description="True se titolo/subtotale/totale.")

class BilancioEstratto(BaseModel):
    righe: List[RigaBilancio]

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
        "raw_pdf_rows": [],
        "camelot_rows": [],
        "camelot_debug": {},
        "estrazione_metodo": "N/D",
        "risultato_pdf": None,
        "risultato_pdf_desc": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ss_init()

def log(step: str, msg: str, level: str = "INFO") -> None:
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
    """
    Esegue un coroutine in modo safe anche dentro Streamlit.
    """
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
# EMBEDDINGS
# =========================================================
@st.cache_resource
def carica_modello_semantico():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

semantic_model = carica_modello_semantico()

# =========================================================
# UTILS
# =========================================================
def simple_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    return text.translate(str.maketrans("", "", string.punctuation)).split()

def normalizza_codice(codice) -> str:
    if not codice:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(codice))

def parse_importo(x) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    s = s.replace("€", "").replace(" ", "")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # 1.234,56 -> 1234.56
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

def normalize_desc(desc: str) -> str:
    return re.sub(r"\s+", " ", (desc or "").strip())

# =========================================================
# CLASSIFICAZIONE: INFERISCE TIPO + BUCKET (ATTIVO/PASSIVO/COSTI/RICAVI)
# =========================================================
def inferisci_tipo_bucket(natura_str: str) -> Tuple[str, Optional[str]]:
    s = (natura_str or "").strip().upper()

    # Caso "A/P/C/R" esplicito
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

    if bucket is None:
        if "ATTIVO" in s:
            bucket = "ATTIVO"; tipo = tipo or "SP"
        elif "PASSIVO" in s or "DEBIT" in s:
            bucket = "PASSIVO"; tipo = tipo or "SP"
        elif "RICAV" in s or "PROVENT" in s:
            bucket = "RICAVI"; tipo = tipo or "CE"
        elif "COST" in s or "ONERI" in s:
            bucket = "COSTI"; tipo = tipo or "CE"

    if tipo not in ("SP", "CE"):
        tipo = "SP"

    return tipo, bucket

@st.cache_data
def inizializza_motori_ricerca(df_classificazione: pd.DataFrame):
    """
    Costruisce BM25 + embedding + mappe descrizione/natura + indici dare/avere.
    Si aspetta che le prime 2 colonne siano: codice, descrizione.
    Cerca una colonna "natura" o "tipo".
    """
    if df_classificazione is None or df_classificazione.empty:
        return None

    df = df_classificazione.copy()
    df.columns = [c.strip() for c in df.columns]

    # Se c'è una colonna che marca "Classificabile", usa solo righe X
    if "Classificabile" in df.columns:
        df = df[df["Classificabile"].astype(str).str.lower().str.strip() == "x"].copy()

    col_cod = df.columns[0]
    col_desc = df.columns[1]
    col_natura = next((c for c in df.columns if any(x in c.lower() for x in ["natura", "tipo"])), None)

    codici: List[str] = []
    desc_sem: List[str] = []
    toks: List[List[str]] = []

    map_desc = {}
    map_natura = {}
    map_tipo = {}
    map_bucket = {}
    combo = []

    idx_dare = []
    idx_avere = []

    for idx, row in df.iterrows():
        code = str(row[col_cod]).strip()
        desc = normalize_desc(str(row[col_desc]).strip())
        nat = str(row[col_natura]).strip() if col_natura else ""

        tipo, bucket = inferisci_tipo_bucket(nat)

        codici.append(code)
        desc_sem.append(desc)
        toks.append(simple_tokenize(desc))

        map_desc[code] = desc
        map_natura[code] = nat
        map_tipo[code] = tipo
        map_bucket[code] = bucket

        combo.append(f"{code} | {desc[:60]}")

        nu = (nat or "").upper()
        if any(x in nu for x in ["ATTIVO", "SP-A", "COST", "ONERI", "CE-C", "A", "C"]):
            idx_dare.append(idx)
        elif any(x in nu for x in ["PASSIVO", "SP-P", "RICAV", "PROVENT", "CE-R", "P", "R"]):
            idx_avere.append(idx)
        else:
            # se non chiaro, permetti entrambi
            idx_dare.append(idx); idx_avere.append(idx)

    emb = semantic_model.encode(desc_sem, convert_to_tensor=True)
    bm25 = BM25Okapi(toks)

    return {
        "codici": codici,
        "desc_sem": desc_sem,
        "emb": emb,
        "bm25": bm25,
        "map_desc": map_desc,
        "map_natura": map_natura,
        "map_tipo": map_tipo,
        "map_bucket": map_bucket,
        "combo": sorted(list(set(combo))),
        "idx_d": idx_dare,
        "idx_a": idx_avere
    }

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
                match = re.search(r"^(\d[\d\.]*)\s+(.+)", line.strip())
                if match:
                    code_raw = match.group(1)
                    desc = match.group(2).strip()
                    clean_code = normalizza_codice(code_raw)
                    if len(clean_code) >= 2:
                        map_pdc[clean_code] = desc
    return map_pdc

def analizza_pdc_universale(file_obj) -> dict:
    pdc_norm = {}
    if file_obj.name.endswith(".pdf"):
        return estrai_pdc_da_pdf(file_obj)

    df = None
    try:
        if file_obj.name.endswith(".xlsx"):
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

def trova_padre_gerarchico(codice_raw: str, pdc_norm_map: dict) -> Tuple[Optional[str], Optional[str]]:
    if not codice_raw or not pdc_norm_map:
        return None, None
    clean = normalizza_codice(codice_raw)
    if clean in pdc_norm_map:
        return clean, pdc_norm_map[clean]
    while len(clean) > 1:
        clean = clean[:-1]
        if clean in pdc_norm_map:
            return clean, f"(Gruppo Padre) {pdc_norm_map[clean]}"
    return None, None

# =========================================================
# FILTRO RAGGRUPPAMENTI
# =========================================================
PATTERN_RAGGR = re.compile(
    r"^(totale|somma|subtotale|saldo|risultato|utile|perdita|"
    r"attivo|passivo|patrimonio|conto economico|stato patrimoniale)\b",
    re.IGNORECASE
)

def e_raggruppamento(r: dict) -> bool:
    if bool(r.get("is_raggruppamento", False)):
        return True

    desc = (r.get("descrizione") or "").strip()
    cod = (r.get("codice") or "").strip()

    if PATTERN_RAGGR.match(desc):
        return True

    # titoli in maiuscolo senza codice
    if cod == "" and (len(desc) <= 40) and (desc.upper() == desc):
        return True

    low = desc.lower()
    if any(x in low for x in ["totale", "subtotale", "somma", "saldo", "risultato", "utile", "perdita"]):
        if cod == "":
            return True

    return False

# =========================================================
# CORREZIONE SEGNO "SOFT" in base a bucket
# =========================================================
def correggi_segno_atteso(sf: float, natura_code: str, tipo: str, bucket: Optional[str]) -> float:
    nu = (natura_code or "").upper()
    b = bucket
    if b is None:
        _, b2 = inferisci_tipo_bucket(nu)
        b = b2

    if tipo == "SP":
        if b == "ATTIVO":
            return abs(sf)
        if b == "PASSIVO":
            return -abs(sf)
    if tipo == "CE":
        if b == "RICAVI":
            return abs(sf)
        if b == "COSTI":
            return -abs(sf)
    return sf

# =========================================================
# ESTRAZIONE VISION (PyMuPDF + OpenAI)
# =========================================================
def crop_to_base64_jpeg(page, rect) -> str:
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
    return base64.b64encode(pix.tobytes("jpg", jpg_quality=80)).decode("utf-8")

def pagina_contiene_dati_utili(page_obj) -> bool:
    text = page_obj.get_text("text") or ""
    if not text.strip():
        # può essere tabella "non testuale", non scartare
        return True
    numeri = re.findall(r"\d+[\.,]\d+", text)
    codici = re.findall(r"\d+(?:\.\d+)+", text)
    return (len(numeri) + len(codici)) >= 4

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

async def processa_chunk_async(client_async, img_base64: str, software_hint: str, sem: asyncio.Semaphore):
    prompt = f"""
Analizza un bilancio ({software_hint}).

OBIETTIVO:
- Estrarre righe contabili con importo e natura (DARE/AVERE) in modo coerente.

REGOLE PER NATURA:
A) Se vedi colonne separate DARE/AVERE:
   - Importo in colonna DARE -> natura=DARE
   - Importo in colonna AVERE -> natura=AVERE
B) Se NON vedi colonne DARE/AVERE (solo SALDO/IMPORTO unico):
   - PASSIVO o RICAVI -> natura=AVERE
   - ATTIVO o COSTI -> natura=DARE
   - Se incerto, usa DARE ma NON inventare.

IMPORTANTE:
- Se l’importo è negativo (preceduto da - oppure tra parentesi), restituisci importo NEGATIVO.
- Mantieni anche righe totali (servono per il risultato d’esercizio).

ESTRAI:
- codice (se presente, altrimenti "")
- descrizione
- natura: DARE o AVERE
- importo: numero
- is_raggruppamento: True se totale/raggruppamento
""".strip()

    async with sem:
        async def call():
            return await client_async.chat.completions.create(
                model="gpt-4o",
                response_model=BilancioEstratto,
                max_tokens=2200,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                        ]
                    }
                ]
            )

        try:
            resp = await retry_async(call)
            return [r.model_dump() for r in resp.righe]  # tengo anche raggruppamenti
        except Exception:
            return []

async def estrai_pagina_adattivo(aclient, page, software_hint: str, sem: asyncio.Semaphore):
    h, w = page.rect.height, page.rect.width

    full = await processa_chunk_async(aclient, crop_to_base64_jpeg(page, fitz.Rect(0, 0, w, h)), software_hint, sem)
    if len(full) >= 8:
        return full

    top = await processa_chunk_async(aclient, crop_to_base64_jpeg(page, fitz.Rect(0, 0, w, h * 0.6)), software_hint, sem)
    bot = await processa_chunk_async(aclient, crop_to_base64_jpeg(page, fitz.Rect(0, h * 0.4, w, h)), software_hint, sem)
    return full + top + bot

async def estrai_tutto_smart(pdf_bytes: bytes, software_hint: str, max_pagine_vision: int = 20):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sem = asyncio.Semaphore(5)
    aclient = instructor.from_openai(AsyncOpenAI(api_key=api_key))

    tasks = []
    pagine_utili = 0
    pagine_scartate = 0
    vision_count = 0

    prog_bar = st.progress(0)
    status_text = st.empty()

    for i, p in enumerate(doc):
        if not pagina_contiene_dati_utili(p):
            pagine_scartate += 1
            status_text.text(f"Pag {i+1}/{len(doc)}: scartata (bassa densità)")
            prog_bar.progress((i + 1) / len(doc))
            continue

        vision_count += 1
        if vision_count > max_pagine_vision:
            pagine_scartate += 1
            status_text.text(f"Pag {i+1}/{len(doc)}: scartata (limite Vision)")
            prog_bar.progress((i + 1) / len(doc))
            continue

        pagine_utili += 1
        status_text.text(f"Pag {i+1}/{len(doc)}: Vision in corso...")
        tasks.append(estrai_pagina_adattivo(aclient, p, software_hint, sem))
        prog_bar.progress((i + 1) / len(doc))

    if pagine_utili == 0:
        return [], 0, pagine_scartate

    results = await asyncio.gather(*tasks)
    flat = [item for sublist in results for item in sublist]
    return flat, pagine_utili, pagine_scartate

# =========================================================
# ESTRAZIONE CAMEL0T (PDF NATIVI)
# =========================================================
def estrai_righe_camelot(pdf_bytes: bytes):
    import tempfile

    rows = []
    debug = {"tables": 0, "rows": 0, "enabled": CAMEL0T_OK}

    if not CAMEL0T_OK:
        return [], {"error": "camelot non disponibile (pip install camelot-py[cv])"}

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
        debug["tables"] = len(tables)

        for t in tables:
            df = t.df.copy()
            df = df.replace("", None).dropna(how="all")
            if df.empty:
                continue

            for _, r in df.iterrows():
                line = " ".join([str(x) for x in r.values if x is not None]).strip()
                if not line or len(line) < 3:
                    continue

                m = re.match(r"^([0-9][0-9\.\-]*)\s+(.*)$", line)
                if m:
                    codice_raw = m.group(1).strip()
                    descr = m.group(2).strip()
                else:
                    codice_raw = ""
                    descr = line.strip()

                nums = re.findall(r"[-]?\(?\d{1,3}(?:\.\d{3})*(?:,\d{2})\)?", line)
                if not nums:
                    continue

                imp = parse_importo(nums[-1])

                # Camelot: spesso non dà segno; teniamo segno se c'è
                natura = "AVERE" if imp < 0 else "DARE"
                imp_abs = abs(imp)

                desc_low = descr.lower()
                is_raggr = False
                if codice_raw == "" and any(k in desc_low for k in ["totale", "subtotale", "somma", "saldo", "risultato", "utile", "perdita"]):
                    is_raggr = True
                if descr.isupper() and codice_raw == "" and len(descr) <= 40:
                    is_raggr = True

                rows.append({
                    "codice": codice_raw,
                    "descrizione": normalize_desc(descr),
                    "natura": natura,
                    "importo": float(imp_abs),
                    "is_raggruppamento": is_raggr
                })

        debug["rows"] = len(rows)
        return rows, debug

    except Exception as e:
        return [], {"error": str(e)}

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# =========================================================
# NORMALIZZAZIONE RIGHE ESTRATTE (preserva importo_originale)
# =========================================================
def normalizza_righe_estratte(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        desc = normalize_desc(str(r.get("descrizione", "")))
        if not desc:
            continue

        imp = parse_importo(r.get("importo", 0))
        natura = (r.get("natura") or "DARE").upper().strip()
        cod = str(r.get("codice", "")).strip()

        if natura not in ("DARE", "AVERE"):
            natura = "DARE"

        out.append({
            "codice": cod,
            "descrizione": desc,
            "importo_originale": float(imp),  # può essere negativo (Vision)
            "importo": float(imp),            # verrà reso abs dopo
            "natura": natura,
            "is_raggruppamento": bool(r.get("is_raggruppamento", False))
        })
    return out

# =========================================================
# NATURA IBRIDA (quando non c'è DARE/AVERE esplicito)
# =========================================================
PASSIVO_KW = [
    "debiti", "debito", "fornitori", "mutuo", "finanziamento", "erario",
    "tfr", "fondo", "inps", "inail", "iva debito", "ratei passivi",
    "banche passive", "scoperto", "fidi"
]
RICAVI_KW = [
    "ricavi", "vendite", "proventi", "interessi attivi", "plusval",
    "contributi", "fatturato"
]
ATTIVO_KW = [
    "crediti", "cassa", "banca", "immobilizzazioni", "rimanenze",
    "iva credito", "ratei attivi", "depositi", "anticipi"
]
COSTI_KW = [
    "acquisti", "servizi", "personale", "ammort", "oneri", "svalut",
    "spese", "canoni", "consulenze", "utenze"
]

def decide_natura_ibrida(riga: dict, vision_affidabile: bool) -> str:
    desc = (riga.get("descrizione") or "").lower()
    natura = (riga.get("natura") or "").upper()
    imp_raw = float(riga.get("importo_originale") or riga.get("importo") or 0.0)

    # Se la Vision ha già capito DARE/AVERE, teniamolo
    if vision_affidabile and natura in ("DARE", "AVERE"):
        riga["importo"] = abs(imp_raw)
        return natura

    # Se abbiamo un segno esplicito, usalo come hint
    if imp_raw < 0:
        riga["importo"] = abs(imp_raw)
        return "AVERE"

    riga["importo"] = abs(imp_raw)

    # Keyword heuristic
    if any(k in desc for k in PASSIVO_KW) or any(k in desc for k in RICAVI_KW):
        return "AVERE"
    if any(k in desc for k in ATTIVO_KW) or any(k in desc for k in COSTI_KW):
        return "DARE"

    return "DARE"

# =========================================================
# DEDUP ROBUSTO (fuzzy)
# =========================================================
def deduplica_e_pulisci(lista: List[dict]) -> List[dict]:
    cleaned = []
    for r in lista:
        desc = normalize_desc(str(r.get("descrizione", "")))
        if not desc or len(desc) < 3:
            continue

        rr = dict(r)
        rr["codice"] = str(rr.get("codice", "")).strip()
        rr["descrizione"] = desc
        rr["importo_originale"] = float(rr.get("importo_originale", rr.get("importo", 0)) or 0)
        rr["importo"] = float(rr.get("importo", 0) or 0)
        rr["natura"] = (rr.get("natura") or "DARE").upper()
        cleaned.append(rr)

    out: List[dict] = []
    for r in cleaned:
        is_dup = False
        for o in out[-250:]:
            same_code = (normalizza_codice(r["codice"]) == normalizza_codice(o["codice"])) and r["codice"] != ""
            close_amt = abs(float(r["importo"]) - float(o["importo"])) < 0.5
            sim_desc = fuzz.token_sort_ratio(r["descrizione"].lower(), o["descrizione"].lower()) >= 93
            if (same_code and close_amt) or (sim_desc and close_amt):
                is_dup = True
                break
        if not is_dup:
            out.append(r)
    return out

# =========================================================
# RETRIEVAL IBRIDO (BM25 + semantico)
# =========================================================
def ricerca_ibrida(query: str, db: dict, indices: List[int]) -> List[dict]:
    cands: List[dict] = []
    toks = simple_tokenize(query)

    bm_sc = db["bm25"].get_scores(toks)
    bm_pairs = [(s, i) for i, s in enumerate(bm_sc) if i in indices]
    bm_pairs.sort(key=lambda x: x[0], reverse=True)

    for s, i in bm_pairs[:10]:
        if s > 0:
            cands.append({"codice": db["codici"][i], "descrizione": db["desc_sem"][i], "score": float(s), "src": "bm25"})

    q_emb = semantic_model.encode(query, convert_to_tensor=True)
    sem_sc = util.cos_sim(q_emb, db["emb"])[0]
    mask = torch.zeros_like(sem_sc)
    mask[indices] = 1.0
    sem_sc = sem_sc * mask

    top = torch.topk(sem_sc, k=10)
    for s, i in zip(top[0], top[1]):
        code = db["codici"][i]
        if code not in [c["codice"] for c in cands]:
            cands.append({"codice": code, "descrizione": db["desc_sem"][i], "score": float(s), "src": "sem"})

    cands.sort(key=lambda x: (x["score"], 1 if x["src"] == "sem" else 0), reverse=True)
    return cands[:10]

def decisione_senza_llm(cands: List[dict]) -> bool:
    if not cands:
        return False
    if cands[0]["score"] > 0.82:
        if len(cands) == 1:
            return True
        if (cands[0]["score"] - cands[1]["score"]) > 0.15:
            return True
    return False

def scegli_fallback(cands: List[dict], db: dict) -> Tuple[str, str]:
    if cands:
        return cands[0]["codice"], "Fallback: scelto top candidato retrieval"
    first = next(iter(db["map_desc"].keys()), "")
    return first, "Fallback: nessun candidato, scelto primo codice disponibile"

def ragionatore_gerarchico(client: OpenAI, r: dict, info_padre: Optional[Tuple[str, str]], cands: List[dict], db: dict) -> Tuple[str, str]:
    desc = r.get("descrizione")
    cod = r.get("codice")

    contest_pdc = "NESSUN MATCH NEL PDC."
    if info_padre:
        cod_padre, desc_padre = info_padre
        contest_pdc = f"MATCH GERARCHICO: Codice {cod} -> Gruppo '{cod_padre} {desc_padre}'."

    safe = []
    for c in cands[:5]:
        code = c["codice"]
        safe.append({
            "codice": code,
            "score": round(float(c["score"]), 4),
            "tipo": db["map_tipo"].get(code, "SP"),
            "bucket": db["map_bucket"].get(code, None),
            "descrizione": (db["map_desc"].get(code, "") or "")[:80],
        })

    cand_only = [x["codice"] for x in safe]

    prompt = f"""
Sei un Revisore Contabile.
Devi selezionare il codice CORRETTO esclusivamente tra i candidati forniti.
Non inventare codici.

INPUT: "{desc}" (Cod: {cod})
IMPORTO: {r.get('importo')} NATURA: {r.get('natura')}
PDC (vincolo): {contest_pdc}

CANDIDATI (scegli uno di questi): {json.dumps(cand_only)}
DETTAGLI CANDIDATI: {json.dumps(safe)}

REGOLE:
1) Se PDC indica debito/passivo, NON selezionare conti costo.
2) Se incerto, scegli il candidato con score maggiore.

JSON OUTPUT:
{{"codice_scelto":"<uno dei candidati>", "ragionamento":"max 2 frasi"}}
""".strip()

    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        d = json.loads(res.choices[0].message.content)
        code = (d.get("codice_scelto") or "").strip()
        reason = (d.get("ragionamento") or "").strip()
        if code not in cand_only:
            return "", "LLM ha restituito codice non in lista (fallback)"
        return code, reason
    except Exception:
        return "", ""

# =========================================================
# RISULTATO D'ESERCIZIO: ricerca robusta nel RAW
# =========================================================
RIS_PATTERNS = [
    re.compile(r"utile\s*(?:\(|\[)?perdita(?:\)|\])?\s*d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"risultato\s*d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"risultato\s+netto", re.IGNORECASE),
    re.compile(r"utile\s+d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"perdita\s+d[' ]?esercizio", re.IGNORECASE),
    re.compile(r"utile\s+del\s+periodo", re.IGNORECASE),
    re.compile(r"perdita\s+del\s+periodo", re.IGNORECASE),
]

def trova_risultato_esercizio(rows: List[dict]) -> Tuple[Optional[float], Optional[str]]:
    best_score = -1
    best_val = None
    best_desc = None

    for r in rows:
        desc = (r.get("descrizione") or "").strip()
        if not desc:
            continue
        dlow = desc.lower()

        score = 0
        for p in RIS_PATTERNS:
            if p.search(desc):
                score += 3

        # boost se contiene parole chiave
        if "esercizio" in dlow:
            score += 1
        if "utile" in dlow:
            score += 1
        if "perdita" in dlow:
            score += 1

        # valore
        val_raw = float(r.get("importo_originale") or r.get("importo") or 0.0)
        val_abs = abs(val_raw)

        # scarta righe senza importo
        if val_abs < 1e-9:
            continue

        # se keyword "perdita" -> segno negativo, "utile" -> positivo
        if "perdita" in dlow:
            val = -val_abs
            score += 1
        elif "utile" in dlow:
            val = val_abs
            score += 1
        else:
            # fallback: usa natura se presente
            nat = (r.get("natura") or "").upper()
            if nat == "AVERE":
                val = val_abs
            else:
                val = -val_abs

        if score > best_score:
            best_score = score
            best_val = val
            best_desc = desc

    return best_val, best_desc

# =========================================================
# ANOMALIE (semplici)
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

# =========================================================
# UI
# =========================================================
banner_img = "Revilaw S.p.A..jpg"
if os.path.exists(banner_img):
    st.image(banner_img, use_container_width=True)
else:
    st.markdown("## 🧬 Revilaw AuditEv Hierarchy (Quality)")

with st.sidebar:
    st.markdown("### 🎛️ Pannello Controllo")
    f_pdf = st.file_uploader("1. Bilancio (PDF)", type="pdf")

    bilancio_tipo = st.selectbox("2. Tipo bilancio", ["Bilancio ordinario", "Bilancio abbreviato"])
    cls_path = CLASSIFICATION_FILES.get(bilancio_tipo)
    st.caption(f"📌 Classificazione: {cls_path.name if cls_path else 'N/D'}")

    f_pdc = st.file_uploader("3. Piano Conti (PDF/XLSX/CSV) (opzionale)", type=["xlsx", "csv", "pdf"])

    st.markdown("---")
    sw = st.selectbox("Software Origine", list(SOFTWARE_PROMPTS.keys()))
    max_pagine_vision = st.slider("Max pagine Vision (costi)", min_value=5, max_value=120, value=25, step=5)

    st.markdown("---")

    ready = (f_pdf is not None) and (api_key is not None) and (len(api_key) > 10)
    start_analysis = st.button("🚀 AVVIA ELABORAZIONE", disabled=not ready, type="primary")

    if not api_key:
        st.caption("🔴 Configura OPENAI_API_KEY (st.secrets o variabile d'ambiente).")
    elif not ready:
        st.caption("🔴 Carica almeno il PDF bilancio.")

# =========================================================
# PIPELINE
# =========================================================
if start_analysis:
    main_placeholder = st.empty()
    main_placeholder.info("⚙️ Inizializzazione...")

    try:
        raw_client = OpenAI(api_key=api_key)

        # reset state
        st.session_state["data_processed"] = None
        st.session_state["logs"] = []
        st.session_state["raw_pdf_rows"] = []
        st.session_state["camelot_rows"] = []
        st.session_state["camelot_debug"] = {}
        st.session_state["estrazione_metodo"] = "N/D"
        st.session_state["risultato_pdf"] = None
        st.session_state["risultato_pdf_desc"] = None

        # 1) Carica classificazione dal file locale
        if not cls_path or not cls_path.exists():
            st.error(f"File classificazione non trovato: {cls_path}")
            st.stop()

        df_c = pd.read_excel(cls_path, dtype=str)
        if df_c is None or df_c.empty:
            st.error("Errore: classificazione vuota o non leggibile.")
            st.stop()

        db = inizializza_motori_ricerca(df_c)
        if db is None:
            st.error("Errore: non riesco a inizializzare i motori di ricerca dalla classificazione.")
            st.stop()

        st.session_state["combo_options"] = db["combo"]
        st.session_state["map_code_desc"] = db["map_desc"]
        st.session_state["map_code_natura"] = db["map_natura"]
        st.session_state["map_code_tipo"] = db["map_tipo"]
        st.session_state["map_code_bucket"] = db["map_bucket"]

        log("CLASSIFICAZIONE", f"{bilancio_tipo}: indicizzate {len(db['codici'])} righe.")

        # 2) PDC
        pdc_norm_map = {}
        if f_pdc:
            main_placeholder.info("🧬 Indicizzazione PDC...")
            pdc_norm_map = analizza_pdc_universale(f_pdc)
            st.success(f"PDC indicizzato: {len(pdc_norm_map)} voci.")
            log("PDC", f"Indicizzato: {len(pdc_norm_map)} voci.")
        else:
            log("PDC", "Non caricato.", level="WARN")

        # 3) Estrazione dati: Camelot -> fallback Vision
        f_pdf.seek(0)
        pdf_bytes = f_pdf.read()

        raw_rows: List[dict] = []
        if CAMEL0T_OK:
            main_placeholder.info("📄 Tentativo estrazione tabelle (Camelot)...")
            raw_camelot, dbg_cam = estrai_righe_camelot(pdf_bytes)

            st.session_state["camelot_rows"] = raw_camelot
            st.session_state["camelot_debug"] = dbg_cam

            if raw_camelot and len(raw_camelot) >= 40:
                raw_rows = raw_camelot
                st.session_state["estrazione_metodo"] = "CAMEL0T"
                log("CAMEL0T", f"OK: tables={dbg_cam.get('tables')} rows={dbg_cam.get('rows')}")
                main_placeholder.success(f"✅ Camelot OK: {dbg_cam.get('rows')} righe estratte da {dbg_cam.get('tables')} tabelle")
            else:
                st.session_state["estrazione_metodo"] = "VISION"
                log("CAMEL0T", f"Fallback Vision: rows={len(raw_camelot)} dbg={dbg_cam}", level="WARN")
        else:
            st.session_state["estrazione_metodo"] = "VISION"
            st.session_state["camelot_debug"] = {"error": "camelot non installato"}
            log("CAMEL0T", "Non disponibile (pip install camelot-py[cv])", level="WARN")

        if not raw_rows:
            main_placeholder.info("🤖 Estrazione bilancio (Vision adattivo + filtro)...")
            raw_v, p_utili, p_scartate = run_async(
                estrai_tutto_smart(pdf_bytes, SOFTWARE_PROMPTS[sw], max_pagine_vision=max_pagine_vision)
            )
            raw_rows = raw_v
            log("VISION", f"Finita: pagine utili={p_utili}, scartate={p_scartate}, righe raw={len(raw_rows)}")
            main_placeholder.success(f"✅ Vision: analizzate {p_utili} pagine. Scartate {p_scartate}.")

        # 4) Normalizza + natura ibrida
        raw_norm = normalizza_righe_estratte(raw_rows)

        vc = pd.Series([r.get("natura", "") for r in raw_norm]).value_counts()
        ratio_avere = vc.get("AVERE", 0) / max(1, len(raw_norm))
        # se metodo Vision e ha abbastanza AVERE, consideriamo affidabile
        vision_affidabile = (st.session_state["estrazione_metodo"] == "VISION") and (ratio_avere >= 0.12)

        log("NATURA", f"Natura estratta: {dict(vc)} | ratio_avere={ratio_avere:.2f} | vision_affidabile={vision_affidabile}")

        for r in raw_norm:
            r["natura"] = decide_natura_ibrida(r, vision_affidabile)

        st.session_state["raw_pdf_rows"] = raw_norm.copy()

        # Risultato d'esercizio dal RAW (anche raggruppamenti)
        ris_val, ris_desc = trova_risultato_esercizio(raw_norm)
        st.session_state["risultato_pdf"] = ris_val
        st.session_state["risultato_pdf_desc"] = ris_desc
        if ris_val is not None:
            log("RISULTATO_PDF", f"Trovato: {ris_val:.2f} | '{ris_desc}'")
        else:
            log("RISULTATO_PDF", "Non trovato", level="WARN")

        # 5) Dedup + pulizia
        unique = deduplica_e_pulisci(raw_norm)

        # 6) Filtra raggruppamenti SOLO per classificazione
        unique_analitici = [r for r in unique if not e_raggruppamento(r)]
        log("PULIZIA", f"Righe totali={len(unique)} | analitiche={len(unique_analitici)} | raggruppamenti={len(unique)-len(unique_analitici)}")

        if not unique_analitici:
            st.error("❌ Nessun conto analitico dopo filtro raggruppamenti.")
            st.stop()

        # 7) Classificazione
        main_placeholder.info(f"🧠 Classificazione su {len(unique_analitici)} righe...")
        final = []
        progress_bar = st.progress(0)
        tot_rows = len(unique_analitici)

        for i, r in enumerate(unique_analitici):
            progress_bar.progress((i + 1) / tot_rows)

            desc = r.get("descrizione", "").strip()
            cod_raw = r.get("codice", "").strip()

            info_padre = None
            if pdc_norm_map and cod_raw:
                p = trova_padre_gerarchico(cod_raw, pdc_norm_map)
                if p and p[0] is not None:
                    info_padre = (p[0], p[1])  # type: ignore

            if vision_affidabile:
                idxs = db["idx_d"] if r.get("natura") == "DARE" else db["idx_a"]
            else:
                idxs = list(range(len(db["codici"])))

            query = desc
            if info_padre and info_padre[1]:
                query += f" | PDC:{info_padre[1]}"

            cands = ricerca_ibrida(query, db, idxs)
            top_score = cands[0]["score"] if cands else 0.0

            if decisione_senza_llm(cands):
                code = cands[0]["codice"]
                reason = "Scelta deterministica: retrieval alto"
            else:
                code, reason = ragionatore_gerarchico(raw_client, r, info_padre, cands, db)

            if not code:
                code, fb_reason = scegli_fallback(cands, db)
                reason = (reason + " | " if reason else "") + fb_reason

            desc_audit = db["map_desc"].get(code, "???")
            combo = f"{code} | {desc_audit}"
            if combo not in db["combo"]:
                combo = next((x for x in db["combo"] if x.startswith(f"{code} |")), combo)

            imp = abs(float(r.get("importo", 0) or 0.0))
            sd = imp if r.get("natura") == "DARE" else 0.0
            sa = imp if r.get("natura") == "AVERE" else 0.0

            tipo = db["map_tipo"].get(code, "SP")
            bucket = db["map_bucket"].get(code, None)
            natura_code = db["map_natura"].get(code, "")

            sf = (sa - sd) if tipo == "CE" else (sd - sa)
            if bucket in ("ATTIVO", "PASSIVO", "COSTI", "RICAVI"):
                sf = correggi_segno_atteso(sf, natura_code, tipo, bucket)

            debug_gerarchia = info_padre[1] if info_padre else "N/A"

            final.append({
                "Codice": cod_raw,
                "Descrizione": desc,
                "Saldo dare": sd,
                "Saldo avere": sa,
                "Saldo finale": sf,
                "Classificazione_Combo": combo,
                "Tipo": tipo,
                "Inverti Segno": False,
                "Gerarchia PDC": debug_gerarchia,
                "Ragionamento AI": reason,
                "Confidence": float(top_score),
            })

        final_df = pd.DataFrame(final)
        final_df.sort_values(by="Codice", inplace=True)
        st.session_state["data_processed"] = final_df
        main_placeholder.empty()
        st.rerun()

    except Exception as e:
        st.error(f"Errore Critico: {str(e)}")
        log("ERRORE", str(e), level="ERROR")

# =========================================================
# VISUALIZZAZIONE
# =========================================================
if st.session_state["data_processed"] is not None:
    df = st.session_state["data_processed"]
    map_tipo = st.session_state.get("map_code_tipo", {})
    combo_options = st.session_state.get("combo_options", [])

    st.markdown("### 📊 Dashboard")

    with st.container():
        df_sp = df[df["Tipo"] == "SP"]
        df_ce = df[df["Tipo"] == "CE"]

        tot_attivo = df_sp[df_sp["Saldo finale"] > 0]["Saldo finale"].sum()
        tot_passivo = df_sp[df_sp["Saldo finale"] < 0]["Saldo finale"].sum()
        tot_risultato = df_ce["Saldo finale"].sum()
        tot_sp_netto = df_sp["Saldo finale"].sum()
        diff = tot_sp_netto - tot_risultato

        ris_pdf = st.session_state.get("risultato_pdf", None)
        ris_desc = st.session_state.get("risultato_pdf_desc", None)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ATTIVO", f"€ {tot_attivo:,.2f}", delta="Patrimoniale")
        c2.metric("PASSIVO", f"€ {tot_passivo:,.2f}", delta="Patrimoniale", delta_color="inverse")
        c3.metric("RISULTATO (RICL.)", f"€ {tot_risultato:,.2f}", delta="Economico")

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

    # Preview estrazione (Camelot/Vision)
    with st.expander("📄 Preview estrazione (Camelot / Vision)", expanded=False):
        metodo = st.session_state.get("estrazione_metodo", "N/D")
        st.markdown(f"**Metodo usato:** `{metodo}`")

        cam_dbg = st.session_state.get("camelot_debug", {})
        cam_rows = st.session_state.get("camelot_rows", [])

        if cam_rows:
            st.markdown(f"✅ **Camelot**: tabelle={cam_dbg.get('tables', 'N/D')} | righe={len(cam_rows)}")
            df_prev = pd.DataFrame(cam_rows)

            st.markdown("### 👀 Prime 20 righe estratte da Camelot")
            st.dataframe(df_prev.head(20), use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📌 Righe SENZA codice (prime 20)")
                no_code = df_prev[df_prev["codice"].astype(str).str.strip() == ""]
                st.dataframe(no_code.head(20), use_container_width=True, hide_index=True)

            with col2:
                st.markdown("### 📌 Raggruppamenti (prime 20)")
                raggr = df_prev[df_prev["is_raggruppamento"] == True]
                st.dataframe(raggr.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("Camelot non ha prodotto righe (oppure non è stato usato).")
            st.write("Debug Camelot:", cam_dbg)

        vision_rows = st.session_state.get("raw_pdf_rows", [])
        if vision_rows:
            st.markdown("---")
            st.markdown(f"✅ **RAW (post-normalizzazione)**: righe={len(vision_rows)}")
            st.dataframe(pd.DataFrame(vision_rows).head(20), use_container_width=True, hide_index=True)

    with st.expander("🪵 Debug log (chiaro)", expanded=False):
        logs = st.session_state.get("logs", [])
        if not logs:
            st.info("Nessun log disponibile.")
        else:
            st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)

    tab_edit, tab_exp = st.tabs(["📝 Revisione", "💾 Export"])

    # ----------------------------
    # TAB EDIT (solo 5 colonne)
    # ----------------------------
    with tab_edit:
        # solo 5 colonne richieste
        df_v = df[["Codice", "Descrizione", "Saldo finale", "Inverti Segno", "Classificazione_Combo"]].copy()

        edited = st.data_editor(
            df_v,
            column_order=["Codice", "Descrizione", "Saldo finale", "Inverti Segno", "Classificazione_Combo"],
            column_config={
                "Classificazione_Combo": st.column_config.SelectboxColumn("Classificazione", options=combo_options, width="medium"),
                "Saldo finale": st.column_config.NumberColumn("Saldo finale", format="%.2f €", disabled=True),
                "Inverti Segno": st.column_config.CheckboxColumn("Inverti Segno"),
                "Codice": st.column_config.TextColumn("Codice conto", disabled=True),
                "Descrizione": st.column_config.TextColumn("Descrizione", disabled=True),
            },
            height=600,
            use_container_width=True,
            hide_index=True
        )

        upd = False
        for i, row in edited.iterrows():
            orig = df.loc[i]
            if (row["Classificazione_Combo"] != orig["Classificazione_Combo"]) or (bool(row["Inverti Segno"]) != bool(orig["Inverti Segno"])):
                upd = True
                c = str(row["Classificazione_Combo"])
                code = c.split("|")[0].strip() if "|" in c else c.strip()

                tipo = map_tipo.get(code, "SP")
                d = float(orig["Saldo dare"])
                a = float(orig["Saldo avere"])

                sf = (a - d) if tipo == "CE" else (d - a)
                if bool(row["Inverti Segno"]):
                    sf = -sf

                df.at[i, "Classificazione_Combo"] = c
                df.at[i, "Saldo finale"] = sf
                df.at[i, "Tipo"] = tipo
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
                    "Saldo dare", "Saldo avere", "Saldo finale",
                    "Classificazione", "Tipo", "Gerarchia PDC",
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

                for r in range(1, len(dx2) + 1):
                    for c in [2, 3, 4]:
                        val = dx2.iloc[r - 1, c]
                        try:
                            fval = float(val)
                        except Exception:
                            fval = 0.0
                        ws.write_number(r, c, fval, fmt_neg if fval < 0 else fmt_curr)

                ws.set_column("A:A", 14)
                ws.set_column("B:B", 55)
                ws.set_column("C:E", 18)
                ws.set_column("F:F", 16)
                ws.set_column("G:G", 10)
                ws.set_column("H:H", 28)
                ws.set_column("I:I", 12)
                ws.set_column("J:J", 60)

            out.seek(0)
            st.download_button("💾 SCARICA EXCEL", data=out, file_name="Revilaw_Bilancio_QUALITY.xlsx", type="primary")

        except Exception as e:
            st.error(f"Errore export Excel: {e}")

else:
    st.markdown("<div style='text-align: center; color: gray; margin-top: 50px;'><h3>👋 Carica i file per iniziare</h3></div>", unsafe_allow_html=True)
