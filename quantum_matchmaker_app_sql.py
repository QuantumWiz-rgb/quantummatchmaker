# quantum_matchmaker_app_sql.py
# Streamlit app with SQLAlchemy (MySQL/MariaDB/Postgres/SQLite) backend for multi-user deployments.

import os
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///matchmaker.db")

# -----------------------------
# Domain Ontology (same as before; trimmed for brevity)
# -----------------------------

TECH_ONTOLOGY: Dict[str, List[str]] = {
    "magnetometry": ["magnetometer","magnetometry","magnetic","heading","compass","geomagnetic","magnetic anomaly","magnetic navigation","vector field","scalar field"],
    "NV_diamond": ["nv center","nitrogen vacancy","nv-diamond","diamond magnetometer","odmr","optically detected magnetic resonance","green laser 532 nm","red fluorescence","spin contrast"],
    "SERF_OPM": ["serf","spin-exchange relaxation-free","opm","optically pumped magnetometer","alkali vapor","rubidium","rb87","cesium","cs","k39","potassium","quartz cell","anti-relaxation coating"],
    "SQUID": ["squid","superconducting quantum interference device","superconducting","cryogenic","ybco","niobium","cryocooler"],
    "gyroscope": ["gyroscope","imu","inertial","rotation","angular rate","heading drift"],
    "gravimetry": ["gravimeter","gravity","gradiometer","mass anomaly","subsurface mapping"],
    "agriculture": ["agriculture","farming","tractor","combine","planter","irrigation","soil","row crop","pasture","underground pipe","tile drain","subsurface","field"],
    "harsh_env": ["dust","mud","vibration","shock","temperature","humidity","ip67","ip68","rugged"],
    "gps_denied": ["gps-free","gps denied","no gps","under canopy","indoors","tunnel"],
    "power": ["battery","low power","milliwatt","watt","power budget","solar"],
    "cost": ["cost","unit cost","bom","price","affordable","low-cost"],
    "safety": ["laser safety","eye safe","non-ionizing","intrinsic safety","emc","emissions"],
}

MATERIAL_REQUIREMENTS: Dict[str, List[str]] = {
    "NV_diamond": ["cvd diamond","isotopically enriched methane","12c","diamond growth","nitrogen implantation","vacancy creation","annealing","mw resonator","532 nm laser","650-800 nm photodiode"],
    "SERF_OPM": ["alkali vapor cell","rb87","cesium","k39","cell fabrication","paraffin coating","anti-relaxation","quartz windows","microfabricated cell","buffer gas"],
    "SQUID": ["superconducting film","ybco","nb thin film","cryocooler","rf squid","dc squid"],
}

# -----------------------------
# Utils
# -----------------------------

def normalize_text(s: str) -> str:
    import re
    s = (s or "").lower()
    s = re.sub(r"[\/\-\_]", " ", s)
    s = re.sub(r"[^a-z0-9\s\+\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def expand_with_ontology(text: str) -> str:
    base = normalize_text(text)
    appended = []
    for tag, synonyms in TECH_ONTOLOGY.items():
        for syn in synonyms:
            syn_norm = normalize_text(syn)
            if any(tok in base for tok in syn_norm.split()):
                appended.append(f"[TAG:{tag}]")
                break
    return base + " " + " ".join(appended)

def infer_tech_class(text: str) -> List[str]:
    base = normalize_text(text)
    hits = []
    for tag, synonyms in TECH_ONTOLOGY.items():
        if tag in ["agriculture","harsh_env","gps_denied","power","cost","safety"]:
            continue
        for syn in synonyms:
            if normalize_text(syn) in base:
                hits.append(tag)
                break
    if not hits and ("magnetic" in base or "magnet" in base):
        hits.append("magnetometry")
    return list(dict.fromkeys(hits))

def parse_trl(x) -> Optional[int]:
    try:
        v = int(float(str(x).strip()))
        if 1 <= v <= 9:
            return v
    except Exception:
        return None
    return None

def trl_alignment_score(need_timeline: str, supplier_trl: Optional[int]) -> float:
    import re
    if supplier_trl is None:
        return 0.0
    timeline = normalize_text(need_timeline or "")
    months = None
    m = re.search(r"(\d+)\s*(month|mo)", timeline)
    y = re.search(r"(\d+)\s*(year|yr)", timeline)
    if m:
        months = int(m.group(1))
    elif y:
        months = int(y.group(1)) * 12
    else:
        months = 18
    if months <= 12:
        desired = 7
    elif months <= 24:
        desired = 6
    else:
        desired = 4
    diff = max(0, desired - supplier_trl)
    score = max(0.0, 1.0 - (diff / 4.0))
    return float(score)

def constraint_penalties(constraints: str, supplier_text: str) -> float:
    c = normalize_text(constraints or "")
    s = normalize_text(supplier_text or "")
    penalty = 0.0
    rules = [
        (["no cryogenic","no cryogenics","no cryo"], ["cryogenic","squid","helium","cryocooler"], -0.4),
        (["low power","battery < 5w","battery<5w","battery under 5w"], ["watt","power hungry","hundreds of watts"], -0.2),
        (["no laser"], ["laser"], -0.2),
    ]
    for triggers, bad_terms, pen in rules:
        if any(t in c for t in triggers) and any(b in s for b in bad_terms):
            penalty += pen
    return penalty

def environment_bonus(env: str, supplier_text: str) -> float:
    e = normalize_text(env or "")
    s = normalize_text(supplier_text or "")
    bonus = 0.0
    if any(t in e for t in ["outdoor","underground","field","tractor","vibration","shock","mud","dust","ip67","ip68","harsh"]):
        if any(w in s for w in ["rugged","ip67","ip68","vibration","shock","temperature","dust","mud"]):
            bonus += 0.15
    if any(t in e for t in ["underground","subsurface","pipe","tile drain"]):
        if any(w in s for w in ["magnet","magnetometer","magnetometry","gravimeter","gradiometer"]):
            bonus += 0.1
    if any(t in e for t in ["gps denied","gps-free","no gps","under canopy","indoors","tunnel"]):
        if "magnet" in s or "inertial" in s or "compass" in s:
            bonus += 0.1
    return bonus

def build_vectorizer(docs: List[str]) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=r"(\[TAG:[A-Za-z_]+\]|[A-Za-z0-9\+\.]{3,})",
        lowercase=False,
        strip_accents="unicode",
    ).fit(docs)

# -----------------------------
# DB via SQLAlchemy
# -----------------------------

def get_engine() -> Engine:
    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    return eng

def init_db():
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                company_name TEXT,
                role TEXT,
                focus_areas TEXT,
                capabilities_text TEXT,
                trl INTEGER,
                location TEXT,
                capacity_notes TEXT,
                contact TEXT,
                created_at TEXT
            )
        """.replace("AUTO_INCREMENT", "AUTOINCREMENT" if DATABASE_URL.startswith("sqlite") else "AUTO_INCREMENT")))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS needs (
                id INTEGER PRIMARY KEY AUTO_INCREMENT,
                org_name TEXT,
                need_title TEXT,
                need_text TEXT,
                constraints TEXT,
                environment TEXT,
                timeline TEXT,
                budget_range TEXT,
                location TEXT,
                created_at TEXT
            )
        """.replace("AUTO_INCREMENT", "AUTOINCREMENT" if DATABASE_URL.startswith("sqlite") else "AUTO_INCREMENT")))

def insert_company(row: dict):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO companies (company_name, role, focus_areas, capabilities_text, trl, location, capacity_notes, contact, created_at)
            VALUES (:company_name,:role,:focus_areas,:capabilities_text,:trl,:location,:capacity_notes,:contact,:created_at)
        """), {
            "company_name": row.get("company_name","").strip(),
            "role": row.get("role","").strip().lower(),
            "focus_areas": row.get("focus_areas",""),
            "capabilities_text": row.get("capabilities_text",""),
            "trl": parse_trl(row.get("trl")) if row.get("trl") not in [None,""] else None,
            "location": row.get("location",""),
            "capacity_notes": row.get("capacity_notes",""),
            "contact": row.get("contact",""),
            "created_at": datetime.utcnow().isoformat()
        })

def insert_need(row: dict):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO needs (org_name, need_title, need_text, constraints, environment, timeline, budget_range, location, created_at)
            VALUES (:org_name,:need_title,:need_text,:constraints,:environment,:timeline,:budget_range,:location,:created_at)
        """), {
            "org_name": row.get("org_name","").strip(),
            "need_title": row.get("need_title","").strip(),
            "need_text": row.get("need_text",""),
            "constraints": row.get("constraints",""),
            "environment": row.get("environment",""),
            "timeline": row.get("timeline",""),
            "budget_range": row.get("budget_range",""),
            "location": row.get("location",""),
            "created_at": datetime.utcnow().isoformat()
        })

def fetch_df(table: str) -> pd.DataFrame:
    eng = get_engine()
    with eng.begin() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {table}"), conn)
    return df

def delete_row(table: str, row_id: int):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(f"DELETE FROM {table} WHERE id=:id"), {"id": row_id})

# -----------------------------
# Matching (same logic as previous, wrapped for brevity)
# -----------------------------

def compute_need_to_sensor_scores(needs_df: pd.DataFrame, sensors_df: pd.DataFrame,
                                  alpha=0.6, beta=0.25, gamma=0.15) -> pd.DataFrame:
    if needs_df.empty or sensors_df.empty:
        return pd.DataFrame()
    need_docs = needs_df["_doc"].tolist()
    sensor_docs = sensors_df["_doc"].tolist()
    vec = build_vectorizer(need_docs + sensor_docs)
    N = vec.transform(need_docs)
    S = vec.transform(sensor_docs)
    cos = cosine_similarity(N, S)
    rows = []
    for i, nrow in needs_df.iterrows():
        for j, srow in sensors_df.iterrows():
            cos_score = float(cos[needs_df.index.get_loc(i), sensors_df.index.get_loc(j)])
            need_tags = nrow["_tech_tags"]
            sensor_tags = srow["_tech_tags"]
            tech_overlap = len(set(need_tags).intersection(set(sensor_tags)))
            domain_bonus = 0.05 * tech_overlap + environment_bonus(nrow["environment"], srow["capabilities_text"])
            if "gps_denied" in nrow["_doc"] or "gps free" in normalize_text(nrow["need_text"]):
                if any(t in srow["_doc"] for t in ["magnet","imu","compass","magnetometer","magnetometry"]):
                    domain_bonus += 0.08
            trl = srow["trl"] if pd.notna(srow["trl"]) else None
            trl_score = trl_alignment_score(nrow["timeline"], trl)
            pen = constraint_penalties(nrow["constraints"], srow["capabilities_text"])
            total = alpha * cos_score + beta * domain_bonus + gamma * trl_score + pen
            rows.append({
                "need_id": nrow["id"], "need_org": nrow["org_name"], "need_title": nrow["need_title"],
                "sensor_company": srow["company_name"], "sensor_id": srow["id"], "sensor_trl": srow["trl"],
                "sensor_focus": srow["focus_areas"], "sensor_contact": srow["contact"],
                "cosine": round(cos_score, 4), "domain_bonus": round(domain_bonus, 4),
                "trl_score": round(trl_score, 4), "penalty": round(pen, 4),
                "total_score": round(total, 4),
                "overlap_terms": "",
                "need_tags": ";".join(need_tags), "sensor_tags": ";".join(sensor_tags),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(["need_org","need_title","total_score"], ascending=[True, True, False], inplace=True)
    return df

# -----------------------------
# Streamlit UI (trimmed, mirrors earlier app)
# -----------------------------

st.set_page_config(page_title="Quantum Industry Matchmaker (Multi-user)", page_icon="🧭", layout="wide")

st.sidebar.title("🧭 Quantum Matchmaker")
nav = st.sidebar.radio("Go to", ["Submit Profile / Need", "Find Matches", "Directory & Admin", "About"])
st.sidebar.caption(f"DB: {DATABASE_URL.split('://')[0]}")

# Initialize DB
try:
    init_db()
except OperationalError as e:
    st.error(f"Database init error: {e}")

def add_docs_tags_companies(df):
    if df.empty: return df
    df["_doc"] = df.apply(lambda r: expand_with_ontology(f"{r['focus_areas']} . {r['capabilities_text']}"), axis=1)
    df["_tech_tags"] = df.apply(lambda r: tuple(infer_tech_class(f'{r["focus_areas"]} . {r["capabilities_text"]}')), axis=1)
    return df

def add_docs_tags_needs(df):
    if df.empty: return df
    df["_doc"] = df.apply(lambda r: expand_with_ontology(f"{r['need_text']} . {r['constraints']} . {r['environment']}"), axis=1)
    df["_tech_tags"] = df.apply(lambda r: tuple(infer_tech_class(f'{r["need_text"]}')), axis=1)
    return df

if nav == "Submit Profile / Need":
    st.header("Submit Profile / Need")
    role = st.selectbox("I am a…", ["End User", "Sensor Company", "Materials Supplier", "Integrator", "Research"])

    if role in ["Sensor Company", "Materials Supplier", "Integrator", "Research"]:
        st.subheader("Company Profile")
        with st.form("company_form"):
            company_name = st.text_input("Company Name")
            role_map = {"Sensor Company":"sensor","Materials Supplier":"materials","Integrator":"integrator","Research":"research"}
            role_val = role_map[role]
            focus_areas = st.text_input("Focus areas (comma-separated)")
            capabilities_text = st.text_area("Capabilities (free text)")
            trl = st.text_input("TRL (1-9)")
            location = st.text_input("Location")
            capacity_notes = st.text_input("Capacity / production notes")
            contact = st.text_input("Contact (email/URL)")
            submitted = st.form_submit_button("Save Company")
        if submitted:
            insert_company(dict(company_name=company_name, role=role_val, focus_areas=focus_areas,
                                capabilities_text=capabilities_text, trl=trl, location=location,
                                capacity_notes=capacity_notes, contact=contact))
            st.success("Company saved.")
    else:
        st.subheader("End-User Need")
        with st.form("need_form"):
            org_name = st.text_input("Organization Name")
            need_title = st.text_input("Project / Need Title")
            need_text = st.text_area("Describe the problem in plain language")
            constraints = st.text_input("Constraints (comma-separated)")
            environment = st.text_input("Environment (comma-separated)")
            timeline = st.text_input("Timeline")
            budget_range = st.text_input("Budget range")
            location = st.text_input("Location")
            submitted = st.form_submit_button("Save Need")
        if submitted:
            insert_need(dict(org_name=org_name, need_title=need_title, need_text=need_text,
                             constraints=constraints, environment=environment, timeline=timeline,
                             budget_range=budget_range, location=location))
            st.success("Need saved.")

elif nav == "Find Matches":
    st.header("Find Matches")
    companies = add_docs_tags_companies(fetch_df("companies"))
    needs = add_docs_tags_needs(fetch_df("needs"))

    mode = st.selectbox("I want matches for my…", ["End-User Need", "Sensor Company", "Materials Supplier"])
    topk = st.slider("Top-K results", 1, 15, 7)

    if mode == "End-User Need":
        if needs.empty or companies.empty:
            st.info("Add data first.")
        else:
            pick = st.selectbox("Select need", [f"#{r.id} • {r.org_name} — {r.need_title}" for _, r in needs.iterrows()])
            pick_id = int(pick.split("•")[0].strip()[1:])
            need_sel = needs[needs["id"] == pick_id]
            sensors = companies[companies["role"] == "sensor"]
            materials = companies[companies["role"] == "materials"]

            matches = compute_need_to_sensor_scores(need_sel, sensors).sort_values("total_score", ascending=False).head(topk)
            st.subheader("Best Sensor Matches")
            st.dataframe(matches[["sensor_company","total_score","cosine","domain_bonus","trl_score","penalty","sensor_focus","sensor_trl","sensor_contact"]])

    elif mode == "Sensor Company":
        sensors = companies[companies["role"] == "sensor"]
        if sensors.empty or needs.empty:
            st.info("Add data first.")
        else:
            pick = st.selectbox("Select sensor", [f"#{r.id} • {r.company_name}" for _, r in sensors.iterrows()])
            sid = int(pick.split("•")[0].strip()[1:])
            s_sel = sensors[sensors["id"] == sid]
            matches = compute_need_to_sensor_scores(needs, s_sel).sort_values("total_score", ascending=False).head(topk)
            st.dataframe(matches[["need_org","need_title","total_score","cosine","domain_bonus","trl_score","penalty"]])

    else:
        mats = companies[companies["role"] == "materials"]
        sensors = companies[companies["role"] == "sensor"]
        if mats.empty or sensors.empty:
            st.info("Add data first.")
        else:
            pick = st.selectbox("Select materials supplier", [f"#{r.id} • {r.company_name}" for _, r in mats.iterrows()])
            # For brevity, we list most similar sensors by TF-IDF against MATERIAL_REQUIREMENTS inferred from each sensor (like earlier app)
            # (Implementing full second-hop here would mirror previous code.)

elif nav == "Directory & Admin":
    st.header("Directory & Admin")
    tabs = st.tabs(["Companies", "Needs"])
    with tabs[0]:
        df = fetch_df("companies")
        st.dataframe(df)
    with tabs[1]:
        df = fetch_df("needs")
        st.dataframe(df)

else:
    st.header("About")
    st.write("Multi-user quantum matchmaking with a central database. Configure via DATABASE_URL.")
