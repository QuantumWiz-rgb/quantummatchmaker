# quantum_matchmaker_app_sql.py
# Streamlit app with SQLAlchemy backend + accounts + per-user privacy.
# Chain of command (per-application):
# - End-user sees SENSOR matches only (private needs/cases)
# - Sensor sees END-USER leads (opt-in) + MATERIALS matches (scoped by selected Opportunity)
# - Materials sees SENSOR matches only (no end-user visibility)

import os, re
from datetime import datetime, timezone
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from passlib.hash import pbkdf2_sha256

# ---------- Config ----------
DATABASE_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL", "sqlite:///matchmaker.db")
ADMIN_EMAILS = set(
    e.strip().lower()
    for e in (os.getenv("ADMIN_EMAILS") or st.secrets.get("ADMIN_EMAILS", "")).replace(";", ",").split(",")
    if e.strip()
)

# ---------- Ontology (trimmed) ----------
TECH_ONTOLOGY: Dict[str, List[str]] = {
    "magnetometry": ["magnetometer","magnetometry","magnetic","heading","compass","geomagnetic","magnetic anomaly","magnetic navigation","vector field","scalar field"],
    "NV_diamond": ["nv center","nitrogen vacancy","nv-diamond","diamond magnetometer","odmr","optically detected magnetic resonance","green laser 532 nm","red fluorescence","spin contrast","ltm","laser threshold","diamond","cvd"],
    "SERF_OPM": ["serf","spin-exchange relaxation-free","opm","optically pumped magnetometer","alkali vapor","rubidium","rb87","cesium","cs","k39","potassium","quartz cell","anti-relaxation coating","paraffin"],
    "SQUID": ["squid","superconducting quantum interference device","superconducting","cryogenic","ybco","niobium","cryocooler","thin film"],
    "gyroscope": ["gyroscope","imu","inertial","rotation","angular rate","heading drift"],
    "gravimetry": ["gravimeter","gravity","gradiometer","mass anomaly","subsurface mapping"],
    "agriculture": ["agriculture","farming","tractor","combine","planter","irrigation","soil","row crop","pasture","underground pipe","tile drain","subsurface","field"],
    "harsh_env": ["dust","mud","vibration","shock","temperature","humidity","ip67","ip68","rugged"],
    "gps_denied": ["gps-free","gps denied","no gps","under canopy","indoors","tunnel","aircraft","airplane","cockpit"],
    "power": ["battery","low power","milliwatt","watt","power budget","solar"],
    "cost": ["cost","unit cost","bom","price","affordable","low-cost"],
    "safety": ["laser safety","eye safe","non-ionizing","intrinsic safety","emc","emissions"],
}

# ---------- Utils ----------
def normalize_text(s: str) -> str:
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
    if supplier_trl is None:
        return 0.0
    timeline = normalize_text(need_timeline or "")
    months = None
    m = re.search(r"(\d+)\s*(month|mo)", timeline)
    y = re.search(r"(\d+)\s*(year|yr)", timeline)
    if m: months = int(m.group(1))
    elif y: months = int(y.group(1)) * 12
    else: months = 18
    desired = 7 if months <= 12 else 6 if months <= 24 else 4
    diff = max(0, desired - supplier_trl)
    return max(0.0, 1.0 - (diff / 4.0))

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
    if any(t in e for t in ["outdoor","underground","field","tractor","vibration","shock","mud","dust","ip67","ip68","harsh","aircraft","airplane"]):
        if any(w in s for w in ["rugged","ip67","ip68","vibration","shock","temperature","dust","mud"]):
            bonus += 0.15
    if any(t in e for t in ["underground","subsurface","pipe","tile drain"]):
        if any(w in s for w in ["magnet","magnetometer","magnetometry","gravimeter","gradiometer"]):
            bonus += 0.1
    if any(t in e for t in ["gps denied","gps-free","no gps","under canopy","indoors","tunnel","aircraft","airplane"]):
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

# ---------- DB ----------
def get_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def _auto_inc_kw():
    return "AUTOINCREMENT" if DATABASE_URL.startswith("sqlite") else "AUTO_INCREMENT"

def init_db():
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY {_auto_inc_kw()},
                email VARCHAR(255) UNIQUE,
                org_name TEXT,
                account_role VARCHAR(32),
                password_hash TEXT,
                created_at TEXT
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY {_auto_inc_kw()},
                company_name TEXT,
                role TEXT,
                focus_areas TEXT,
                capabilities_text TEXT,
                trl INTEGER,
                location TEXT,
                capacity_notes TEXT,
                contact TEXT,
                public_profile TINYINT,
                owner_id INTEGER,
                created_at TEXT
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS needs (
                id INTEGER PRIMARY KEY {_auto_inc_kw()},
                org_name TEXT,
                need_title TEXT,
                need_text TEXT,
                constraints TEXT,
                environment TEXT,
                timeline TEXT,
                budget_range TEXT,
                location TEXT,
                share_with_suppliers TINYINT,
                owner_id INTEGER,
                created_at TEXT
            )
        """))
        # NEW: opportunities table (per-application containers owned by sensors)
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS opportunities (
                id INTEGER PRIMARY KEY {_auto_inc_kw()},
                owner_id INTEGER,
                sensor_company_id INTEGER,
                need_id INTEGER NULL,
                title TEXT,
                technique TEXT,
                status TEXT,
                created_at TEXT
            )
        """))
        # Migrations (add columns if missing)
        def ensure_col(table, name, sqldef):
            try:
                conn.execute(text(f"SELECT {name} FROM {table} LIMIT 1"))
            except Exception:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {sqldef}"))
        ensure_col("companies", "public_profile", "public_profile TINYINT")
        ensure_col("companies", "owner_id", "owner_id INTEGER")
        ensure_col("needs", "share_with_suppliers", "share_with_suppliers TINYINT")
        ensure_col("needs", "owner_id", "owner_id INTEGER")
        try:
            conn.execute(text("SELECT owner_id, sensor_company_id, need_id, technique, status FROM opportunities LIMIT 1"))
        except Exception:
            pass

def create_user(email: str, org: str, role: str, password: str) -> Optional[int]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email.lower()}).fetchone()
        if row: return None
        ph = pbkdf2_sha256.hash(password)
        conn.execute(text("""
            INSERT INTO users (email, org_name, account_role, password_hash, created_at)
            VALUES (:e,:o,:r,:p,:ts)
        """), {"e": email.lower(), "o": org, "r": role, "p": ph, "ts": datetime.now(timezone.utc).isoformat()})
        uid = conn.execute(text("SELECT id FROM users WHERE email=:e"), {"e": email.lower()}).fetchone()[0]
        return uid

def verify_user(email: str, password: str) -> Optional[dict]:
    eng = get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("SELECT id,email,org_name,account_role,password_hash FROM users WHERE email=:e"),
                           {"e": email.lower()}).fetchone()
        if not row: return None
        if pbkdf2_sha256.verify(password, row[4]):
            return {"id": row[0], "email": row[1], "org": row[2], "role": row[3]}
        return None

def insert_company(row: dict, owner_id: Optional[int]):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO companies (company_name, role, focus_areas, capabilities_text, trl, location,
                                   capacity_notes, contact, public_profile, owner_id, created_at)
            VALUES (:company_name,:role,:focus_areas,:capabilities_text,:trl,:location,
                    :capacity_notes,:contact,:public_profile,:owner_id,:ts)
        """), {
            "company_name": row.get("company_name",""),
            "role": row.get("role",""),
            "focus_areas": row.get("focus_areas",""),
            "capabilities_text": row.get("capabilities_text",""),
            "trl": parse_trl(row.get("trl")) if row.get("trl") not in [None,""] else None,
            "location": row.get("location",""),
            "capacity_notes": row.get("capacity_notes",""),
            "contact": row.get("contact",""),
            "public_profile": 1 if row.get("public_profile", True) else 0,
            "owner_id": owner_id,
            "ts": datetime.now(timezone.utc).isoformat()
        })

def insert_need(row: dict, owner_id: Optional[int]):
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO needs (org_name, need_title, need_text, constraints, environment, timeline,
                               budget_range, location, share_with_suppliers, owner_id, created_at)
            VALUES (:org_name,:need_title,:need_text,:constraints,:environment,:timeline,
                    :budget_range,:location,:share_with_suppliers,:owner_id,:ts)
        """), {
            "org_name": row.get("org_name",""),
            "need_title": row.get("need_title",""),
            "need_text": row.get("need_text",""),
            "constraints": row.get("constraints",""),
            "environment": row.get("environment",""),
            "timeline": row.get("timeline",""),
            "budget_range": row.get("budget_range",""),
            "location": row.get("location",""),
            "share_with_suppliers": 1 if row.get("share_with_suppliers", False) else 0,
            "owner_id": owner_id,
            "ts": datetime.now(timezone.utc).isoformat()
        })

def insert_opportunity(owner_id: int, sensor_company_id: int, need_id: Optional[int], title: str, technique: str) -> int:
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO opportunities (owner_id, sensor_company_id, need_id, title, technique, status, created_at)
            VALUES (:owner_id,:sensor_company_id,:need_id,:title,:technique,:status,:ts)
        """), {
            "owner_id": owner_id,
            "sensor_company_id": sensor_company_id,
            "need_id": need_id,
            "title": title or "New Opportunity",
            "technique": technique or "",
            "status": "open",
            "ts": datetime.now(timezone.utc).isoformat()
        })
        oid = conn.execute(text("SELECT last_insert_rowid()")) if DATABASE_URL.startswith("sqlite") else conn.execute(text("SELECT LAST_INSERT_ID()"))
        return int(list(oid)[0][0])

def fetch_df(table: str, where: Optional[str] = None, params: Optional[dict] = None) -> pd.DataFrame:
    eng = get_engine()
    with eng.begin() as conn:
        q = f"SELECT * FROM {table}"
        if where: q += f" WHERE {where}"
        df = pd.read_sql(text(q), conn, params=params)
    return df

def fetch_opportunities(owner_id: int, sensor_company_id: Optional[int] = None) -> pd.DataFrame:
    where = "owner_id=:oid"
    params = {"oid": owner_id}
    if sensor_company_id:
        where += " AND sensor_company_id=:sid"
        params["sid"] = sensor_company_id
    return fetch_df("opportunities", where=where, params=params)

def delete_row(table: str, row_id: int, owner_id: Optional[int] = None, admin: bool = False) -> bool:
    eng = get_engine()
    with eng.begin() as conn:
        if admin:
            res = conn.execute(text(f"DELETE FROM {table} WHERE id=:id"), {"id": row_id}).rowcount
        else:
            res = conn.execute(
                text(f"DELETE FROM {table} WHERE id=:id AND owner_id=:oid"),
                {"id": row_id, "oid": owner_id}
            ).rowcount
    return res > 0

# ---------- Matching ----------
def add_docs_tags_companies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.fillna("")
    df["_doc"] = df.apply(lambda r: expand_with_ontology(f"{r.get('focus_areas','')} . {r.get('capabilities_text','')}"), axis=1)
    df["_tech_tags"] = df.apply(lambda r: tuple(infer_tech_class(f"{r.get('focus_areas','')} . {r.get('capabilities_text','')}")), axis=1)
    return df

def add_docs_tags_needs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.fillna("")
    df["_doc"] = df.apply(lambda r: expand_with_ontology(f"{r.get('need_text','')} . {r.get('constraints','')} . {r.get('environment','')}"), axis=1)
    df["_tech_tags"] = df.apply(lambda r: tuple(infer_tech_class(f"{r.get('need_text','')}")), axis=1)
    return df

def _safe_sort(df: pd.DataFrame, by, ascending):
    if df.empty:
        return df
    by = [c for c in by if c in df.columns]
    if not by:
        return df
    if isinstance(ascending, list):
        ascending = ascending[:len(by)]
    return df.sort_values(by, ascending=ascending)

def compute_need_to_sensor_scores(needs_df: pd.DataFrame, sensors_df: pd.DataFrame,
                                  alpha=0.6, beta=0.25, gamma=0.15) -> pd.DataFrame:
    cols = [
        "need_id","need_org","need_title","sensor_company","sensor_id","sensor_trl",
        "sensor_focus","sensor_contact","cosine","domain_bonus","trl_score","penalty",
        "total_score","need_tags","sensor_tags"
    ]
    if needs_df is None or needs_df.empty or sensors_df is None or sensors_df.empty:
        return pd.DataFrame(columns=cols)

    need_docs = needs_df["_doc"].tolist()
    sensor_docs = sensors_df["_doc"].tolist()
    try:
        vec = build_vectorizer(need_docs + sensor_docs)
        N = vec.transform(need_docs)
        S = vec.transform(sensor_docs)
        cos = cosine_similarity(N, S)
    except Exception:
        cos = np.zeros((len(need_docs), len(sensor_docs)))

    rows = []
    for i, nrow in needs_df.iterrows():
        for j, srow in sensors_df.iterrows():
            cos_score = float(cos[needs_df.index.get_loc(i), sensors_df.index.get_loc(j)])
            need_tags = nrow["_tech_tags"]; sensor_tags = srow["_tech_tags"]
            tech_overlap = len(set(need_tags).intersection(set(sensor_tags)))
            domain_bonus = 0.05 * tech_overlap + environment_bonus(nrow.get("environment",""), srow.get("capabilities_text",""))
            if "gps_denied" in nrow["_doc"] or "gps free" in normalize_text(nrow.get("need_text","")):
                if any(t in srow["_doc"] for t in ["magnet","imu","compass","magnetometer","magnetometry"]):
                    domain_bonus += 0.08
            trl = srow.get("trl") if pd.notna(srow.get("trl")) else None
            trl_score = trl_alignment_score(nrow.get("timeline",""), trl)
            pen = constraint_penalties(nrow.get("constraints",""), srow.get("capabilities_text",""))
            total = alpha * cos_score + beta * domain_bonus + gamma * trl_score + pen
            rows.append({
                "need_id": nrow["id"], "need_org": nrow["org_name"], "need_title": nrow["need_title"],
                "sensor_company": srow["company_name"], "sensor_id": srow["id"], "sensor_trl": srow.get("trl"),
                "sensor_focus": srow.get("focus_areas",""), "sensor_contact": srow.get("contact",""),
                "cosine": round(cos_score, 4), "domain_bonus": round(domain_bonus, 4),
                "trl_score": round(trl_score, 4), "penalty": round(pen, 4),
                "total_score": round(total, 4),
                "need_tags": ";".join(need_tags), "sensor_tags": ";".join(sensor_tags),
            })
    df = pd.DataFrame(rows, columns=cols)
    return _safe_sort(df, ["need_org","need_title","total_score"], [True, True, False])

def compute_sensor_to_material_scores(sensor_df: pd.DataFrame, materials_df: pd.DataFrame,
                                      need_context: Optional[pd.Series] = None,
                                      alpha=0.65, beta=0.25, gamma=0.10) -> pd.DataFrame:
    """Match ONE sensor profile to many materials; optionally use a need context for env/constraints bonuses."""
    cols = [
        "sensor_id","sensor_company","materials_company","materials_id",
        "cosine","domain_bonus","context_bonus","total_score",
        "sensor_tags","materials_tags","materials_focus","materials_contact"
    ]
    if sensor_df is None or sensor_df.empty or materials_df is None or materials_df.empty:
        return pd.DataFrame(columns=cols)

    # Ensure positional indices align with cosine vector
    sensor_df = sensor_df.iloc[[0]].reset_index(drop=True)
    materials_df = materials_df.reset_index(drop=True)

    sdocs = sensor_df["_doc"].tolist()
    mdocs = materials_df["_doc"].tolist()
    try:
        vec = build_vectorizer(sdocs + mdocs)
        S = vec.transform(sdocs)
        M = vec.transform(mdocs)
        cos = cosine_similarity(S, M)[0]  # shape (m,)
    except Exception:
        cos = np.zeros(len(mdocs))

    env = need_context.get("environment","") if need_context is not None else ""
    constraints = need_context.get("constraints","") if need_context is not None else ""
    sensor_row = sensor_df.iloc[0]
    s_tags = set(sensor_row["_tech_tags"])

    rows = []
    for pos, mrow in materials_df.iterrows():  # pos = 0..m-1 after reset_index
        m_tags = set(mrow["_tech_tags"])
        dom_bonus = 0.06 * len(s_tags.intersection(m_tags))
        ctx_bonus = 0.0
        if need_context is not None:
            ctx_bonus += environment_bonus(env, mrow.get("capabilities_text",""))
            ctx_bonus += constraint_penalties(constraints, mrow.get("capabilities_text",""))
        total = alpha * float(cos[pos]) + beta * dom_bonus + gamma * ctx_bonus
        rows.append({
            "sensor_id": sensor_row["id"], "sensor_company": sensor_row["company_name"],
            "materials_company": mrow["company_name"], "materials_id": mrow["id"],
            "cosine": round(float(cos[pos]), 4), "domain_bonus": round(dom_bonus, 4),
            "context_bonus": round(ctx_bonus, 4), "total_score": round(total, 4),
            "sensor_tags": ";".join(sorted(s_tags)), "materials_tags": ";".join(sorted(m_tags)),
            "materials_focus": mrow.get("focus_areas",""), "materials_contact": mrow.get("contact",""),
        })
    df = pd.DataFrame(rows, columns=cols)
    return _safe_sort(df, ["total_score"], [False])

def compute_material_to_sensor_scores(material_df: pd.DataFrame, sensors_df: pd.DataFrame,
                                      alpha=0.65, beta=0.25, gamma=0.10) -> pd.DataFrame:
    """Mirror of sensor->materials for the materials user (no end-user context)."""
    cols = [
        "materials_id","materials_company","sensor_company","sensor_id",
        "cosine","domain_bonus","total_score","materials_tags","sensor_tags",
        "sensor_focus","sensor_contact"
    ]
    if material_df is None or material_df.empty or sensors_df is None or sensors_df.empty:
        return pd.DataFrame(columns=cols)

    # Ensure positional indices align with cosine vector
    material_df = material_df.iloc[[0]].reset_index(drop=True)
    sensors_df = sensors_df.reset_index(drop=True)

    mdocs = material_df["_doc"].tolist()
    sdocs = sensors_df["_doc"].tolist()
    try:
        vec = build_vectorizer(mdocs + sdocs)
        M = vec.transform(mdocs)
        S = vec.transform(sdocs)
        cos = cosine_similarity(M, S)[0]  # shape (n_sensors,)
    except Exception:
        cos = np.zeros(len(sdocs))

    m_tags = set(material_df.iloc[0]["_tech_tags"])
    rows = []
    for pos, srow in sensors_df.iterrows():  # pos = 0..n-1 after reset_index
        s_tags = set(srow["_tech_tags"])
        dom_bonus = 0.06 * len(m_tags.intersection(s_tags))
        total = alpha * float(cos[pos]) + beta * dom_bonus + gamma * 0.0
        rows.append({
            "materials_id": material_df.iloc[0]["id"], "materials_company": material_df.iloc[0]["company_name"],
            "sensor_company": srow["company_name"], "sensor_id": srow["id"],
            "cosine": round(float(cos[pos]), 4), "domain_bonus": round(dom_bonus, 4),
            "total_score": round(total, 4),
            "materials_tags": ";".join(sorted(m_tags)), "sensor_tags": ";".join(sorted(s_tags)),
            "sensor_focus": srow.get("focus_areas",""), "sensor_contact": srow.get("contact",""),
        })
    df = pd.DataFrame(rows, columns=cols)
    return _safe_sort(df, ["total_score"], [False])

# ---------- UI ----------
st.set_page_config(page_title="Quantum Matchmaker (Auth)", page_icon="🧭", layout="wide")
init_db()

def auth_bar():
    st.sidebar.markdown("### Account")
    if st.session_state.get("user"):
        u = st.session_state["user"]
        st.sidebar.write(f"Signed in as **{u['org']}** ({u['email']})")
        if st.sidebar.button("Sign out"):
            st.session_state.pop("user", None)
            st.rerun()
        st.sidebar.markdown("---")
    else:
        tab_login, tab_signup = st.sidebar.tabs(["Sign in", "Create account"])
        with tab_login:
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Sign in"):
                user = verify_user(email, pw)
                if user:
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
        with tab_signup:
            org = st.text_input("Organization")
            email2 = st.text_input("Work email")
            role = st.selectbox("Primary role", ["end_user","sensor","materials","integrator","research"])
            pw1 = st.text_input("Password", type="password")
            pw2 = st.text_input("Confirm password", type="password")
            if st.button("Create account"):
                if not org or not email2 or not pw1:
                    st.error("All fields required.")
                elif pw1 != pw2:
                    st.error("Passwords do not match.")
                else:
                    uid = create_user(email2, org, role, pw1)
                    st.success("Account created. Please sign in.") if uid else st.error("Email already registered.")

auth_bar()

nav = st.sidebar.radio("Go to", ["Submit Profile / Need", "Find Matches", "Directory & Admin", "About"])
st.sidebar.caption(f"DB: {DATABASE_URL.split('://')[0]}")

def require_login():
    if not st.session_state.get("user"):
        st.warning("Please sign in first (see sidebar).")
        st.stop()

# ----- Submit -----
if nav == "Submit Profile / Need":
    require_login()
    user = st.session_state["user"]
    st.header("Submit Profile / Need")

    role_sel = st.selectbox("I am a…", ["End User", "Sensor Company", "Materials Supplier", "Integrator", "Research"])

    if role_sel in ["Sensor Company", "Materials Supplier", "Integrator", "Research"]:
        st.subheader("Company Profile")
        with st.form("company_form"):
            company_name = st.text_input("Company Name", value=user["org"] or "")
            role_map = {"Sensor Company":"sensor","Materials Supplier":"materials","Integrator":"integrator","Research":"research"}
            role_val = role_map[role_sel]
            focus_areas = st.text_input("Focus areas (comma-separated)")
            capabilities_text = st.text_area("Capabilities (free text)")
            trl = st.text_input("TRL (1-9)")
            location = st.text_input("Location")
            capacity_notes = st.text_input("Capacity / production notes")
            contact = st.text_input("Contact (email/URL)")
            public_profile = st.checkbox("Make this profile discoverable for matching", value=True)
            submitted = st.form_submit_button("Save Company")
        if submitted:
            insert_company(dict(company_name=company_name, role=role_val, focus_areas=focus_areas,
                                capabilities_text=capabilities_text, trl=trl, location=location,
                                capacity_notes=capacity_notes, contact=contact, public_profile=public_profile),
                           owner_id=user["id"])
            st.success("Company saved.")
    else:
        st.subheader("End-User Need (private to your account)")
        with st.form("need_form"):
            org_name = user["org"] or st.text_input("Organization Name")
            need_title = st.text_input("Project / Need Title")
            need_text = st.text_area("Describe the problem in plain language")
            constraints = st.text_input("Constraints (comma-separated)")
            environment = st.text_input("Environment (comma-separated)")
            timeline = st.text_input("Timeline")
            budget_range = st.text_input("Budget range")
            location = st.text_input("Location")
            share_with_suppliers = st.checkbox("Allow matched suppliers to view this need title/org (public lead)", value=False)
            submitted = st.form_submit_button("Save Need")
        if submitted:
            insert_need(dict(org_name=org_name, need_title=need_title, need_text=need_text,
                             constraints=constraints, environment=environment, timeline=timeline,
                             budget_range=budget_range, location=location, share_with_suppliers=share_with_suppliers),
                        owner_id=user["id"])
            st.success("Need saved.")

# ----- Find Matches -----
elif nav == "Find Matches":
    require_login()
    user = st.session_state["user"]
    st.header("Find Matches")

    # Shared prep
    companies_public = fetch_df("companies", where="public_profile=1", params={})
    needs_owned = fetch_df("needs", where="owner_id=:oid", params={"oid": user["id"]})
    companies_public = add_docs_tags_companies(companies_public)
    needs_owned = add_docs_tags_needs(needs_owned)

    mode = st.selectbox("I am looking for matches for my…", ["End-User Need", "Sensor Company", "Materials Supplier"])
    topk = st.slider("Top-K results", 1, 20, 7)

    # ---- End-User: see sensors only
    if mode == "End-User Need":
        if needs_owned.empty:
            st.info("You have no saved needs yet. Add one in Submit.")
        else:
            pick = st.selectbox("Select need", [f"#{r.id} • {r.org_name} — {r.need_title}" for _, r in needs_owned.iterrows()])
            pick_id = int(pick.split("•")[0].strip()[1:])
            need_sel = needs_owned[needs_owned["id"] == pick_id]
            sensors = companies_public[companies_public["role"] == "sensor"]
            matches = compute_need_to_sensor_scores(need_sel, sensors)
            if matches.empty:
                st.info("No matching sensors yet. Add supplier profiles, or broaden your need description.")
            else:
                st.subheader("Best Sensor Matches")
                show_cols = [c for c in [
                    "sensor_company","total_score","cosine","domain_bonus","trl_score","penalty",
                    "sensor_focus","sensor_trl","sensor_contact"
                ] if c in matches.columns]
                st.dataframe(matches.head(topk)[show_cols], use_container_width=True)

    # ---- Sensor: see end-user leads + materials (scoped by chosen Opportunity)
    elif mode == "Sensor Company":
        my_sensors = fetch_df("companies", where="owner_id=:oid AND role='sensor'", params={"oid": user["id"]})
        my_sensors = add_docs_tags_companies(my_sensors)
        if my_sensors.empty:
            st.info("You have no sensor profiles yet. Save one in Submit.")
        else:
            pick_sensor = st.selectbox("Select your sensor profile", [f"#{r.id} • {r.company_name}" for _, r in my_sensors.iterrows()])
            sid = int(pick_sensor.split("•")[0].strip()[1:])
            s_sel = my_sensors[my_sensors["id"] == sid]

            tabs = st.tabs(["End-User Leads", "Materials Matches"])

            # --- End-User Leads (opt-in)
            with tabs[0]:
                public_needs = fetch_df("needs", where="share_with_suppliers=1", params={})
                public_needs = add_docs_tags_needs(public_needs)
                lead_matches = compute_need_to_sensor_scores(public_needs, s_sel)
                if lead_matches.empty:
                    st.info("No public end-user leads yet. End-users must opt-in by checking 'share with suppliers'.")
                else:
                    st.subheader("Matching End-User Leads")
                    show_cols = [c for c in [
                        "need_org","need_title","total_score","cosine","domain_bonus","trl_score","penalty"
                    ] if c in lead_matches.columns]
                    st.dataframe(lead_matches.head(topk)[show_cols], use_container_width=True)

                    # Create an Opportunity from a selected lead
                    pick_lead = st.selectbox(
                        "Create an Opportunity for this lead",
                        [f"#{r.need_id} • {r.need_org} — {r.need_title}" for _, r in lead_matches.iterrows()]
                    )
                    if pick_lead:
                        lead_id = int(pick_lead.split("•")[0].strip()[1:])
                        with st.form("form_create_opp"):
                            opp_title = st.text_input("Opportunity title", value=f"{s_sel.iloc[0]['company_name']} × Lead #{lead_id}")
                            technique = st.text_input("Technique (e.g., NV-ODMR, NV-LTM, SERF OPM)")
                            create = st.form_submit_button("Start Opportunity")
                        if create:
                            oid = insert_opportunity(owner_id=user["id"], sensor_company_id=sid, need_id=lead_id,
                                                     title=opp_title, technique=technique)
                            st.success(f"Opportunity #{oid} created.")

            # --- Materials Matches (scoped by Opportunity if chosen)
            with tabs[1]:
                st.caption("Tip: choose an Opportunity to include that lead's environment/constraints in the ranking.")
                my_opps = fetch_opportunities(owner_id=user["id"], sensor_company_id=sid)
                opp_label_list = ["Ad-hoc (no lead context)"] + [f"#{r.id} • {r.title or 'Untitled'}" for _, r in my_opps.iterrows()]
                pick_opp = st.selectbox("Use Opportunity context", opp_label_list)
                need_ctx = None
                if pick_opp != "Ad-hoc (no lead context)":
                    opp_id = int(pick_opp.split("•")[0].strip()[1:])
                    opp_row = my_opps[my_opps["id"] == opp_id].iloc[0]
                    if pd.notna(opp_row.get("need_id")) and int(opp_row["need_id"]) > 0:
                        need_ctx_df = fetch_df("needs", where="id=:nid", params={"nid": int(opp_row["need_id"])})
                        need_ctx = need_ctx_df.iloc[0] if not need_ctx_df.empty else None

                materials = companies_public[companies_public["role"] == "materials"]
                s2m = compute_sensor_to_material_scores(s_sel, materials, need_context=need_ctx)
                if s2m.empty:
                    st.info("No matching materials yet. Encourage materials suppliers to add public profiles, or broaden your technique/capability text.")
                else:
                    st.subheader("Best Materials Matches")
                    show_cols = [c for c in [
                        "materials_company","total_score","cosine","domain_bonus","context_bonus",
                        "materials_focus","materials_contact","materials_tags"
                    ] if c in s2m.columns]
                    st.dataframe(s2m.head(topk)[show_cols], use_container_width=True)

    # ---- Materials: see sensors only (no end-user visibility)
    else:
        my_mats = fetch_df("companies", where="owner_id=:oid AND role='materials'", params={"oid": user["id"]})
        my_mats = add_docs_tags_companies(my_mats)
        if my_mats.empty:
            st.info("You have no materials profiles yet. Save one in Submit.")
        else:
            pick_mat = st.selectbox("Select your materials profile", [f"#{r.id} • {r.company_name}" for _, r in my_mats.iterrows()])
            mid = int(pick_mat.split("•")[0].strip()[1:])
            m_sel = my_mats[my_mats["id"] == mid]
            sensors_pub = companies_public[companies_public["role"] == "sensor"]
            m2s = compute_material_to_sensor_scores(m_sel, sensors_pub)
            if m2s.empty:
                st.info("No matching sensors yet. Encourage sensors to publish profiles, or broaden your capability text.")
            else:
                st.subheader("Best Sensor Matches")
                show_cols = [c for c in [
                    "sensor_company","total_score","cosine","domain_bonus","sensor_focus","sensor_contact","sensor_tags"
                ] if c in m2s.columns]
                st.dataframe(m2s.head(topk)[show_cols], use_container_width=True)

# ----- Directory & Admin -----
elif nav == "Directory & Admin":
    require_login()
    user = st.session_state["user"]
    if user["email"].lower() not in ADMIN_EMAILS:
        st.error("Admin access only.")
        st.stop()

    st.header("Directory & Admin")
    tabs = st.tabs(["Companies", "Needs", "Opportunities", "Users"])

    with tabs[0]:
        df = fetch_df("companies")
        st.dataframe(df)
        del_id = st.number_input("Delete company by ID", min_value=0, step=1, value=0)
        if st.button("Delete Company"):
            ok = delete_row("companies", int(del_id), owner_id=user["id"], admin=True)
            st.success("Deleted." if ok else "Not found.")

    with tabs[1]:
        df = fetch_df("needs")
        st.dataframe(df)
        del_id = st.number_input("Delete need by ID", min_value=0, step=1, value=0, key="del_need")
        if st.button("Delete Need"):
            ok = delete_row("needs", int(del_id), owner_id=user["id"], admin=True)
            st.success("Deleted." if ok else "Not found.")

    with tabs[2]:
        df = fetch_df("opportunities")
        st.dataframe(df)
        del_id = st.number_input("Delete opportunity by ID", min_value=0, step=1, value=0, key="del_opp")
        if st.button("Delete Opportunity"):
            ok = delete_row("opportunities", int(del_id), owner_id=user["id"], admin=True)
            st.success("Deleted." if ok else "Not found.")

    with tabs[3]:
        df = fetch_df("users")
        st.dataframe(df[["id","email","org_name","account_role","created_at"]])

# ----- About -----
else:
    st.header("About")
    st.write("""
This build implements **chain of command** with **per-application Opportunities**:
- End-users match to **sensors** (private).
- Sensors match to **end-user leads** (opt-in) **and** to **materials** within a selected Opportunity.
- Materials match to **sensors** only (no end-user visibility).
""")
