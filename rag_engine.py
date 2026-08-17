"""
rag_engine.py — Medical RAG (Retrieval-Augmented Generation) Engine for NeuroScan AI
Provides clinical decision support for Doctors and grounded plain-language medical Q&A for Patients.
"""

import os
import glob
import re
from typing import List, Dict, Any, Tuple

# Try loading environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Try SentenceTransformers & FAISS
EMBEDDINGS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    EMBEDDINGS_AVAILABLE = True
except Exception:
    EMBEDDINGS_AVAILABLE = False

# Try Groq API for ultra-fast Llama-3 inference
GROQ_AVAILABLE = False
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ── Global In-Memory Vector Store & Chunks ────────────────────────────────────
_KNOWLEDGE_CHUNKS: List[Dict[str, Any]] = []
_EMBEDDING_MODEL = None
_FAISS_INDEX = None


def _load_knowledge_base_files() -> List[Dict[str, str]]:
    """Read all markdown clinical guides from the knowledge_base directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(base_dir, "knowledge_base")
    
    docs = []
    if not os.path.exists(kb_path):
        return docs

    md_files = glob.glob(os.path.join(kb_path, "*.md"))
    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                filename = os.path.basename(file_path)
                category = filename.replace(".md", "").title()
                docs.append({
                    "filename": filename,
                    "category": category,
                    "content": content
                })
        except Exception as e:
            print(f"[RAG] Error reading {file_path}: {e}")
            
    return docs


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping paragraph-aligned chunks."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(current_chunk) + len(p) < chunk_size:
            current_chunk += "\n\n" + p if current_chunk else p
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def init_rag_engine():
    """Index knowledge base into local semantic vector index."""
    global _KNOWLEDGE_CHUNKS, _EMBEDDING_MODEL, _FAISS_INDEX

    if _KNOWLEDGE_CHUNKS and (_FAISS_INDEX is not None or not EMBEDDINGS_AVAILABLE):
        return

    docs = _load_knowledge_base_files()
    if not docs:
        print("[RAG] Warning: No knowledge base documents found.")
        return

    all_chunks = []
    for doc in docs:
        chunks = _chunk_text(doc["content"])
        for chunk in chunks:
            all_chunks.append({
                "source": doc["filename"],
                "category": doc["category"],
                "text": chunk
            })

    _KNOWLEDGE_CHUNKS = all_chunks

    if EMBEDDINGS_AVAILABLE:
        try:
            # Attempt offline-first loading from local cache to eliminate DNS/network errors and retry pauses
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                # If not in cache, fallback to online download
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

            texts = [c["text"] for c in _KNOWLEDGE_CHUNKS]
            embeddings = _EMBEDDING_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            
            dimension = embeddings.shape[1]
            _FAISS_INDEX = faiss.IndexFlatIP(dimension)
            _FAISS_INDEX.add(embeddings.astype("float32"))
            print(f"[RAG] Indexed {len(_KNOWLEDGE_CHUNKS)} clinical chunks into FAISS vector store.")
        except Exception as e:
            print(f"[RAG] Vector index fallback active: {e}")



def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve top-K most relevant medical passages for a query."""
    init_rag_engine()
    if not _KNOWLEDGE_CHUNKS:
        return []

    # Semantic FAISS retrieval
    if EMBEDDINGS_AVAILABLE and _EMBEDDING_MODEL and _FAISS_INDEX is not None:
        try:
            q_emb = _EMBEDDING_MODEL.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            distances, indices = _FAISS_INDEX.search(q_emb.astype("float32"), top_k)
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if idx < len(_KNOWLEDGE_CHUNKS):
                    item = _KNOWLEDGE_CHUNKS[idx].copy()
                    item["score"] = float(score)
                    results.append(item)
            return results
        except Exception as e:
            print(f"[RAG Retrieval Exception]: {e}")

    # Fallback keyword relevance retrieval
    terms = set(re.findall(r"\w+", query.lower()))
    scored_chunks = []
    for c in _KNOWLEDGE_CHUNKS:
        text_lower = c["text"].lower()
        score = sum(text_lower.count(t) for t in terms)
        if score > 0:
            item = c.copy()
            item["score"] = score
            scored_chunks.append(item)

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k] if scored_chunks else _KNOWLEDGE_CHUNKS[:top_k]


def _call_groq_llm(messages: List[Dict[str, str]], username: str = None) -> str:
    """Call Groq API for ultra-fast Llama-3 inference with auto-fallback to supported models and audit error logging."""
    import traceback
    try:
        import database as db
    except ImportError:
        db = None

    groq_key = None
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "groq" in st.secrets:
            groq_key = st.secrets["groq"].get("api_key")
    except Exception:
        pass

    if not groq_key:
        groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key or not GROQ_AVAILABLE:
        return None

    candidate_models = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ]

    last_error = None
    try:
        client = groq.Groq(api_key=groq_key)
        for model_name in candidate_models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=650
                )
                if response and response.choices:
                    return response.choices[0].message.content
            except Exception as model_err:
                last_error = str(model_err)
                # Log model attempt failure if needed
                continue
    except Exception as e:
        last_error = str(e)
        print(f"[Groq LLM Exception]: {e}")

    # Log to Database Error Logs Table if LLM failed
    if last_error and db is not None:
        try:
            db.log_error(
                error_type="LLM_API_ERROR",
                severity="WARNING",
                message=f"Groq LLM Inference Error: {last_error[:450]}",
                component="rag_llm",
                username=username,
                stack_trace=traceback.format_exc()
            )
        except Exception:
            pass

    return None


def query_medical_rag(
    user_query: str,
    diagnosis_context: str = None,
    tumor_area_cm2: float = None,
    role: str = "patient",
    username: str = None,
    report_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Main RAG query interface with deep report awareness and clinical grounding.
    Returns: {"answer": str, "sources": list, "confidence": float, "mode": str}
    """
    # Extract diagnosis from report_data if provided
    active_diag = diagnosis_context
    active_area = tumor_area_cm2
    if report_data:
        if not active_diag and report_data.get("diagnosis"):
            active_diag = report_data.get("diagnosis")
        if active_area is None and report_data.get("area_cm2") is not None:
            active_area = report_data.get("area_cm2")

    # Build enriched context query for retrieval
    search_query = user_query
    if active_diag and active_diag.lower() != "notumor":
        search_query = f"{active_diag} brain tumor {user_query}"

    retrieved = retrieve_relevant_chunks(search_query, top_k=3)
    sources = list({c["source"] for c in retrieved})
    passages_text = "\n\n".join([f"[{c['category']} Reference]:\n{c['text']}" for c in retrieved])

    # Build structured report context block
    report_block = ""
    if report_data:
        p_name = report_data.get("patient_name") or "Patient"
        p_age = report_data.get("patient_age") or "N/A"
        p_gender = report_data.get("patient_gender") or "N/A"
        diag_val = report_data.get("diagnosis", active_diag or "Not specified")
        conf_pct = report_data.get("confidence_pct")
        probs = report_data.get("probabilities", {})
        area_cm = report_data.get("area_cm2", active_area)
        area_mm = report_data.get("area_mm2")
        px = report_data.get("tumor_pixels")
        shape_lbl = report_data.get("shape_label")
        circ = report_data.get("circularity")
        comp = report_data.get("compactness")
        sol = report_data.get("solidity")
        focus = report_data.get("gradcam_focus")
        mean_conf = report_data.get("mean_seg_confidence")

        prob_str = ", ".join([f"{k.title()}: {v*100:.1f}%" for k, v in probs.items() if isinstance(v, (int, float))]) if probs else "N/A"

        report_block = f"""
=== ACTIVE MRI SCAN DIAGNOSTIC REPORT ===
• Patient: {p_name} | Age: {p_age} | Gender: {p_gender}
• Primary AI Diagnosis: {diag_val.upper() if diag_val else 'Pending'} ({f'{conf_pct:.1f}% confidence' if conf_pct else 'Assessed'})
• Differential Class Probabilities: {prob_str}
• Segmented Tumor Area: {f'{area_cm:.2f} cm² ({area_mm:,.1f} mm², {px:,} pixels)' if area_cm else 'No tumor segmented / N/A'}
• Tumor Shape & Morphometry: {f'{shape_lbl} (Circularity: {circ:.2f}, Compactness: {comp:.2f}, Solidity: {sol:.2f})' if shape_lbl else 'N/A'}
• Segmentation Model Confidence: {f'{mean_conf:.1f}% mean' if mean_conf else 'N/A'}
• Classifier Attention Region: {focus or 'N/A'}
=========================================
"""

    # System Instructions depending on Role
    if role == "doctor":
        system_prompt = (
            "You are NeuroScan AI Clinical Copilot, an expert neuro-oncology clinical decision support system. "
            "You provide authoritative, evidence-based recommendations, surgical pathways (Simpson / GTR), "
            "WHO CNS 5 classification criteria, molecular biomarker testing protocols (IDH1/2, MGMT, 1p/19q), "
            "and radiation/chemotherapy regimens (Stupp protocol). "
            "Directly answer questions referencing the active patient diagnostic report metrics whenever available, "
            "and ground your response in the provided clinical literature."
        )
        user_prompt = f"""
{report_block if report_block else f"Clinical Context: Diagnosis={active_diag or 'Pending'}, Area={f'{active_area:.2f} cm²' if active_area else 'N/A'}"}

Retrieved Oncology Literature Guidelines:
{passages_text}

Doctor's Query:
{user_query}

Provide a direct, medically precise, and structured response addressing the doctor's query. Incorporate relevant numbers from the scan report if applicable.
"""
    else:
        system_prompt = (
            "You are NeuroScan Health Copilot, an intelligent, empathetic, and medically accurate patient educator. "
            "Your job is to clearly explain brain MRI scan results, medical terms, tumor types, recovery steps, and doctor recommendations. "
            "Use clear, reassuring language without medical jargon. When the patient asks about their scan or report, "
            "reference the specific findings from their active diagnostic report accurately. "
            "Always encourage discussion with their specialist for definitive medical decisions."
        )
        user_prompt = f"""
{report_block if report_block else f"Patient Scan Context: Result={active_diag.title() if active_diag else 'Brain MRI Review'}, Area={f'{active_area:.2f} cm²' if active_area else 'N/A'}"}

Verified Medical Knowledge:
{passages_text}

Patient's Question:
{user_query}

Provide a clear, comforting, and well-structured answer explaining what this means, addressing their question directly using their scan report details.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Try LLM first
    llm_response = _call_groq_llm(messages, username=username)
    if llm_response:
        return {
            "answer": llm_response,
            "sources": sources,
            "grounded": True,
            "mode": "Groq Llama-3 (Cloud)"
        }

    # Fallback to intelligent local clinical synthesizer
    return {
        "answer": _synthesize_local_rag_answer(user_query, active_diag, active_area, retrieved, role, report_data),
        "sources": sources,
        "grounded": True,
        "mode": "Local Medical Knowledge Engine"
    }


def _synthesize_local_rag_answer(
    user_query: str,
    diagnosis_context: str,
    tumor_area_cm2: float,
    retrieved_chunks: List[Dict[str, Any]],
    role: str = "patient",
    report_data: Dict[str, Any] = None
) -> str:
    query_lower = (user_query or "").lower()


    diag_text = (diagnosis_context or "").lower()
    combined = f"{diag_text} {query_lower}"

    # Extract highlights from top retrieved knowledge base chunks
    chunk_snippets = []
    for c in (retrieved_chunks or [])[:2]:
        first_few_lines = "\n".join([line for line in c["text"].split("\n") if line.strip() and not line.startswith("#")][:3])
        if first_few_lines:
            chunk_snippets.append(f"> **[{c['category']} Reference]**\n> {first_few_lines}")
    ref_block = "\n\n".join(chunk_snippets) if chunk_snippets else ""

    # Build Report Context Header if available
    report_summary_header = ""

    if report_data:
        p_name = report_data.get("patient_name") or "Patient"
        diag_val = report_data.get("diagnosis", diagnosis_context or "Scan Analyzed")
        conf_pct = report_data.get("confidence_pct")
        area_cm = report_data.get("area_cm2", tumor_area_cm2)
        shape_lbl = report_data.get("shape_label")
        conf_txt = f" ({conf_pct:.1f}% Confidence)" if conf_pct else ""
        area_txt = f"• **Tumor Surface Area**: `{area_cm:.2f} cm²` | **Shape**: `{shape_lbl or 'Assessed'}`\n\n" if area_cm else "\n\n"
        report_summary_header = (
            f"**📊 Active Scan Finding ({p_name})**:\n"
            f"• **Primary Finding**: `{diag_val.upper()}`{conf_txt}\n"
            f"{area_txt}"
        )

    # Direct Report Explanation Queries
    if any(phrase in query_lower for phrase in ["my report", "my scan", "my result", "my diagnosis", "tumor area", "explain my", "what is my"]):
        if report_data:
            p_name = report_data.get("patient_name") or "Patient"
            diag_val = report_data.get("diagnosis", diagnosis_context or "Scan Analyzed")
            conf_pct = report_data.get("confidence_pct", 0)
            area_cm = report_data.get("area_cm2", tumor_area_cm2)
            shape_lbl = report_data.get("shape_label", "Standard")
            circ = report_data.get("circularity", 0)
            focus = report_data.get("gradcam_focus", "Central region")

            if role == "doctor":
                return (
                    f"### 📋 Diagnostic Summary for Patient: {p_name}\n\n"
                    f"• **Classification**: **{diag_val.upper()}** ({conf_pct:.1f}% model confidence)\n"
                    f"• **Surface Area**: **{area_cm:.2f} cm²** (cross-sectional segmentation)\n"
                    f"• **Morphometry**: {shape_lbl} (Circularity index: {circ:.2f})\n"
                    f"• **Attention Region**: {focus}\n\n"
                    f"**Recommended Clinical Action Plan**:\n"
                    f"1. Multidisciplinary Neuro-Oncology Tumor Board review.\n"
                    f"2. Pre-operative contrast-enhanced volumetric MRI (T1+Gd, T2/FLAIR, DTI tractography if near eloquent pathways).\n"
                    f"3. Molecular profiling panel (IDH1/2 mutation, MGMT promoter methylation status, 1p/19q codeletion).\n\n"
                    f"{ref_block}"
                )
            else:
                area_line = f"• **Tumor Size / Area**: The segmented region measures approximately **{area_cm:.2f} cm²**.\n" if area_cm else ""
                shape_line = f"• **Shape Analysis**: The tumor contour is categorized as **{shape_lbl}**.\n\n" if shape_lbl else "\n"
                return (
                    f"### 🧠 Summary of Your MRI Scan Results\n\n"
                    f"Hello **{p_name}**, here is an easy-to-understand explanation of your recent scan analysis:\n\n"
                    f"• **AI Finding**: The scan showed patterns of **{diag_val.title()}** with **{conf_pct:.1f}% AI confidence**.\n"
                    f"{area_line}"
                    f"{shape_line}"
                    f"**What does this mean for you?**\n"
                    f"• This AI analysis provides helpful measurements for your healthcare team.\n"
                    f"• Your neurologist or neurosurgeon will review your complete scans alongside your overall health to decide if observation, medication, or surgery is best.\n"
                    f"• Please keep your scheduled appointment and bring your downloadable PDF report with you."
                )


    if role == "doctor":
        if any(w in combined for w in ["glioma", "stupp", "glioblastoma", "astrocytoma", "oligodendroglioma", "temozolomide", "radiation"]):
            area_str = f" `{tumor_area_cm2:.2f} cm²`" if tumor_area_cm2 else " (Pending scan segmentation)"
            return (
                f"{report_summary_header}"
                f"### 📋 Clinical Protocol: Glioma & Glioblastoma Management (WHO CNS 5)\n\n"
                f"• **Surface / Volumetric Burden**:{area_str}\n"
                f"• **Molecular Biomarker Diagnostic Panel**:\n"
                f"  - **IDH1/IDH2 Mutation**: Mandatory for distinguishing Astrocytoma IDH-mutant from Glioblastoma IDH-wildtype.\n"
                f"  - **MGMT Promoter Methylation**: Predictive biomarker for Temozolomide alkylating chemotherapy sensitivity.\n"
                f"  - **1p/19q Whole-Arm Codeletion**: Diagnostic for Oligodendroglioma.\n"
                f"  - **TERT Promoter / EGFR Amplification / +7/-10**: Defines molecular Glioblastoma (WHO Grade 4).\n"
                f"• **Surgical Considerations**: Maximal safe resection (GTR) utilizing 5-ALA fluorescence guidance and intraoperative cortical/subcortical mapping.\n"
                f"• **Adjuvant Protocol**: Stupp Regimen (60 Gy in 30 fractions with concurrent daily Temozolomide 75 mg/m², followed by 6 cycles of adjuvant TMZ 150-200 mg/m²).\n\n"
                f"{ref_block}"
            )
        elif any(w in combined for w in ["meningioma", "simpson", "dural", "arachnoid", "gamma knife"]):
            return (
                f"{report_summary_header}"
                f"### 📋 Clinical Protocol: Meningioma Management & Resection Grading\n\n"
                f"• **Pathology**: Non-glial extra-axial arachnoid cap neoplasm (~80% WHO Grade 1 benign, 15-20% Grade 2 atypical, 1-3% Grade 3 anaplastic).\n"
                f"• **Simpson Resection Classification**:\n"
                f"  - **Grade I**: Complete tumor resection + removal of affected dura & abnormal bone.\n"
                f"  - **Grade II**: Complete tumor resection + coagulation of dural attachment.\n"
                f"  - **Grade III**: Macroscopic complete resection without dural coagulation/excision.\n"
                f"  - **Grade IV/V**: Subtotal resection or decompression.\n"
                f"• **Radiosurgery (SRS / Gamma Knife)**: 12–16 Gy margin dose for skull base / cavernous sinus meningiomas or unresectable Simpson IV/V remnants.\n"
                f"• **Active Surveillance**: Incidental, asymptomatic small lesions (<2.5 cm) observed with MRI at 3, 6, and 12 months.\n\n"
                f"{ref_block}"
            )
        elif any(w in combined for w in ["pituitary", "prolactin", "pitnet", "macroadenoma", "chiasm", "transsphenoidal"]):
            return (
                f"{report_summary_header}"
                f"### 📋 Clinical Protocol: Pituitary Neuroendocrine Tumors (PitNET)\n\n"
                f"• **Endocrine Diagnostic Workup**: Serum Prolactin, IGF-1, 24-hr UFC / 1mg DST, Morning Serum Cortisol & ACTH, Free T4 & TSH, LH/FSH, Testosterone/Estradiol.\n"
                f"• **Neuro-Ophthalmology Evaluation**: Automated Humphrey 24-2 visual field perimetry to assess optic chiasm compression / bitemporal hemianopsia.\n"
                f"• **First-Line Management Strategies**:\n"
                f"  - **Prolactinoma**: Medical dopamine agonist therapy (Cabergoline 0.25–0.5 mg 1–2x/week).\n"
                f"  - **Macroadenoma with Chiasmal Compression / Non-Prolactinoma**: Endoscopic Endonasal Transsphenoidal Surgery (EETS).\n"
                f"  - **ACTH-Secreting (Cushing Disease)**: Surgical resection followed by cortisol monitoring for remission.\n\n"
                f"{ref_block}"
            )
        elif any(w in combined for w in ["idh", "mgmt", "biomarker", "molecular", "gene", "mutation"]):
            return (
                f"{report_summary_header}"
                f"### 📋 Molecular Neuro-Oncology Biomarkers Reference\n\n"
                f"1. **IDH1 R132H & IDH2**: Defines standard glioma molecular lineage. IDH-mutant gliomas carry significantly better overall prognosis than IDH-wildtype.\n"
                f"2. **1p/19q Co-deletion**: Essential diagnostic criterion for Oligodendroglioma (WHO Grade 2/3).\n"
                f"3. **MGMT Promoter Methylation**: DNA repair enzyme silencing; strong predictive indicator of Temozolomide chemosensitivity.\n"
                f"4. **CDKN2A/B Homotrzous Deletion**: Indicates WHO Grade 4 behavior even in histological Grade 2/3 IDH-mutant astrocytomas.\n\n"
                f"{ref_block}"
            )
        else:
            return (
                f"{report_summary_header}"
                f"### 📋 NeuroScan Clinical Decision Support\n\n"
                f"Based on the clinical query and retrieved neuro-oncology guidelines:\n\n"
                f"{ref_block}\n\n"
                f"• **Standard Pathway**: Correlate multiplanar MRI (T1+Gd, T2, FLAIR, ADC) with patient neuro-exam and histological/molecular biomarkers."
            )
    else:
        # Patient-Friendly Guidance & Care Maintenance
        if any(w in combined for w in ["remedy", "remedies", "diet", "nutrition", "maintain", "maintenance", "lifestyle", "food", "exercise", "care", "routine"]):
            return (
                f"{report_summary_header}"
                f"### 🥗 Health Maintenance & Supportive Daily Care\n\n"
                f"Here is practical, evidence-based guidance to support your brain health, daily energy, and recovery:\n\n"
                f"1. **Anti-Inflammatory Diet & Nutrition**:\n"
                f"   - Focus on antioxidant-rich whole foods: blueberries, dark leafy greens (spinach, kale), walnuts, flaxseeds, and Omega-3 fatty acids (salmon, avocados).\n"
                f"   - Limit refined sugars, excessive caffeine, and ultra-processed foods that trigger neuro-inflammation.\n\n"
                f"2. **Hydration & Sleep Glymphatic Cleansing**:\n"
                f"   - Drink 2.0 to 2.5 liters of clean water daily to optimize cerebral circulation and reduce fatigue.\n"
                f"   - Aim for 7–9 hours of deep sleep. The brain's natural lymphatic cleansing system operates primarily during restorative deep sleep.\n\n"
                f"3. **Supportive Home Remedies for Common Symptoms**:\n"
                f"   - **Headaches / Tension**: Use cold gel compresses applied to the forehead or back of the neck for 15 minutes in a quiet, dimmed room.\n"
                f"   - **Managing Fatigue**: Pace your daily schedule—tackle important tasks during morning peak energy and take short 20-minute rest breaks.\n"
                f"   - **Calm & Stress Relief**: Practice 10 minutes of daily diaphragmatic breathing or gentle meditation to reduce cortisol levels.\n\n"
                f"4. **Daily Safety & Activity Precautions**:\n"
                f"   - Engage in low-impact daily movement like 20–30 minutes of gentle walking or stretching.\n"
                f"   - Avoid heavy weightlifting or high-impact contact sports until cleared by your doctor.\n"
                f"   - If you have experienced twitching, dizziness, or seizures, avoid driving or unsupervised swimming.\n\n"
                f"💡 *Remember: These supportive remedies complement—but do not replace—your physician's personalized care plan.*"
            )
        elif any(w in combined for w in ["red flag", "emergency", "warning", "danger", "urgent"]):
            return (
                f"### 🚨 Warning Signs & When to Seek Urgent Medical Care\n\n"
                f"If you or a loved one experience any of the following **red flag symptoms**, seek emergency medical evaluation immediately:\n\n"
                f"1. **New or sudden seizures / convulsions**\n"
                f"2. **Severe, thunderclap headache**, or headache worsening progressively in the early morning\n"
                f"3. **Sudden weakness, numbness, or loss of balance** on one side of your face or body\n"
                f"4. **Acute vision changes**, such as double vision or loss of peripheral vision\n"
                f"5. **Sudden confusion, speech difficulty, or severe personality changes**\n\n"
                f"⚠️ *Always dial your local emergency number (e.g. 911 / 112) or go to the nearest emergency department.*"
            )
        elif any(w in combined for w in ["question", "ask my doctor", "consult", "doctor"]):
            return (
                f"### 🩺 Key Questions to Ask Your Specialist\n\n"
                f"Here are helpful, recommended questions to bring to your neurology or oncology appointment:\n\n"
                f"1. **Diagnosis & Nature**: *What specific type of tumor or lesion is suspected on my MRI? Is it benign or requiring treatment?*\n"
                f"2. **Treatment Options**: *What are the recommended next steps—active observation, medication, surgery, or targeted radiation?*\n"
                f"3. **Surgical Details**: *If surgery is recommended, what is the goal of the surgery and expected recovery timeline?*\n"
                f"4. **Biomarker Testing**: *Will my tumor tissue undergo genetic/biomarker testing (like IDH or MGMT)?*\n"
                f"5. **Daily Life & Activities**: *Are there any restrictions on driving, working, exercise, or air travel?*"
            )
        elif any(w in combined for w in ["glioma", "glioblastoma"]):
            area_str = f"localized tumor area of **{tumor_area_cm2:.2f} cm²**" if tumor_area_cm2 else "brain tissue change"
            return (
                f"{report_summary_header}"
                f"### 🧠 Understanding Gliomas & Daily Care\n\n"
                f"A **glioma** is a growth that develops from glial cells (supportive cells in the brain). "
                f"Our AI system analyzed your scan and detected a {area_str}.\n\n"
                f"**Key Care & Next Steps**:\n"
                f"• **Specialist Consultation**: Schedule a prompt consultation with a neurosurgeon and neuro-oncologist to review your multiplanar MRI.\n"
                f"• **Daily Health Log**: Keep a daily log of symptoms, energy levels, and any headaches.\n"
                f"• **Seizure Safety**: If you experience twitching or sensory shifts, let family members know basic first-aid steps (stay calm, roll to side, call doctor).\n"
                f"• **Anti-Inflammatory Nutrition**: Support your body with antioxidant whole foods, clean hydration, and adequate restorative sleep."
            )
        elif any(w in combined for w in ["meningioma", "benign"]):
            return (
                f"{report_summary_header}"
                f"### 🧠 Understanding Meningiomas & Watchful Observation\n\n"
                f"A **meningioma** is a growth on the outer membranes (the meninges) that surround the brain and spinal cord.\n\n"
                f"**Important Health Facts & Maintenance**:\n"
                f"• **Over 80% are completely benign (non-cancerous)** and develop very slowly over many years.\n"
                f"• **Observation Routine**: Many small meningiomas only require regular check-up MRIs (every 6 to 12 months) to confirm stability without surgery.\n"
                f"• **Daily Wellness**: Support healthy blood vessels with light cardiovascular walking, low-sodium nutrition, and stress management.\n"
                f"• **High Success Rate**: If medical treatment or surgical removal is ever needed, modern methods have excellent outcomes."
            )
        elif any(w in combined for w in ["pituitary", "hormone"]):
            return (
                f"{report_summary_header}"
                f"### 🧠 Understanding Pituitary Growths & Hormonal Wellness\n\n"
                f"The **pituitary gland** is located at the base of the brain and regulates your body's vital hormone systems.\n\n"
                f"**Key Maintenance Points**:\n"
                f"• Almost all pituitary growths are **benign (non-cancerous)**.\n"
                f"• **Endocrine Checks**: Your doctor will likely schedule routine blood tests for hormone balance and a standard eye checkup.\n"
                f"• **Daily Balance**: Maintain consistent meal times with balanced protein and complex carbs to avoid blood sugar fluctuations.\n"
                f"• **Gentle Solutions**: Many pituitary conditions respond effectively to simple daily medications or gentle, minimally invasive endoscopic care."
            )
        else:
            return (
                f"{report_summary_header}"
                f"### 🧠 NeuroScan Patient Health Guidance\n\n"
                f"Thank you for your question. Here is verified medical guidance to support your brain wellness:\n\n"
                f"{ref_block}\n\n"
                f"💡 **Tip**: Share your MRI report and these questions with your neurologist or neurosurgeon for personalized medical recommendations."
            )




# Auto-initialize on module load
init_rag_engine()

