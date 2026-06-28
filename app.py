import os
import streamlit as st
import time
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_core.documents import Document
import tiktoken
from fpdf import FPDF
import requests
from urllib.parse import quote

# --- LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 2. CREDENTIALS ---
INDEX_NAME = "pdf-agent"

def get_secret(key: str) -> str:
    try:
        return st.secrets[key]
    except KeyError:
        return ""

pinecone_key = get_secret("PINECONE_API_KEY")
if not pinecone_key:
    st.error(
        "⚠️ **PINECONE_API_KEY missing!**\n\n"
        "Go to: **Streamlit Cloud → Your App → Settings → Secrets** and add:\n"
        "```\nPINECONE_API_KEY = 'your-free-key'\n```"
    )
    st.stop()
os.environ["PINECONE_API_KEY"] = pinecone_key

# --- 3. LLM PROVIDER CONFIG (ENHANCED) ---
LLM_PROVIDERS = {
    "🟦 Google Gemini Flash": {
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-flash-8b"],
        "secret_key": "GEMINI_API_KEY",
        "type": "gemini",
        "color": "#1A73E8"
    },
    "🟣 Groq High-Speed Engine": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        "secret_key": "GROQ_API_KEY",
        "type": "groq",
        "color": "#8A2BE2"
    }
}

# --- ANALYSIS MODES CONFIG ---
ANALYSIS_MODES = {
    "📚 In-depth Knowledge (Database)": {
        "description": "Full detailed knowledge from your document database",
        "requires_online": False,
        "requires_database": True
    },
    "📄 Summary (Database)": {
        "description": "Concise summary from your document database",
        "requires_online": False,
        "requires_database": True
    },
    "📚 In-depth + Online": {
        "description": "Detailed database knowledge combined with online research",
        "requires_online": True,
        "requires_database": True
    },
    "🌐 Online Only": {
        "description": "Answer sourced entirely from online search",
        "requires_online": True,
        "requires_database": False
    },
    "📋 SOP Style": {
        "description": "Format answer as Standard Operating Procedure (PURPOSE, SCOPE, PROCEDURE, etc.)",
        "requires_online": False,
        "requires_database": True
    },
    "📊 Summary + Online": {
        "description": "Concise summary combining database findings and online research",
        "requires_online": True,
        "requires_database": True
    },
    "✅ Regulatory Audit Checklist": {
        "description": "Compliance checklist with cross-references to USP/BP/Ph.Eur./ICH standards",
        "requires_online": False,
        "requires_database": True
    }
}

# --- 4. PAGE CONFIG ---
st.set_page_config(
    page_title="PharmAgent Analytics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Premium Corporate Layout
st.markdown("""
    <style>
        .reportview-container .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1.5rem;
        }
        .stButton button {
            border-radius: 6px;
        }
        .sop-section {
            border-left: 4px solid #1A73E8;
            padding-left: 15px;
            margin: 15px 0;
        }
        .sop-number {
            font-weight: bold;
            color: #1A73E8;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 PharmAgent Pro — Enterprise Reference RAG Platform")
st.caption("Persistent Document Intelligence & Multi-Source Audit Workspace")
st.markdown("---")

# --- 5. CACHED RESOURCES ---
@st.cache_resource
def init_vector_db():
    try:
        api_key = get_secret("PINECONE_API_KEY")
        if not api_key:
            st.error("PINECONE_API_KEY missing")
            return None

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                "device":"cpu"
            },
            encode_kwargs={
                "normalize_embeddings":True
            }
        )

        pc = Pinecone(
            api_key=api_key
        )

        index = pc.Index(
            INDEX_NAME
        )

        logger.info("✓ Pinecone initialized")

        return {
            "index":index,
            "embeddings":embeddings
        }

    except Exception as e:
        st.error(
            f"Pinecone initialization failed: {e}"
        )
        return None

@st.cache_resource
def get_tokenizer():
    try:
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        class FallbackTokenizer:
            def encode(self, text):
                return [0] * (len(text) // 4)
        return FallbackTokenizer()

db = init_vector_db()
tokenizer = get_tokenizer()

# --- 6. LLM CLIENT FACTORIES ---
def get_gemini_client():
    from google import genai
    key = get_secret("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY missing.")
    return genai.Client(api_key=key)

def get_groq_client():
    from groq import Groq
    key = get_secret("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY missing.")
    return Groq(api_key=key)

# --- 7. WEB SEARCH FUNCTION ---
def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using DuckDuckGo-like approach.
    Returns list of {title, link, snippet}
    """
    try:
        # Using a simple web search approach
        search_url = f"https://api.search.brave.com/res/v1/web/search?q={quote(query)}&count={max_results}"
        headers = {"Accept": "application/json"}
        
        # For this implementation, we'll use a fallback approach
        # In production, use your preferred search API (Google, Brave, etc.)
        return search_fallback(query, max_results)
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []

def search_fallback(query: str, max_results: int = 5) -> List[Dict]:
    """Fallback web search using requests"""
    try:
        # Using DuckDuckGo via requests (no API key needed for basic use)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Note: In production, integrate with Brave API, Google Search API, or similar
        # For now, returning structured format that can be enhanced
        return []
    except Exception as e:
        logger.warning(f"Search fallback failed: {e}")
        return []

# --- 8. PERSISTENT CHAT HISTORY INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show past analysis sheets seamlessly
for idx, msg in enumerate(st.session_state.messages):
    role_icon = "📝" if msg["role"] == "user" else "🔬"
    with st.chat_message(msg["role"], avatar=role_icon):
        st.markdown(msg["content"])
        if "metadata" in msg:
            meta = msg["metadata"]
            st.caption(
                f"⏱️ Generation Time: {meta.get('elapsed_time')} | "
                f"Input Tokens: {meta.get('input_tokens')} | "
                f"Output Tokens: {meta.get('output_tokens')} | "
                f"Mode: {meta.get('mode', 'N/A')}"
            )

# --- 9. STREAMING HELPER LOGIC ---
SYSTEM_PROMPTS = {
    "default": (
        "You are an expert pharmaceutical data extraction engine, lead regulatory compliance auditor, and expert QC document writer. "
        "Provide precise, structured, audit-ready responses with exact LaTeX formatting for mathematical expressions."
    ),
    "sop": (
        "You are an expert pharmaceutical SOP writer and regulatory compliance specialist. "
        "Format all responses as Standard Operating Procedures (SOP) with clear sections: PURPOSE, SCOPE, RESPONSIBILITIES, "
        "TRAINING REQUIREMENTS, ASSOCIATED DOCUMENTS, ABBREVIATIONS AND DEFINITIONS, PRECAUTIONS, PROCEDURE, REFERENCES, and APPENDICES. "
        "Use numbered lists and hierarchical structure like pharmaceutical industry standards."
    ),
    "audit": (
        "You are an advanced pharmaceutical regulatory compliance auditor. "
        "Create comprehensive compliance checklists cross-referenced with USP/BP/Ph.Eur./ICH standards. "
        "Include status indicators (✅ Compliant / ⚠️ Partial / ❌ Non-Compliant) for each item with detailed reference citations."
    )
}

def _stream_gemini(model: str, prompt: str, placeholder) -> str:
    client = get_gemini_client()
    response_text = ""
    try:
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPTS["default"],
            temperature=0.1
        )
        for chunk in client.models.generate_content_stream(model=model, contents=prompt, config=config):
            if chunk.text:
                response_text += chunk.text
                placeholder.markdown(response_text + "▌")
    except Exception:
        response = client.models.generate_content(model=model, contents=prompt)
        response_text = response.text
    placeholder.markdown(response_text)
    return response_text

def _stream_openai_compat(client, model: str, prompt: str, placeholder, system_prompt: str = None) -> str:
    response_text = ""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPTS["default"]
    
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}
        ],
        stream=True,
        max_tokens=4096,
        temperature=0.1
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            delta = chunk.choices[0].delta.content
            response_text += delta
            placeholder.markdown(response_text + "▌")
    placeholder.markdown(response_text)
    return response_text

def _no_stream_openai_compat(client, model: str, prompt: str, system_prompt: str = None) -> str:
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPTS["default"]
    
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=4096,
        temperature=0.1
    )
    return resp.choices[0].message.content

def generate_response(provider_name: str, model: str, prompt: str, stream: bool, placeholder, 
                     system_prompt: str = None, mode: str = None) -> Tuple[str, Dict]:
    delays = [2, 4, 8, 16, 32]
    max_retries = 5
    provider_type = LLM_PROVIDERS[provider_name]["type"]

    if system_prompt is None:
        system_prompt = SYSTEM_PROMPTS["default"]

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response_text = ""

            if provider_type == "gemini":
                if stream:
                    response_text = _stream_gemini(model, prompt, placeholder)
                else:
                    client = get_gemini_client()
                    from google.genai import types
                    config = types.GenerateContentConfig(system_instruction=system_prompt, temperature=0.1)
                    response = client.models.generate_content(model=model, contents=prompt, config=config)
                    response_text = response.text
                    placeholder.markdown(response_text)
            elif provider_type == "groq":
                client = get_groq_client()
                if stream:
                    response_text = _stream_openai_compat(client, model, prompt, placeholder, system_prompt)
                else:
                    response_text = _no_stream_openai_compat(client, model, prompt, system_prompt)
                    placeholder.markdown(response_text)

            elapsed = time.time() - start_time
            input_tokens  = len(tokenizer.encode(prompt))
            output_tokens = len(tokenizer.encode(response_text))

            metadata = {
                "provider":       provider_name,
                "model":          model,
                "attempt":        attempt + 1,
                "elapsed_time":   f"{elapsed:.2f}s",
                "input_tokens":   input_tokens,
                "output_tokens":  output_tokens,
                "total_tokens":   input_tokens + output_tokens,
                "mode":           mode or "Standard"
            }

            logger.info(f"✓ [{provider_name}] {output_tokens} tokens in {elapsed:.2f}s | Mode: {mode}")
            return response_text, metadata

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {err_str}")
            if "rate" in err_str.lower() or "429" in err_str:
                wait = delays[attempt] * 2
                st.warning(f"⏳ Rate limit hit. Waiting {wait}s before retry...")
            elif attempt < max_retries - 1:
                wait = delays[attempt]
                st.warning(f"⏳ Retrying in {wait}s… (Attempt {attempt + 2}/{max_retries})")
            else:
                raise
            time.sleep(wait if "rate" in err_str.lower() or "429" in err_str else delays[attempt])

# --- 10. CONTEXT & PROMPT BUILDERS ---
def assemble_context(docs: List) -> Tuple[str, List[Dict]]:
    blocks, refs = [], []
    for idx, d in enumerate(docs, 1):
        source = d.metadata.get('source', 'Unknown').split('/')[-1]
        page   = d.metadata.get('page', 0) + 1
        blocks.append(f"[{idx}. 📄 {source} | Page {page}]\n{d.page_content}")
        refs.append({"index": idx, "source": source, "page": page})
    return "\n\n--- Document Chunk ---\n\n".join(blocks), refs

def build_prompt_indepth_db(user_query: str, context: str, search_depth: int) -> str:
    """In-depth knowledge from database"""
    return f"""You are a pharmaceutical specialist with access to internal documentation.

USER QUERY: {user_query}

ANALYSIS TYPE: In-depth Database Knowledge Extraction
CHUNKS ANALYZED: {search_depth}
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTIONS:
1. Provide COMPLETE, detailed information extracted directly from the provided documents
2. Include ALL relevant details, specifications, parameters, and procedures
3. Use inline citations: [Source: filename | Page: N]
4. Organize with clear sections and hierarchies
5. Preserve exact numerical values, specifications, and technical details
6. Use LaTeX for mathematical expressions: $inline$ and $$block$$
7. Format tables in Markdown
8. End with "📋 REFERENCE METADATA" section listing:
   - Documents used with page numbers
   - Key parameters verified
   - Standards referenced (USP/BP/Ph.Eur./ICH)

DOCUMENT CONTEXT:
{context}

DETAILED ANALYSIS:"""

def build_prompt_summary_db(user_query: str, context: str, search_depth: int) -> str:
    """Summary from database"""
    return f"""You are a pharmaceutical data analyst specializing in concise summaries.

USER QUERY: {user_query}

ANALYSIS TYPE: Database Summary
CHUNKS ANALYZED: {search_depth}
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTIONS:
1. Synthesize key findings from documents in a concise format
2. Focus on the most critical information and decision points
3. Structure as: Overview → Key Findings → Critical Parameters → Conclusions
4. Include citations: [Source: filename | Page: N]
5. Keep paragraphs focused and direct
6. Use bullet points for key takeaways
7. Limit to essential information only
8. End with "📋 REFERENCE METADATA" section

DOCUMENT CONTEXT:
{context}

EXECUTIVE SUMMARY:"""

def build_prompt_sop_style(user_query: str, context: str, search_depth: int) -> str:
    """SOP Style formatting"""
    return f"""You are an expert pharmaceutical SOP writer.

USER QUERY: {user_query}

ANALYSIS TYPE: Standard Operating Procedure (SOP) Format
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FORMAT YOUR RESPONSE WITH THESE SECTIONS (Numbered):
1. PURPOSE
   - Clear objective statement

2. SCOPE
   - Applicability and boundaries

3. RESPONSIBILITIES
   - Personnel roles (numbered list)

4. TRAINING REQUIREMENTS
   - Required training for personnel

5. ASSOCIATED DOCUMENTS
   - Reference documents and SOPs

6. ABBREVIATIONS AND DEFINITIONS
   - Format as: TERM : Definition

7. PRECAUTIONS
   - Safety and quality measures (numbered list)

8. PROCEDURE
   - Main steps with hierarchy (8.1, 8.1.1, 8.1.1.1 format)

9. REFERENCES
   - Standards: USP, BP, Ph.Eur., ICH, etc.

10. APPENDICES
    - Supporting documents and forms

FORMATTING RULES:
- Use numbered sections with underlines for main sections
- Use hierarchical numbering (8.1, 8.1.1, 8.1.1.1)
- Include [Source: filename | Page: N] citations
- Use tables for complex information
- Preserve exact specifications from documents
- Use LaTeX for chemical formulas and math

DOCUMENT CONTEXT:
{context}

STANDARD OPERATING PROCEDURE:"""

def build_prompt_indepth_online(user_query: str, db_context: str, online_context: str = None, search_depth: int = 12) -> str:
    """In-depth knowledge + online"""
    online_section = ""
    if online_context:
        online_section = f"\nONLINE RESEARCH FINDINGS:\n{online_context}"
    
    return f"""You are a pharmaceutical specialist with access to both proprietary documents and external research.

USER QUERY: {user_query}

ANALYSIS TYPE: In-depth Database + Online Research
DATABASE CHUNKS: {search_depth}
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTIONS:
1. Synthesize information from BOTH database and online sources
2. Clearly distinguish between internal database findings and external research
3. Start with database information, then add complementary online insights
4. Include ALL relevant details with proper citations:
   - Database: [Database: filename | Page: N]
   - Online: [Online: source title]
5. Highlight agreements and discrepancies between sources
6. Use sections: Database Findings → Online Research → Integrated Analysis
7. Use LaTeX for chemical expressions
8. Format as: Details → [Citations] → Implications
9. End with comprehensive metadata

DATABASE CONTEXT:
{db_context}
{online_section}

COMPREHENSIVE ANALYSIS:"""

def build_prompt_online_only(user_query: str) -> str:
    """Online only"""
    return f"""You are a pharmaceutical research analyst with access to online resources.

USER QUERY: {user_query}

ANALYSIS TYPE: Online Research Only
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTIONS:
1. Provide comprehensive answer based on current online research
2. Focus on peer-reviewed sources, official guidelines, and authoritative databases
3. Include relevant standards: USP, BP, Ph.Eur., ICH, FDA, EMA
4. Cite sources: [Source: organization/publication title]
5. Structure: Overview → Current Standards → Best Practices → References
6. Include relevant specifications, procedures, and technical details
7. Note publication dates for time-sensitive information
8. Use LaTeX for chemical expressions
9. End with "📚 SOURCES REFERENCED" section

ANALYTICAL RESPONSE:"""

def build_prompt_summary_online(user_query: str, db_summary: str = None, online_summary: str = None) -> str:
    """Summary + online"""
    db_section = f"DATABASE SUMMARY:\n{db_summary}\n\n" if db_summary else ""
    online_section = f"ONLINE RESEARCH SUMMARY:\n{online_summary}\n\n" if online_summary else ""
    
    return f"""You are a pharmaceutical analyst specializing in concise synthesis.

USER QUERY: {user_query}

ANALYSIS TYPE: Database + Online Summary
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTIONS:
1. Combine database and online findings into ONE concise summary
2. Extract only critical information
3. Structure: Key Points → Standards Alignment → Practical Implications
4. Keep each section to 2-3 sentences maximum
5. Include citations: [Database: file] [Online: source]
6. Use bullet format for takeaways
7. End with "📋 REFERENCES" section

{db_section}{online_section}
EXECUTIVE SUMMARY:"""

def build_prompt_audit_checklist(user_query: str, context: str, search_depth: int) -> str:
    """Regulatory audit checklist with references"""
    return f"""You are a pharmaceutical regulatory compliance auditor.

USER QUERY: {user_query}

ANALYSIS TYPE: Regulatory Audit Checklist with References
CHUNKS ANALYZED: {search_depth}
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTIONS:
1. Create comprehensive compliance checklist
2. Cross-reference with USP/BP/Ph.Eur./ICH/FDA standards
3. For EACH item, include:
   - Requirement statement
   - Status: ✅ Compliant | ⚠️ Partial | ❌ Non-Compliant
   - Reference: [Document: filename | Page: N]
   - Finding details
   - Remediation (if needed)

4. Format as table:
   | Requirement | Status | Reference | Finding | Remediation |
   |---|---|---|---|---|

5. Include sections:
   - Documentation Requirements
   - Technical Specifications
   - Quality Standards
   - Training & Competency
   - Audit & Monitoring
   - Corrective Actions

6. Use visual indicators: ✅/⚠️/❌
7. Flag critical gaps with 🚨
8. Include applicability standard for each item

DOCUMENT CONTEXT:
{context}

AUDIT CHECKLIST:"""

# --- 11. HIGH-FIDELITY PDF GENERATOR ENGINE (ENHANCED) ---
class ReportPDF(FPDF):
    def __init__(self, title_text="PHARMACEUTICAL COMPLIANCE REPORT"):
        super().__init__()
        self.title_text = title_text

    def header(self):
        self.set_fill_color(26, 115, 232)
        self.rect(0, 0, 210, 8, 'F')
        
        self.set_y(12)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, "PHARMAGENT PRO - MULTI-SOURCE COMPLIANCE INTELLIGENCE ENGINE", 0, 0, "L")
        self.cell(0, 5, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 0, 1, "R")
        self.set_draw_color(200, 200, 200)
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y() - 2, 200, self.get_y() - 2)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, "STRICTLY CONFIDENTIAL - INTERNAL LAB USE ONLY", 0, 0, "L")
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "R")

def clean_for_pdf_latin1(text: str) -> str:
    replacements = {
        "\\mu": "u", "\\text": "", "\\le": "<=", "\\ge": ">=", "\\pm": "+/-",
        "\\times": "x", "\\%": "%", "$": "", "µ": "u", "≤": "<=", "≥": ">=",
        "±": "+/-", "°": " degrees ", "'": "'", """: '"', """: '"', "–": "-",
        "—": "-", "✅": "[PASS]", "❌": "[FAIL]", "⚠️": "[PARTIAL]", "🚨": "[CRITICAL]",
        "📋": "", "📝": "", "🔬": "", "🧪": "", "📊": "", "🚀": "", "⏳": "", "⏱️": "",
        "📥": "", "📤": "", "💚": "", "🛡️": "", "📚": "", "🌐": ""
    }
    cleaned = text
    for target, rep in replacements.items():
        cleaned = cleaned.replace(target, rep)
    cleaned = cleaned.replace("{", "").replace("}", "")
    return cleaned.encode('latin-1', 'ignore').decode('latin-1')

def generate_pdf_report(markdown_content: str, query: str, mode: str = "Standard") -> bytes:
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_margins(12, 20, 12)
    
    pdf.set_y(22)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(26, 115, 232)
    
    title = f"{mode.upper()} ANALYSIS REPORT" if mode != "Standard" else "PHARMACEUTICAL ANALYSIS REPORT"
    pdf.multi_cell(0, 8, title)
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Analysis Mode: {mode}", 0, 1)
    pdf.cell(0, 6, f"Query: '{query[:80]}...'", 0, 1)
    pdf.ln(4)
    
    lines = markdown_content.split('\n')
    in_table = False
    table_headers = []
    table_rows = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                draw_pdf_table(pdf, table_headers, table_rows)
                in_table = False
                table_headers = []
                table_rows = []
            pdf.ln(2)
            continue
            
        if stripped.startswith('|'):
            if '---' in stripped:
                continue
            parts = [clean_for_pdf_latin1(x.strip()) for x in stripped.split('|')[1:-1]]
            if not in_table:
                in_table = True
                table_headers = parts
            else:
                table_rows.append(parts)
            continue
        else:
            if in_table:
                draw_pdf_table(pdf, table_headers, table_rows)
                in_table = False
                table_headers = []
                table_rows = []
        
        if stripped.startswith('# '):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(26, 115, 232)
            pdf.multi_cell(0, 8, clean_for_pdf_latin1(stripped[2:]))
            pdf.ln(1)
        elif stripped.startswith('## '):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(0, 7, clean_for_pdf_latin1(stripped[3:]))
            pdf.ln(1)
        elif stripped.startswith('### '):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(0, 6, clean_for_pdf_latin1(stripped[4:]))
            pdf.ln(1)
        elif stripped.startswith('* ') or stripped.startswith('- '):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(16)
            pdf.multi_cell(0, 5, f"- {clean_for_pdf_latin1(stripped[2:])}")
        elif stripped[0].isdigit() and '.' in stripped.split()[0] if stripped.split() else False:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.set_x(16)
            pdf.multi_cell(0, 5, clean_for_pdf_latin1(stripped))
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 5, clean_for_pdf_latin1(stripped))
            
    if in_table and table_headers:
        draw_pdf_table(pdf, table_headers, table_rows)

    return pdf.output()

def draw_pdf_table(pdf, headers, rows):
    if not headers:
        return
    num_cols = len(headers)
    col_width = (210 - 24) / num_cols
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(30, 41, 59)
    for header in headers:
        pdf.cell(col_width, 6, clean_for_pdf_latin1(header), 1, 0, 'C', True)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(60, 60, 60)
    for row in rows:
        while len(row) < num_cols:
            row.append("")
        for cell in row[:num_cols]:
            pdf.cell(col_width, 6, clean_for_pdf_latin1(cell), 1, 0, 'L')
        pdf.ln()
    pdf.ln(2)

# --- 12. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Settings")

    provider_name = st.selectbox(
        "LLM Provider:",
        list(LLM_PROVIDERS.keys())
    )
    pinfo = LLM_PROVIDERS[provider_name]
    model_choice = st.selectbox("Model Engine:", pinfo["models"])

    st.markdown("---")
    
    # Analysis Mode Selection with Info
    st.subheader("📊 Analysis Protocol Mode")
    selected_mode = st.selectbox(
        "Select Analysis Type:",
        list(ANALYSIS_MODES.keys()),
        help="Choose the type of analysis for your query"
    )
    
    mode_info = ANALYSIS_MODES[selected_mode]
    st.info(f"ℹ️ {mode_info['description']}")

    st.markdown("---")
    search_depth = st.slider(
        "Vector Scan Depth (Chunks)", 
        min_value=4, 
        max_value=20, 
        value=12,
        help="Number of document chunks to analyze"
    )

    st.markdown("---")
    stream_output  = st.toggle("🌊 Enable Streaming", value=True)
    show_tokens    = st.toggle("🔢 Show Token Usage", value=True)
    show_sources   = st.toggle("📚 Show Source Traces", value=True)

# --- 13. MAIN INTERFACE RENDERING ---
st.markdown(
    f"<div style='background:{pinfo['color']}15; border-left:4px solid {pinfo['color']};"
    f"padding:10px 16px; border-radius:6px; margin-bottom:16px;'>"
    f"<b>Engine Active:</b> {provider_name} ({model_choice}) | "
    f"<b>Mode:</b> {selected_mode}"
    f"</div>",
    unsafe_allow_html=True
)

user_query = st.text_area(
    "📝 Enter Analytical Target or Protocol Definition:",
    placeholder="e.g., What are the requirements for analytical balance validation according to pharmacopeial standards?",
    height=110
)

# --- 14. EXECUTION PROCESSOR ---
if st.button("🚀 Execute Analysis", type="primary", use_container_width=True):
    if not user_query.strip():
        st.error("❌ Please enter a query before executing.")
    else:
        progress_bar = st.progress(0)
        status_text  = st.empty()

        try:
            # Determine which mode is selected
            mode_key = selected_mode
            mode_config = ANALYSIS_MODES[mode_key]
            
            # Initialize context variables
            db_context = ""
            metadata_refs = []
            system_prompt = SYSTEM_PROMPTS["default"]

            # Step 1: Database Search (if required)
            if mode_config["requires_database"] and db is not None:
                status_text.text("Step 1/3: Scanning Pinecone Database...")
                progress_bar.progress(20)
                t0 = time.time()

                # --- FIX: THIS BLOCK IS NOW PROPERLY INDENTED ---
                query_vector = db["embeddings"].embed_query(user_query)

                results = db["index"].query(
                    vector=query_vector,
                    top_k=search_depth,
                    include_metadata=True
                )

                docs = []

                for match in results.get("matches", []):
                    metadata = match.get("metadata", {})
                    content = metadata.get("text", "")
                    
                    docs.append(
                        Document(
                            page_content=content,
                            metadata=metadata
                        )
                    )

                search_time = time.time() - t0
                # ------------------------------------------------

                if docs:
                    db_context, metadata_refs = assemble_context(docs)
                else:
                    st.warning("⚠️ No relevant documents found in database.")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
            elif mode_config["requires_database"] and db is None:
                st.error("❌ Database required for this mode but not initialized.")
                st.stop()

            # Step 2: Build appropriate prompt
            status_text.text("Step 2/3: Structuring extraction parameters...")
            progress_bar.progress(50)
            
            if mode_key == "📚 In-depth Knowledge (Database)":
                prompt = build_prompt_indepth_db(user_query, db_context, search_depth)
            elif mode_key == "📄 Summary (Database)":
                prompt = build_prompt_summary_db(user_query, db_context, search_depth)
            elif mode_key == "📋 SOP Style":
                prompt = build_prompt_sop_style(user_query, db_context, search_depth)
                system_prompt = SYSTEM_PROMPTS["sop"]
            elif mode_key == "✅ Regulatory Audit Checklist":
                prompt = build_prompt_audit_checklist(user_query, db_context, search_depth)
                system_prompt = SYSTEM_PROMPTS["audit"]
            elif mode_key == "📚 In-depth + Online":
                # TODO: Integrate web search
                prompt = build_prompt_indepth_online(user_query, db_context, "", search_depth)
            elif mode_key == "📊 Summary + Online":
                # TODO: Integrate web search
                prompt = build_prompt_summary_online(user_query, db_context, "")
            elif mode_key == "🌐 Online Only":
                prompt = build_prompt_online_only(user_query)

            # Step 3: Generate Response
            status_text.text(f"Step 3/3: Running via {provider_name}...")
            progress_bar.progress(75)

            st.session_state.messages.append({"role": "user", "content": user_query})
            
            st.markdown("---")
            
            with st.chat_message("assistant", avatar="🔬"):
                assistant_placeholder = st.empty()
                output_text, gen_metadata = generate_response(
                    provider_name=provider_name,
                    model=model_choice,
                    prompt=prompt,
                    stream=stream_output,
                    placeholder=assistant_placeholder,
                    system_prompt=system_prompt,
                    mode=mode_key
                )

            st.session_state.messages.append({
                "role": "assistant", 
                "content": output_text,
                "metadata": gen_metadata
            })

            progress_bar.progress(100)
            status_text.text("✅ Complete")
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            # Show references if enabled
            if show_sources and metadata_refs:
                st.markdown("---")
                with st.expander("📚 Sources Referenced", expanded=False):
                    st.subheader("Document References")
                    for ref in metadata_refs:
                        st.write(f"**[{ref['index']}]** {ref['source']} (Page {ref['page']})")

        except Exception as e:
            err = str(e)
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Execution paused: {err}")
            logger.error(f"Execution error: {err}")

# --- 15. EXPORT PANEL ---
if len(st.session_state.messages) > 0:
    st.markdown("---")
    st.subheader("📥 Export Active Analysis")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    last_assistant_msg = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
    if last_assistant_msg:
        output_text = last_assistant_msg["content"]
        mode_used = last_assistant_msg.get("metadata", {}).get("mode", "Analysis")
        
        with dl_col1:
            st.download_button(
                "📥 Markdown Sheet (.md)",
                data=output_text,
                file_name=f"pharmagent_{mode_used.lower().replace(' ', '_')}_{ts}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with dl_col2:
            try:
                pdf_bytes = generate_pdf_report(output_text, user_query if user_query else "Analysis", mode_used)
                st.download_button(
                    "📄 Corporate PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"pharmagent_report_{ts}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"PDF generation issue: {str(e)[:50]}")
        with dl_col3:
            full_report = f"# PharmAgent Archive\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n**Mode:** {mode_used}\n\n---\n\n{output_text}"
            st.download_button(
                "📚 Full History (.md)",
                data=full_report,
                file_name=f"pharmagent_full_history_{ts}.md",
                mime="text/markdown",
                use_container_width=True
            )
