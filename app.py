import html
import json
import re
import textwrap
import urllib.error
import urllib.request

import streamlit as st


# ============================================================
# RuleBeaconAI
# Compact Streamlit UI for the existing FastAPI/RAG backend
# ============================================================

API_URL = "https://rulebeaconai-git-830839785981.asia-south1.run.app"

COMPANIES = [
    ("Apple", "AAPL"),
    ("Archer Daniels Midland", "ADM"),
    ("Amazon", "AMZN"),
    ("Alphabet", "GOOGL"),
    ("JPMorgan Chase", "JPM"),
    ("Meta Platforms", "META"),
    ("Microsoft", "MSFT"),
    ("Netflix", "NFLX"),
    ("NVIDIA", "NVDA"),
    ("Tesla", "TSLA"),
    ("Walmart", "WMT"),
]

COMPANY_ALIASES = {
    "apple": "Apple",
    "adm": "ADM",
    "archer daniels midland": "Archer Daniels Midland",
    "amazon": "Amazon",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "jpmorgan": "JPMorgan Chase",
    "jpmorgan chase": "JPMorgan Chase",
    "meta": "Meta Platforms",
    "facebook": "Meta Platforms",
    "microsoft": "Microsoft",
    "netflix": "Netflix",
    "nvidia": "NVIDIA",
    "tesla": "Tesla",
    "walmart": "Walmart",
}

SUGGESTED_QUESTIONS = [
    ("1. AAPL", "amber", "What were Apple's net sales in fiscal year 2023?"),
    ("2. ADM", "cyan", "What was ADM's net earnings in fiscal year 2022?"),
    ("3. NVDA", "green", "What was NVIDIA's revenue in fiscal year 2024?"),
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RuleBeaconAI - SEC Filing Research Assistant",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

def init_state():
    defaults = {
        "messages": [],
        "query_input": "",
        "clear_query_input": False,
        "sample_document_open": False,
        "processing": False,
        "pending_question": None,
        "active_company": None,
        "active_year": None,
        "active_metric": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat():
    st.session_state.messages = []
    st.session_state.query_input = ""
    st.session_state.clear_query_input = False
    st.session_state.sample_document_open = False
    st.session_state.processing = False
    st.session_state.pending_question = None
    st.session_state.active_company = None
    st.session_state.active_year = None
    st.session_state.active_metric = None


init_state()


def render_html(markup):
    """Render HTML without Markdown's indentation/code-block interpretation."""
    st.markdown(
        textwrap.dedent(markup).strip(),
        unsafe_allow_html=True,
    )


# ============================================================
# COMPACT THEME
# ============================================================

st.markdown(
    """
<style>

/* Remove Streamlit chrome that is not part of the application. */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

header[data-testid="stHeader"] {
    background: #0d1117 !important;
}

.stApp {
    background: #0d1117 !important;
    color: #e5e7eb !important;
}

/* Keep Streamlit's normal responsive layout, but make it compact. */
.block-container {
    max-width: 1180px !important;
    padding: 28px 30px 32px !important;
}

/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #0b0e14 !important;
    border-right: 1px solid #1e293b !important;
}

section[data-testid="stSidebar"] > div {
    padding: 20px 14px 24px !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
    display: none !important;
}

.rb-brand {
    color: #ffffff;
    font-family: Georgia, serif;
    font-size: 24px;
    line-height: 1.1;
    font-weight: 700;
    letter-spacing: -0.035em;
}

.rb-subtitle {
    margin-top: 6px;
    color: #9ca3af;
    font-size: 12px;
}

.rb-sticky-brand {
    background: #0b0e14;
    padding-bottom: 6px;
}

section[data-testid="stSidebar"]
div[data-testid="stElementContainer"]:has(.rb-sticky-brand) {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: #0b0e14;
    padding-top: 20px;
    margin-top: -20px;
}

section[data-testid="stSidebar"]
div[data-testid="stElementContainer"]:has(.rb-sticky-brand)::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: -35px;
    height: 35px;
    background: #0b0e14;
    pointer-events: none;
}

.rb-sidebar-heading {
    margin: 35px 0 11px;
    color: #9ca3af;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.rb-corpus {
    padding: 11px 12px 12px;
    background: #10151d;
    border: 1px solid #262d3d;
    border-radius: 7px;
}

.rb-metric {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 27px;
    color: #9ca3af;
    font-size: 12px;
}

.rb-metric + .rb-metric {
    border-top: 1px dotted #293242;
}

.rb-metric-value {
    color: #f3f4f6;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
}

.rb-coverage {
    color: #fbbf24;
    font-size: 11px;
}

.rb-tracked {
    margin-top: 9px;
    padding-top: 10px;
    border-top: 1px solid #262d3d;
}

.rb-tracked-heading {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    color: #9ca3af;
    font-size: 10px;
    letter-spacing: .07em;
    text-transform: uppercase;
}

.rb-tracked-count {
    color: #64748b;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    letter-spacing: 0;
}

.rb-entities {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}

.rb-chip {
    display: inline-flex;
    align-items: baseline;
    gap: 3px;
    padding: 3px 5px;
    background: #111827;
    border: 1px solid #263044;
    border-radius: 4px;
    white-space: nowrap;
}

.rb-chip-name {
    color: #f3f4f6;
    font-size: 9.5px;
    font-weight: 500;
}

.rb-chip-ticker {
    color: #9ca3af;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 8px;
}

.rb-pipeline {
    margin: 15px 0 0;
    padding-top: 13px;
    padding-bottom: 13px;
    border-top: 1px solid #202938;
    color: #8f9aa8;
    font-size: 11px;
    line-height: 1.35;
}

.rb-sidebar-button {
    margin-top: 14px;
}

section[data-testid="stSidebar"] div[data-testid="stButton"] {
    margin-top: 9px !important;
}

.rb-sidebar-link {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 35px;
    box-sizing: border-box;
    background: #171c25;
    border: 1px solid #283244;
    border-radius: 5px;
    color: #e8eef6 !important;
    font-size: 12px;
    font-weight: 400;
    text-decoration: none !important;
    box-shadow: none;
    cursor: pointer;
}

.rb-sidebar-link:visited {
    color: #e8eef6 !important;
    text-decoration: none !important;
}

.rb-sidebar-link:hover {
    background: #1c2330;
    border-color: #39465d;
    color: #e8eef6 !important;
    text-decoration: none !important;
}

.rb-sidebar-link:active {
    color: #e8eef6 !important;
    text-decoration: none !important;
}

div[data-testid="stButton"] button {
    min-height: 35px !important;
    width: 100% !important;
    background: #171c25 !important;
    background-color: #171c25 !important;
    border: 1px solid #283244 !important;
    border-radius: 5px !important;
    color: #cbd5e1 !important;
    font-size: 12px !important;
    box-shadow: none !important;
}

div[data-testid="stButton"] button:hover {
    background: #1c2330 !important;
    border-color: #39465d !important;
    color: #f3f4f6 !important;
}

div[data-testid="stButton"] button:focus,
div[data-testid="stButton"] button:active {
    background: #171c25 !important;
    border-color: #39465d !important;
    color: #f3f4f6 !important;
    box-shadow: none !important;
}

.rb-sample {
    margin-top: 9px;
    padding: 10px 11px;
    background: #111720;
    border: 1px solid #252f41;
    border-radius: 5px;
    color: #9ca3af;
    font-size: 11px;
    line-height: 1.45;
}

/* ==========================================================
   MAIN
   ========================================================== */

.rb-main {
    width: min(850px, 100%);
    margin: 0 auto;
}

.rb-hero {
    margin-top: 20vh;
    color: #aeb8c5;
    font-size: 15px;
    line-height: 1.45;
    text-align: center;
}

 .rb-suggestions {
    margin-top: 26px;
}

.rb-suggestion-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 0 3px 10px;
}

.rb-suggestion-title {
    color: #9ca3af;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.rb-suggestion-help {
    color: #667085;
    font-size: 11px;
}

.rb-card {
    min-height: 112px;
    padding: 14px 15px;
    background: #1f2430;
    border: 1px solid #2a3242;
    border-radius: 10px;
}

.rb-badge {
    margin-bottom: 17px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .06em;
}

.rb-badge.amber { color: #f59e0b; }
.rb-badge.cyan { color: #22d3ee; }
.rb-badge.green { color: #34d399; }

.rb-card-question {
    color: #d7dce3;
    font-size: 13px;
    line-height: 1.45;
}

div[data-testid="stButton"] button[kind="primary"] {
    height: 40px !important;
    min-height: 40px !important;
    width: 48px !important;
    padding: 0 !important;
    background: #e3ad38 !important;
    background-color: #e3ad38 !important;
    border: 1px solid #e3ad38 !important;
    border-radius: 8px !important;
    color: #f1f5f9 !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

div[data-testid="stButton"]:has(button[kind="primary"]) {
    height: 40px !important;
    min-height: 40px !important;
    margin: 0 !important;
}

div[data-testid="stButton"]:has(button[kind="primary"]) > button[kind="primary"] {
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stButton"] button[kind="primary"]:focus,
div[data-testid="stButton"] button[kind="primary"]:active {
    background: #f5bb3d !important;
    background-color: #f5bb3d !important;
    border-color: #f5bb3d !important;
    color: #111827 !important;
    box-shadow: none !important;
}

/* ==========================================================
   CHAT
   ========================================================== */

.rb-chat {
    margin-top: 26px;
}

.rb-message {
    margin: 18px 0;
}

.rb-label {
    margin-bottom: 5px;
    color: #687384;
    font-size: 10px;
    letter-spacing: .07em;
}

.rb-user {
    display: inline-block;
    max-width: 88%;
    padding: 9px 12px;
    background: #1a2029;
    border: 1px solid #283140;
    border-radius: 6px;
    color: #dbe1e7;
    font-size: 14px;
    line-height: 1.45;
}

.rb-answer {
    color: #e5e7eb;
    font-size: 14px;
    line-height: 1.55;
}

.rb-answer code,
.rb-answer pre {
    background: transparent !important;
    color: inherit !important;
    font-family: inherit !important;
}

.rb-thinking {
    color: #9ca3af !important;
}

.rb-source {
    margin-top: 6px;
}

.rb-source summary {
    cursor: pointer;
    color: #6f7b8a;
    font-size: 11px;
    list-style: none;
}

.rb-source summary::-webkit-details-marker {
    display: none;
}

.rb-source summary::before {
    content: "▶";
    margin-right: 6px;
    font-size: 8px;
}

.rb-source-list {
    margin-top: 6px;
    padding-left: 11px;
    border-left: 1px solid #293343;
}

.rb-source-item {
    padding: 4px 0;
    color: #8d99a8;
    font-size: 10.5px;
    line-height: 1.4;
}

/* ==========================================================
   INPUT
   ========================================================== */

.rb-input-label {
    display: none;
}

.rb-input-row {
    margin-top: 26px;
}

.rb-input-col div[data-testid="stTextInput"] input,
.rb-input-col div[data-testid="stTextArea"] textarea {
    height: 40px !important;
    min-height: 40px !important;
    padding: 12px 14px !important;
    background: #181d26 !important;
    background-color: #181d26 !important;
    border: 1px solid rgba(212, 160, 52, .72) !important;
    border-right: 0 !important;
    border-radius: 10px 0 0 10px !important;
    color: #111827 !important;
    box-shadow: none !important;
    font-size: 15px !important;
    resize: none !important;
    line-height: 1.4 !important;
}

.rb-input-col div[data-testid="stTextInput"] input::placeholder,
.rb-input-col div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important;
    opacity: 1 !important;
}

.rb-input-col div[data-testid="stTextInput"] input:focus,
.rb-input-col div[data-testid="stTextArea"] textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 1px rgba(245, 158, 11, .15) !important;
}

.rb-send-col {
    margin-left: -1px;
}

.rb-send-col button {
    height: 48px !important;
    min-height: 48px !important;
    background: #e3ad38 !important;
    border: 1px solid #e3ad38 !important;
    border-radius: 0 10px 10px 0 !important;
    color: #111827 !important;
    font-size: 21px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
}

.rb-send-col button:hover {
    background: #f5bb3d !important;
    border-color: #f5bb3d !important;
    color: #111827 !important;
}

.rb-send-col button:focus,
.rb-send-col button:active {
    background: #e3ad38 !important;
    border-color: #e3ad38 !important;
    color: #111827 !important;
    box-shadow: none !important;
}

.rb-send-col button:disabled {
    background: #524322 !important;
    border-color: #524322 !important;
    color: #8c826e !important;
    cursor: not-allowed !important;
    opacity: 0.6 !important;
}

.rb-caption {
    margin: 7px 2px 0;
    color: #626d7b;
    font-size: 10.5px;
    text-align: right;
}

/* Reduce default widget spacing in this compact layout. */
div[data-testid="stVerticalBlock"] {
    gap: .35rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND / QUERY HELPERS
# ============================================================

def extract_company(text):
    lowered = text.lower()

    for alias, company in sorted(
        COMPANY_ALIASES.items(),
        key=lambda item: -len(item[0]),
    ):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return company

    return None


COMMON_METRICS = [
    "net sales growth",
    "sales growth",
    "net sales",
    "total revenues",
    "total revenue",
    "net revenues",
    "net revenue",
    "revenues",
    "revenue",
    "net earnings",
    "net income",
    "net profit",
    "operating income",
    "operating profit",
    "income from operations",
    "gross profit",
    "earnings per share",
    "diluted earnings per share",
    "basic earnings per share",
    "eps",
    "cash and cash equivalents",
    "research and development",
    "r&d",
    "total assets",
    "total liabilities",
    "operating cash flow",
]


def extract_metric(text: str) -> str | None:
    lowered = text.lower()
    for metric in sorted(COMMON_METRICS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(metric)}\b", lowered):
            return metric
    return None


def extract_years(text: str) -> list[str]:
    return re.findall(r"\b(20\d{2})\b", text)


def extract_year(text: str) -> str | None:
    years = extract_years(text)
    return years[0] if years else None


def build_contextual_question(question: str) -> str:
    company = extract_company(question)
    years = extract_years(question)
    metric = extract_metric(question)

    if company:
        st.session_state.active_company = company

    if metric:
        st.session_state.active_metric = metric

    additions = []

    is_comparison = bool(
        re.search(
            r"\b(compare|comparison|versus|vs|how does it compare|difference|growth|change)\b",
            question.lower(),
        )
    )

    if len(years) > 1:
        st.session_state.active_year = max(years)
    elif len(years) == 1:
        curr_year = years[0]
        prev_year = st.session_state.active_year
        if is_comparison and prev_year and prev_year != curr_year:
            additions.append(
                f"Comparison context: compare fiscal year {curr_year} with fiscal year {prev_year}."
            )
        st.session_state.active_year = curr_year
    else:
        if st.session_state.active_year:
            additions.append(
                f"Fiscal-year context: use fiscal year "
                f"{st.session_state.active_year} unless the question says otherwise."
            )

    if not company and st.session_state.active_company:
        additions.append(
            f"Company context: {st.session_state.active_company}."
        )

    if not metric and st.session_state.active_metric:
        additions.append(
            f"Metric context: {st.session_state.active_metric}."
        )

    if additions:
        return question + "\n\n" + " ".join(additions)

    return question


def ask_api(question):
    payload = json.dumps({"question": question}).encode("utf-8")

    request = urllib.request.Request(
        f"{API_URL}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def clean_answer(answer):
    answer = str(answer or "").strip()

    for marker in ("```text", "```markdown", "```"):
        answer = answer.replace(marker, "")

    return answer.strip()


def render_sources(sources):
    if not sources:
        return

    items = []

    for index, source in enumerate(sources, start=1):
        file_name = html.escape(
            str(source.get("file_name") or "Unknown filing")
        )

        section = html.escape(
            str(source.get("section") or "Section unavailable")
        )

        items.append(
            f'<div class="rb-source-item">{index:02d} · {file_name}<br>{section}</div>'
        )

    sources_html = "".join(items)

    render_html(
        f"""
        <details class="rb-source">
            <summary>Sources · {len(sources)} retrieved</summary>
            <div class="rb-source-list">
                {sources_html}
            </div>
        </details>
        """
    )


def run_query(question):
    question = question.strip()

    if not question:
        return

    contextual_question = build_contextual_question(question)

    try:
        result = ask_api(contextual_question)

        answer = clean_answer(
            result.get("answer") or "No answer was returned."
        )

        sources = result.get("sources") or []

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )

    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = str(exc)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"FastAPI returned HTTP {exc.code}. "
                    "Check the FastAPI terminal for the server traceback."
                ),
                "sources": [],
                "error": detail,
            }
        )

    except urllib.error.URLError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "RuleBeaconAI API is not reachable. "
                    "Start FastAPI with "
                    "`uvicorn src.api.main:app --reload`."
                ),
                "sources": [],
            }
        )

    except Exception as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Unexpected client error: {exc}",
                "sources": [],
            }
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
    """
    <div class="rb-sticky-brand">
        <div class="rb-brand">RuleBeaconAI</div>
        <div class="rb-subtitle">SEC filing research assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.markdown(
        '<div class="rb-sidebar-heading">CORPUS</div>',
        unsafe_allow_html=True,
    )

    chips = "".join(
        f"""
        <span class="rb-chip">
            <span class="rb-chip-name">{html.escape(name)}</span>
            <span class="rb-chip-ticker">{ticker}</span>
        </span>
        """
        for name, ticker in COMPANIES
    )

    render_html(
        f"""
        <div class="rb-corpus">
            <div class="rb-metric">
                <span>Filings</span>
                <span class="rb-metric-value">55</span>
            </div>
            <div class="rb-metric">
                <span>Companies</span>
                <span class="rb-metric-value">11</span>
            </div>
            <div class="rb-metric">
                <span>Coverage</span>
                <span class="rb-metric-value rb-coverage">FY2020–FY2024</span>
            </div>
            <div class="rb-tracked">
                <div class="rb-tracked-heading">
                    <span>Tracked Entities</span>
                    <span class="rb-tracked-count">11 companies</span>
                </div>
                <div class="rb-entities">{chips}</div>
            </div>
        </div>
        """
    )

    st.markdown(
        '<div class="rb-pipeline">'
        'Hybrid Retrieval · Reranking · Grounded Generation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
    """
    <div class="rb-sidebar-button">
        <a
            href="https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAPL_2023.pdf"
            target="_blank"
            class="rb-sidebar-link"
        >
            📄  Sample Document  ›
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.markdown(
    """
    <div class="rb-sidebar-button">
        <a
            href="https://drive.google.com/file/d/1YROxH_F8bH4VqA8nB_KcPvfuZ8dIk4i0/view?usp=sharing"
            target="_blank"
            class="rb-sidebar-link"
        >
            📋  Sample Questions to Ask  ›
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

    
    st.markdown(
        '<div class="rb-sidebar-button">',
        unsafe_allow_html=True,
    )

    if st.button(
        "↻  Refresh",
        key="refresh",
        use_container_width=True,
    ):
        reset_chat()
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN PANE
# ============================================================

st.markdown('<div class="rb-main">', unsafe_allow_html=True)

# Landing-page hero remains compact and centered.
if not st.session_state.messages and not st.session_state.processing:
    st.markdown(
        '<div class="rb-hero">'
        'Query corporate 10-K filings across FY2020–FY2024 '
        'with grounded citations.'
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.messages or st.session_state.processing:

    st.markdown(
        '<div class="rb-chat">',
        unsafe_allow_html=True,
    )

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                '<div class="rb-message">'
                '<div class="rb-label">YOU</div>'
                '<div class="rb-user">'
                + html.escape(str(message["content"]))
                + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="rb-message">'
                '<div class="rb-label">RULEBEACONAI</div>'
                '<div class="rb-answer">'
                + message["content"]
                + '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            render_sources(
                message.get("sources") or []
            )

            if message.get("error"):
                with st.expander("Technical details"):
                    st.code(message["error"])

    if st.session_state.processing:
        st.markdown(
            '<div class="rb-message">'
            '<div class="rb-label">RULEBEACONAI</div>'
            '<div class="rb-answer rb-thinking">Thinking · • • •</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

if not st.session_state.messages and not st.session_state.processing:

    st.markdown(
        '<div class="rb-suggestions">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="rb-suggestion-header">'
        '<span class="rb-suggestion-title">SUGGESTED QUESTIONS</span>'
        '<span class="rb-suggestion-help"></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    columns = st.columns(3, gap="small")

    for index, (badge, color, question) in enumerate(SUGGESTED_QUESTIONS):
        with columns[index]:
            render_html(
                f"""
                <div class="rb-card">
                    <div class="rb-badge {color}">{badge}</div>
                    <div class="rb-card-question">
                        {html.escape(question)}
                    </div>
                </div>
                """
            )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# QUERY BOX
# ============================================================

st.markdown(
    '<div class="rb-input-row">',
    unsafe_allow_html=True,
)

input_col, send_col = st.columns(
    [1, 0.055],
    gap="small",
)

with input_col:
    # Clear the widget value before it is instantiated on this rerun.
    if st.session_state.clear_query_input:
        st.session_state.query_input = ""
        st.session_state.clear_query_input = False

    st.markdown(
        '<div class="rb-input-col">',
        unsafe_allow_html=True,
    )

    st.text_input(
    "Query",
    key="query_input",
    label_visibility="collapsed",
    placeholder="Ask a question about corporate 10-K filings...",
    )

    st.markdown("</div>", unsafe_allow_html=True)

with send_col:
    st.markdown(
        '<div class="rb-send-col">',
        unsafe_allow_html=True,
    )

    send_clicked = st.button(
        "↑",
        key="send_query",
        use_container_width=True,
        type="primary",
        disabled=st.session_state.processing,
    )

    st.markdown("</div>", unsafe_allow_html=True)

render_html(
    """
    <script>
    (function() {
        function attachHandler() {
            try {
                const doc = (window.parent && window.parent.document) ? window.parent.document : document;
                const textarea = doc.querySelector('div[data-testid="stTextArea"] textarea');
                if (textarea && !textarea.dataset.enterBound) {
                    textarea.dataset.enterBound = "true";
                    textarea.addEventListener('keydown', function(e) {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            const sendBtn = doc.querySelector('.rb-send-col button');
                            if (sendBtn && !sendBtn.disabled) {
                                sendBtn.click();
                            }
                        }
                    });
                }
            } catch(e) {}
        }
        attachHandler();
        setTimeout(attachHandler, 300);
        setTimeout(attachHandler, 1000);
    })();
    </script>
    """
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SUBMIT & EXECUTE
# ============================================================

if send_clicked and st.session_state.query_input.strip() and not st.session_state.processing:
    question = st.session_state.query_input.strip()
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    st.session_state.pending_question = question
    st.session_state.processing = True
    st.session_state.clear_query_input = True
    st.rerun()

if st.session_state.processing and st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    try:
        run_query(question)
    finally:
        st.session_state.processing = False
    st.rerun()


st.markdown("</div>", unsafe_allow_html=True)
