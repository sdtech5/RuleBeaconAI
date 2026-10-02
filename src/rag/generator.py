from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def create_llm():
    """
    Create the LLM used to generate grounded answers.
    """

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )



def clean_answer(answer: str) -> str:
    """
    Remove accidental Markdown backticks that can make
    ordinary financial numbers appear as code blocks/gray
    highlighted text in the Streamlit UI.
    """

    if not answer:
        return answer

    return answer.replace("`", "")


def generate_answer(
    llm,
    question,
    retrieved_chunks,
    calculation_hint=None,
):
    """
    Generate an answer using only the retrieved filing
    evidence.

    When the question asks for a derived metric (growth rate, margin,
    share of total, ...), an optional ``calculation_hint`` is supplied so
    the model knows which formula to apply and that it must compute the
    result deterministically from the retrieved inputs.
    """

    context_blocks = []
    for chunk in retrieved_chunks:
        file_name = chunk.metadata.get("file_name") or ""
        section = chunk.metadata.get("section") or ""
        header = f"[Filing: {file_name} | {section}]" if file_name else ""
        if header:
            context_blocks.append(f"{header}\n{chunk.page_content}")
        else:
            context_blocks.append(chunk.page_content)

    context = "\n\n".join(context_blocks)

    if calculation_hint:
        calculation_section = (
            "\nCALCULATION RULES (this question asks for a derived metric):\n"
            f"- {calculation_hint}\n"
            "- Compute the value from figures that appear in the filing "
            "context above. Do NOT report the metric as unavailable merely "
            "because the filing does not state the percentage explicitly.\n"
            "- Clearly show the input figures you used and the computed "
            "result.\n"
            "- Use the correct fiscal-year figure for each input; never "
            "reuse one year's figure for a different year.\n"
            "- If a required input does not appear anywhere in the context, "
            "state that the input is not available and do not estimate it.\n"
        )
    else:
        calculation_section = ""

    prompt = f"""
You are a financial research assistant.

Answer the user's question using ONLY the information
contained in the filing context below.

IMPORTANT RULES:

1. Answer the exact fiscal year requested by the user.
2. Do not substitute a different fiscal year even if
   another year appears prominently in the context.
3. Pay close attention to the exact financial metric
   requested.
4. SYNONYM RULE — treat these as the same metric unless
   the filing explicitly distinguishes them:
   "net sales", "revenues", "total revenues", "net revenues",
   "net revenue", "revenue", "total revenue".
   If the user asks for "net sales" and the filing reports
   "Revenues", use that figure and note the label used.
5. Distinguish between truly different metrics (e.g. gross
   profit vs. operating income) when the filing does so.
6. When comparing figures across fiscal years, state each
   fiscal year's figure clearly before presenting any
   comparison or percentage change.
   - IMPORTANT: Many income-statement tables include figures
     for multiple years side-by-side. Use all years shown
     in that table even if the chunk is labelled as a single
     year's filing.
7. If the requested information is genuinely not present
   anywhere in the provided context, say so clearly.
8. Keep the answer concise and direct.
9. You may use a short bullet list when useful.
10. Do NOT use Markdown tables.
11. Do NOT use backticks or code formatting.
12. Do not invent or estimate a number.
{calculation_section}
Filing context:
{context}

User question:
{question}

Answer:
"""


    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            block.get("text", 
                      "")
            for block in answer
            if isinstance(block, dict)
        )

    return clean_answer(answer)