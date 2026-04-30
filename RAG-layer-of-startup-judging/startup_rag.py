import os
import json
import re
from datetime import datetime, timezone
import snowflake.connector
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from project .env and override stale shell values.
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

SNOWFLAKE_SYSTEM_PROMPT = """
SNOWFLAKE OUTPUT REQUIREMENTS (STRICT)

You must produce a Snowflake-ready analytics record with stable schema and deterministic fields.

Rules:
- Return only valid JSON.
- Do not rename keys, add extra nesting, or omit required fields.
- Use fixed numeric scales:
  - scores: 0.00 to 10.00 (2 decimals)
  - confidence: 0.00 to 1.00 (2 decimals)
- reason_codes must be uppercase snake_case.
- timestamp must be ISO-8601 UTC.
- If evidence is insufficient, lower confidence and populate missing_evidence.
- Ensure score coherence:
  - high execution_risk should not coexist with very high final_score unless justified in rationale.
- Keep text fields concise for dashboard readability (<= 180 chars each where possible).

Required top-level object:
{
  "snowflake_record": {
    "startup_id": "string",
    "run_id": "string",
    "timestamp_utc": "string",
    "model_name": "string",
    "prompt_version": "string",
    "scores": {
      "team": 0.00,
      "market": 0.00,
      "traction": 0.00,
      "defensibility": 0.00,
      "execution_risk": 0.00,
      "final_score": 0.00
    },
    "confidence": 0.00,
    "reason_codes": ["STRING"],
    "benchmark_group": "string",
    "estimated_percentile": 0.00,
    "evidence_lineage": [
      {
        "claim": "string",
        "chunk_id": "string",
        "support_strength": 0.00
      }
    ],
    "missing_evidence": ["string"],
    "counterfactuals": [
      {
        "scenario": "string",
        "expected_final_score_delta": 0.00
      }
    ],
    "governance_flags": {
      "drift_flag": false,
      "stability_risk": "low|medium|high"
    }
  }
}

Before returning output, verify:
1) JSON parses with no extra prose.
2) All required keys present.
3) Numeric fields are in range and 2-decimal format.
4) reason_codes are uppercase snake_case.
5) If confidence < 0.60, missing_evidence has at least 2 items.
If any check fails, fix and re-output JSON only.
""".strip()


def _is_iso_utc(timestamp_str):
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", timestamp_str))


def _is_two_decimal_number(value):
    return isinstance(value, (int, float)) and round(float(value), 2) == float(value)


def validate_snowflake_record(payload):
    if "snowflake_record" not in payload:
        raise ValueError("Missing top-level key: snowflake_record")
    record = payload["snowflake_record"]

    required_root = [
        "startup_id", "run_id", "timestamp_utc", "model_name", "prompt_version",
        "scores", "confidence", "reason_codes", "benchmark_group",
        "estimated_percentile", "evidence_lineage", "missing_evidence",
        "counterfactuals", "governance_flags"
    ]
    for key in required_root:
        if key not in record:
            raise ValueError(f"Missing key in snowflake_record: {key}")

    scores = record["scores"]
    required_scores = ["team", "market", "traction", "defensibility", "execution_risk", "final_score"]
    for key in required_scores:
        if key not in scores:
            raise ValueError(f"Missing score: {key}")
        if not _is_two_decimal_number(scores[key]) or not (0.00 <= float(scores[key]) <= 10.00):
            raise ValueError(f"Invalid score {key}: must be 0.00-10.00 with 2 decimals")

    confidence = record["confidence"]
    if not _is_two_decimal_number(confidence) or not (0.00 <= float(confidence) <= 1.00):
        raise ValueError("Invalid confidence: must be 0.00-1.00 with 2 decimals")

    if not _is_iso_utc(record["timestamp_utc"]):
        raise ValueError("timestamp_utc must be ISO-8601 UTC format (YYYY-MM-DDTHH:MM:SSZ)")

    for reason in record["reason_codes"]:
        if not re.match(r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$", reason):
            raise ValueError(f"Invalid reason_code: {reason}")

    if float(confidence) < 0.60 and len(record["missing_evidence"]) < 2:
        raise ValueError("If confidence < 0.60, missing_evidence must include at least 2 items")

    if scores["execution_risk"] >= 8.00 and scores["final_score"] >= 8.50:
        raise ValueError("Score coherence failed: high execution_risk conflicts with very high final_score")

    flags = record["governance_flags"]
    if "drift_flag" not in flags or "stability_risk" not in flags:
        raise ValueError("governance_flags must include drift_flag and stability_risk")
    if flags["stability_risk"] not in {"low", "medium", "high"}:
        raise ValueError("stability_risk must be one of: low, medium, high")


def get_snowflake_connection():
    required_env = [
        "SNOWFLAKE_USER",
        "SNOWFLAKE_ACCOUNT",
        "SNOWFLAKE_PRIVATE_KEY_FILE",
        "SNOWFLAKE_WAREHOUSE",
        "SNOWFLAKE_DATABASE",
        "SNOWFLAKE_SCHEMA",
        "SNOWFLAKE_ROLE",
    ]
    missing = [key for key in required_env if not os.getenv(key)]
    if missing:
        raise ValueError(f"Missing required Snowflake environment variables: {', '.join(missing)}")

    return snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        private_key_file=os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        schema=os.getenv("SNOWFLAKE_SCHEMA"),
        role=os.getenv("SNOWFLAKE_ROLE"),
    )


def insert_snowflake_record(payload):
    record = payload["snowflake_record"]
    conn = get_snowflake_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO APP_DB.APP_SCHEMA.STARTUP_AI_LOGS (
              startup_id, run_id, timestamp_utc, model_name, prompt_version,
              scores, confidence, reason_codes, benchmark_group, estimated_percentile,
              evidence_lineage, missing_evidence, counterfactuals, governance_flags, raw_payload
            )
            SELECT
              %s, %s, %s, %s, %s,
              PARSE_JSON(%s), %s, PARSE_JSON(%s), %s, %s,
              PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s)
            """,
            (
                record["startup_id"],
                record["run_id"],
                record["timestamp_utc"],
                record["model_name"],
                record["prompt_version"],
                json.dumps(record["scores"]),
                float(record["confidence"]),
                json.dumps(record["reason_codes"]),
                record["benchmark_group"],
                float(record["estimated_percentile"]),
                json.dumps(record["evidence_lineage"]),
                json.dumps(record["missing_evidence"]),
                json.dumps(record["counterfactuals"]),
                json.dumps(record["governance_flags"]),
                json.dumps(payload),
            ),
        )
    finally:
        cur.close()
        conn.close()


def setup_rag(pdf_path):
    """
    Sets up a RAG system from a PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} was not found.")

    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    # 3. Initialize OpenAI embeddings and create a FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 4. Set up deterministic LLM + retriever
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return {"llm": llm, "retriever": retriever}

def query_rag(rag_system, question, startup_id="startup_demo", run_id="run_demo"):
    """
    Queries the RAG system and enforces strict Snowflake JSON output.
    """
    docs = rag_system["retriever"].invoke(question)
    context = "\n\n".join(
        f"[chunk_{idx}] {doc.page_content[:1200]}" for idx, doc in enumerate(docs, start=1)
    )

    user_prompt = f"""
Question:
{question}

startup_id: {startup_id}
run_id: {run_id}
timestamp_utc: {datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
model_name: gpt-3.5-turbo
prompt_version: v1

Retrieved context:
{context}
""".strip()

    response = rag_system["llm"].invoke(
        [
            ("system", SNOWFLAKE_SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    )

    payload = json.loads(response.content)
    validate_snowflake_record(payload)
    insert_snowflake_record(payload)
    return payload

if __name__ == "__main__":
    # Path to the PDF file
    pdf_file = "RAG-layer-of-startup-judging/startup_failure_reason.pdf"
    
    try:
        # Check for API key
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable not set in .env or system environment.")
        else:
            print(f"Initializing RAG with {pdf_file}...")
            rag_system = setup_rag(pdf_file)
            
            sample_question = "Evaluate this startup idea and return only strict Snowflake JSON output."
            print(f"\nQuestion: {sample_question}")
            
            answer = query_rag(
                rag_system,
                sample_question,
                startup_id="startup_alpha",
                run_id="run_001"
            )
            print("\nSnowflake JSON output:")
            print(json.dumps(answer, indent=2))
            
    except Exception as e:
        print(f"An error occurred: {e}")
