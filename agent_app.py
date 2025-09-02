import streamlit as st
import sqlite3
import pandas as pd
import re
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

# ----------------- Setup -----------------
# Use different models for different agents
llm_rewriter = OllamaLLM(model="mistral:7b-instruct")  # Fast schema-aware rewriter + explainer
llm_sql = OllamaLLM(model="sqlcoder:15b")              # Strong SQL generator

DB_PATH = "claims.db"

# ----------------- State -----------------
class QueryState(dict):
    question: str
    clarified_question: str
    raw_sql: str
    cleaned_sql: str
    results: pd.DataFrame
    explanation: str

# ----------------- Helpers -----------------
def get_column_names():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        schema[table] = [col[1] for col in cursor.fetchall()]
    conn.close()
    return schema

schema = get_column_names()

# Synonym dictionary: business terms → schema columns
SYNONYMS = {
    "doctor": "PROVIDER_NAME",
    "physician": "PROVIDER_NAME",
    "hospital": "PROVIDER_GROUP_NAME",
    "clinic": "PROVIDER_GROUP_NAME",
    "payer": "PAYER_NAME",
    "insurance": "PAYER_NAME",
}

def enhance_clarification(original_query: str) -> str:
    """Enhance query clarification to prevent specialty issues"""
    if not original_query:
        return original_query
        
    query_lower = original_query.lower()
    
    # Remove any existing enhancements to avoid duplication
    original_query = re.sub(r'\. Use Claim\.SPECIALITY.*$', '', original_query)
    original_query = re.sub(r'\. Use Provider\.PROVIDER_SPECIALTY.*$', '', original_query)
    
    # Fix specialty-related queries
    if any(term in query_lower for term in ['specialty', 'speciality', 'specialization', 'specialities']):
        if 'provider' in query_lower:
            return f"{original_query}. CRITICAL: Use Claim.SPECIALITY (Provider.PROVIDER_SPECIALTY column does NOT exist) and JOIN Provider with Claim on both DISPUTE_NUMBER and DLI_NUMBER."
    
    # Fix other common column issues
    if 'median' in query_lower and 'offer' in query_lower:
        if 'provider' in query_lower:
            return f"{original_query}. Use Provider.PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT for median provider offer percentages."
        elif 'payer' in query_lower:
            return f"{original_query}. Use Client.PAYER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PAYER_OFFER_AMOUNT for median payer offer percentages."
    
    return original_query

def fix_common_sql_issues(sql: str) -> str:
    """Fix common SQL issues and hallucinations"""
    if not sql:
        return sql
        
    # Fix specialty column issues
    specialty_patterns = [
        (r'(?i)provider\.provider_specialty', 'claim.speciality'),
        (r'(?i)provider_specialty', 'claim.speciality'),
        (r'(?i)provider\.specialty', 'claim.speciality'),
        (r'(?i)specialty', 'speciality'),
    ]
    
    for pattern, replacement in specialty_patterns:
        sql = re.sub(pattern, replacement, sql)
    
    # Fix median column issues
    median_patterns = [
        (r'(?i)median_payer_offer\.median_percent', 'Client.PAYER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PAYER_OFFER_AMOUNT'),
        (r'(?i)median_provider_offer\.median_percent', 'Provider.PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT'),
    ]
    
    for pattern, replacement in median_patterns:
        sql = re.sub(pattern, replacement, sql)
    
    # ... rest of the existing fixes ...
    
    return sql.strip()

# ----------------- Agents -----------------
# 1. User Input Agent (schema-aware NL → clarified NL)
def user_input_agent(state: QueryState):
    question = state["question"]
    schema_info = get_column_names()

    # Step A: Detect relevant columns from synonyms
    detected_cols = []
    for keyword, col in SYNONYMS.items():
        if keyword.lower() in question.lower():
            detected_cols.append(col)

    # Step B: Use LLM to reformulate
    prompt = f"""
    You are an assistant that reformulates user questions into schema-aware queries.

    Schema:
    {schema_info}

    Task:
    - Identify which table(s) and column(s) from the schema match this question.
    - If the query requires fields from multiple tables (like Claim + Provider or Claim + Client), 
      explicitly state that they must be JOINed on DISPUTE_NUMBER and DLI_NUMBER.
    - Rewrite the user question into a schema-aware version.
    - Always use exact table and column names.
    - Do not invent new columns or tables.

    Examples:
    User: Show claims by doctor specialty
    Reformulated: Use Claim.SPECIALITY to count claims grouped by specialty.

    User: Count claims by provider name and specialty
    Reformulated: Use Claim.SPECIALITY and Provider.PROVIDER_NAME, 
    and JOIN Claim with Provider on DISPUTE_NUMBER and DLI_NUMBER 
    to count claims grouped by provider name and specialty.

    User: List hospital names
    Reformulated: Use Provider.PROVIDER_GROUP_NAME to list provider group names.

    User: Show payer names
    Reformulated: Use Client.PAYER_NAME to list all payer names.

    User: {question}
    Reformulated:
    """

    clarified = llm_rewriter.invoke(prompt).strip()
    
    # Enhance clarification to prevent specialty issues
    clarified = enhance_clarification(clarified)

    # Step C: Add synonym hints if any detected
    if detected_cols:
        clarified += f"\n(Hint: also consider {', '.join(detected_cols)})"

    # Step D: Confidence check
    if not detected_cols and not any(
        tbl in clarified.upper() for tbl in ["PROVIDER", "CLAIM", "CLIENT"]
    ):
        clarified += "\n⚠️ Warning: Could not confidently map columns, results may be incomplete."

    state["clarified_question"] = clarified
    return state


# 2. SQL Generator Agent
# Update the SQL generator to handle median queries properly
def sql_generator(state: QueryState, retries: int = 2):
    clarified = state["clarified_question"]
    schema_info = get_column_names()

    # Check if this is a median query and provide specific guidance
    median_fix = ""
    if "median" in clarified.lower() and "offer" in clarified.lower():
        if "provider" in clarified.lower():
            median_fix = "CRITICAL: For median provider offers, use Provider.PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT"
        elif "payer" in clarified.lower():
            median_fix = "CRITICAL: For median payer offers, use Client.PAYER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PAYER_OFFER_AMOUNT"

    prompt = f"""
    You are an expert SQLite SQL generator. Generate ONLY SQL code, not Python or any other language.

    Schema:
    {schema_info}

    Rules:
    - ONLY generate SQL code, no explanations, no Python, no other languages.
    - Only use the following tables: Claim, Provider, Client.
    - Always use exact table and column names from schema.
    - CRITICAL: Provider.PROVIDER_SPECIALTY does NOT exist. Use Claim.SPECIALITY instead.
    {median_fix}
    - If the query requires columns from multiple tables, 
      JOIN them using both DISPUTE_NUMBER and DLI_NUMBER.
    - Always add LIMIT 50 unless otherwise specified.
    - SQLite only: use RANDOM(), LIKE.
    - Do NOT invent columns or tables.

    Examples:

    Q: Show all providers and their specialties
    A: SELECT DISTINCT p.PROVIDER_NAME, c.SPECIALITY
       FROM Provider p
       JOIN Claim c ON p.DISPUTE_NUMBER = c.DISPUTE_NUMBER AND p.DLI_NUMBER = c.DLI_NUMBER
       ORDER BY p.PROVIDER_NAME, c.SPECIALITY
       LIMIT 50;

    Q: What is the median offer percentage of providers?
    A: SELECT AVG(PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT) as avg_median_offer_percentage
       FROM Provider
       LIMIT 50;

    Q: Which providers have a higher offer percentage than the median provider offer?
    A: SELECT PROVIDER_NAME, PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT
       FROM Provider
       WHERE PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT > (
           SELECT AVG(PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT) 
           FROM Provider
       )
       ORDER BY PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT DESC
       LIMIT 50;

    Now generate SQL for:
    {clarified}
    SQL:
    """

    # --- Step 1: Generate SQL ---
    sql = llm_sql.invoke(prompt).strip()
    
    # --- Guard 0: Check if it's Python code and reject immediately ---
    if any(keyword in sql.lower() for keyword in ['import ', 'def ', 'class ', 'pd.', 'fake.', 'random.', 'if __name__']):
        if retries > 0:
            return sql_generator({**state, "clarified_question": clarified}, retries - 1)
        else:
            state["raw_sql"] = "-- ERROR: LLM generated Python instead of SQL"
            return state

    # --- Guard 1: remove fenced code blocks ---
    if sql.startswith("```"):
        sql = re.sub(r"^```sql|```$", "", sql, flags=re.MULTILINE).strip()

    # --- Guard 2: keep only first statement ---
    if ";" in sql:
        parts = [s.strip() for s in sql.split(";") if s.strip()]
        sql = parts[0]

    # --- Guard 3: sanitize Postgres junk ---
    sql = sql.replace("ILIKE", "LIKE")
    sql = re.sub(r"::[a-zA-Z0-9_\[\]]+", "", sql)
    sql = re.sub(r"\bDISTINCT ON\b", "DISTINCT", sql, flags=re.IGNORECASE)

    # --- Enhanced Guard 4: Apply common SQL fixes ---
    sql = fix_common_sql_issues(sql)

    # --- Specific fixes for common issues ---
    # Fix median_payer_offer.median_percent issue
    sql = re.sub(r'median_payer_offer\.median_percent', 'Client.PAYER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PAYER_OFFER_AMOUNT', sql, flags=re.IGNORECASE)
    sql = re.sub(r'median_provider_offer\.median_percent', 'Provider.PROVIDER_OFFER_AS_PERCENTAGE__OF_MEDIAN_PROVIDER_OFFER_AMOUNT', sql, flags=re.IGNORECASE)
    
    # Fix PROVIDER_SPECIALTY usage
    sql = re.sub(r'Provider\.PROVIDER_SPECIALTY', 'Claim.SPECIALITY', sql, flags=re.IGNORECASE)
    sql = re.sub(r'provider\.provider_specialty', 'claim.speciality', sql, flags=re.IGNORECASE)

    # --- Guard 5: enforce LIMIT ---
    if (sql.lower().startswith("select") and 
        "limit" not in sql.lower() and 
        not re.search(r'(?i)count\(|sum\(|avg\(|max\(|min\(|group by', sql)):
        sql += " LIMIT 50"

    # --- Step 2: Validate with SQLite ---
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        conn.close()
    except Exception as e:
        if retries > 0:
            # Provide a hardcoded fallback for the specific query
            if "show all providers and their specialties" in clarified.lower():
                fallback_sql = """
                SELECT DISTINCT p.PROVIDER_NAME, c.SPECIALITY
                FROM Provider p
                JOIN Claim c ON p.DISPUTE_NUMBER = c.DISPUTE_NUMBER AND p.DLI_NUMBER = c.DLI_NUMBER
                ORDER BY p.PROVIDER_NAME, c.SPECIALITY
                LIMIT 50
                """
                state["raw_sql"] = fallback_sql
                return state
                
            return sql_generator({**state, "clarified_question": clarified}, retries - 1)
        else:
            state["raw_sql"] = f"-- Invalid SQL: {e}"
            return state

    state["raw_sql"] = sql
    return state

# 3. SQL Validator Agent
def sql_validator(state: QueryState):
    sql = state["raw_sql"]

    # Clean hallucinations
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL).strip()
    sql = re.sub(r'""(\w+)""', r'"\1"', sql)
    sql = re.sub(r"''(\w+)''", r'"\1"', sql)
    sql = sql.replace("RAND()", "RANDOM()")
    sql = re.sub(r"NULLS (FIRST|LAST)", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r'(\b\w+\b)(_\1)+', r'\1', sql, flags=re.IGNORECASE)

    # Apply common SQL fixes
    sql = fix_common_sql_issues(sql)

    # Ensure LIMIT safeguard
    if (sql.lower().startswith("select") and 
        "limit" not in sql.lower() and 
        not re.search(r'(?i)count\(|sum\(|avg\(|max\(|min\(|group by', sql)):
        sql = sql.rstrip(";") + " LIMIT 50;"

    state["cleaned_sql"] = sql
    return state

# 4. DB Executor Agent
def db_executor(state: QueryState):
    sql = state["cleaned_sql"]
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        state["results"] = df
    except Exception as e:
        state["results"] = pd.DataFrame({"Error": [str(e)]})
    return state

# 5. Explainer Agent
def explainer(state: QueryState):
    df = state["results"]
    sql = state["cleaned_sql"]

    if "Error" in df.columns:
        state["explanation"] = f"⚠️ Query failed: {df['Error'].iloc[0]}"
    elif df.empty:
        state["explanation"] = "ℹ️ No results found."
    else:
        preview = df.head(5).to_dict(orient="records")

        # Detect if JOIN was used
        if "JOIN" in sql.upper():
            join_note = (
                "Note: This query joins multiple tables "
                "using DISPUTE_NUMBER and DLI_NUMBER as common keys."
            )
        else:
            join_note = ""

        prompt = f"""
        You are a data analyst. Explain the following SQL result in simple English.

        SQL: {sql}
        Results (first 5 rows): {preview}

        {join_note}

        Keep it concise and user-friendly.
        """

        state["explanation"] = llm_rewriter.invoke(prompt).strip()

    return state


# ----------------- LangGraph -----------------
graph = StateGraph(QueryState)

graph.add_node("User Input", user_input_agent)
graph.add_node("SQL Generator", sql_generator)
graph.add_node("SQL Validator", sql_validator)
graph.add_node("DB Executor", db_executor)
graph.add_node("Explainer", explainer)

graph.set_entry_point("User Input")
graph.add_edge("User Input", "SQL Generator")
graph.add_edge("SQL Generator", "SQL Validator")
graph.add_edge("SQL Validator", "DB Executor")
graph.add_edge("DB Executor", "Explainer")
graph.add_edge("Explainer", END)

app = graph.compile()

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Agentic NL → SQL Explorer", page_icon="🤖", layout="wide")
st.title("🤖 Agentic NL → SQL Explorer (Hybrid LLMs + Synonyms + LangGraph)")

q = st.text_input("💬 Ask in plain English (e.g. 'count claims by provider name and specialty'):")

if st.button("Run Query") and q:
    with st.spinner("Running multi-agent workflow..."):
        state = app.invoke({"question": q})

        st.subheader("🔍 Clarified Query")
        st.write(state["clarified_question"])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("💻 Raw SQL")
            st.code(state["raw_sql"], language="sql")
            st.subheader("✅ Cleaned SQL")
            st.code(state["cleaned_sql"], language="sql")
        with col2:
            st.subheader("📊 Results")
            st.dataframe(state["results"])
            st.subheader("📝 Explanation")
            st.write(state["explanation"])