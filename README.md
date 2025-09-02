Agentic NL → SQL Explorer

The Agentic NL → SQL Explorer is an advanced application that enables seamless interaction with relational databases through natural language queries. Instead of writing complex SQL code, users can ask questions in plain English—such as “Show all providers and their specialties” or “Count claims by provider name and specialty”—and receive accurate, schema-aware SQL queries along with results and clear explanations.

At the core, the system uses a multi-agent workflow built with LangGraph. Each agent has a specialized role:

User Input Agent reformulates user queries into schema-aware requests by mapping them to existing tables and columns.

SQL Generator Agent leverages state-of-the-art language models (sqlcoder, mistral) to translate reformulated queries into valid SQL.

SQL Validator Agent cleans and validates queries to ensure compatibility with SQLite, removing unsupported syntax.

DB Executor Agent runs queries and returns structured outputs as tables.

Explainer Agent produces concise, user-friendly interpretations of the query results, helping non-technical users understand insights without SQL knowledge.

The app integrates synonym handling (e.g., “doctor” → PROVIDER_NAME, “insurance” → PAYER_NAME), ensuring business-friendly phrasing is mapped to the correct schema. It also includes query caching, allowing frequently used or user-“liked” queries to be reused instantly for faster results.

Designed to improve accessibility, accuracy, and efficiency, this tool empowers analysts, healthcare professionals, and decision-makers to explore large claims datasets without technical barriers, accelerating insight generation and supporting informed decision-making.
