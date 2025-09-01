# Design Note

## Architecture Overview

This application provides intelligent course recommendations and career learning path guidance using GenAI techniques. It is built with Python and Streamlit for a simple, interactive UI.

### Key Components

- **User Profile Input:** Sidebar form captures background, interests, goals, and skills.
-- **Course Recommendation Engine:** Uses a static dataset (`courses.json`) of 30–50 online courses, each with title, provider, description, skill level, and tags. User profile and course descriptions are encoded using either OpenAI Embeddings (`text-embedding-ada-002`) or SentenceTransformers. Top courses are recommended using three modes:
	- **Semantic (Embedding-based) Search:** Ranks courses by embedding similarity and tag preferences.
	- **Keyword Search:** Ranks courses by keyword and tag matches in title, description, and tags.
	- **Hybrid Search:** Combines both semantic and keyword scores for more robust recommendations.
- **Feedback Learning Loop:** Users can like/dislike courses. The system adapts recommendations using vector-based adjustment and tag preference modeling. Feedback is stored persistently in `feedback_store.json` for session-to-session learning.
- **Learning Path Q&A Agent:** Users can ask questions about learning paths and career advice. Answers are generated using either OpenAI GPT-4o (prompt engineering + context) or a local rule-based fallback. Retrieval from a curated knowledge base (`kb.md`) ensures factual and context-aware responses.

### GenAI Techniques

-- **Embedding-Based Search:** User profile and course descriptions are converted to embeddings for semantic similarity search using OpenAI `text-embedding-ada-002` or SentenceTransformers.
-- **Keyword Search:** Uses keyword and tag matching for fast, interpretable recommendations.
-- **Hybrid Search:** Combines semantic and keyword-based scores for improved personalization and robustness.
-- **Retrieval-Augmented Generation (RAG):** For Q&A, relevant knowledge base chunks are retrieved and used as context for LLM (OpenAI GPT-4o) or rule-based answers.

### Feedback Loop

- **Vector-Based Adjustment:** User embedding is updated based on feedback to personalize future recommendations.
- **Tag Preference Modeling:** Learns user preferences for course tags and re-ranks recommendations accordingly.
- **Persistent Storage:** Feedback is saved in JSON for long-term personalization.

### Model Fallback Logic

- The app automatically uses OpenAI for embeddings and Q&A if an API key is provided. If not, it falls back to local SentenceTransformers and rule-based Q&A. This makes the `openai-enhanced` branch work for both local-only and OpenAI-enabled scenarios without code changes.

### Improvements & Bonus Features

- Visual display of course tags, skill coverage, and rationale for recommendations.
- Filtering by platform, cost, and difficulty.
- Tracks session history and personalizes based on past interactions.
- Learning path templates and ranked recommendations in the knowledge base.

### Design Choices

- **Modularity:** Functions are organized for maintainability and readability.
- **Performance:** Uses fast in-memory numpy similarity for retrieval.
- **Extensibility:** Easily switch between local and OpenAI models; knowledge base and course dataset are editable.

### Future Enhancements

- Add more visualizations (e.g., tag clouds, skill coverage charts).
- Batching OpenAI embedding requests for efficiency (currently one-by-one).
- Store feedback in a database (e.g., SQLite) for scalability.
- Expand the knowledge base and course dataset for broader coverage.
