# Intelligent Course Recommender & Learning Path Q&A

A mini app that recommends courses and answers learning path questions using embeddings + a feedback loop.

## Features
- **User Profile Input**: background, interests, goals, skills.
- **Course Recommendation Engine**: 40 curated online courses with provider, tags, difficulty, cost, duration.
- **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`) for user profile & course similarity.
- **Feedback Learning Loop** (threefold):
  1) **Tag Preference Modeling** — like/dislike updates per-tag scores and re-ranks.
  2) **Vector Adjustment** — user embedding is nudged toward liked courses, away from disliked ones.
  3) **Persistent Storage** — feedback stored in `feedback_store.json`.
- **Learning Path Q&A Agent**: simple RAG over a small knowledge base (kb.md). No external LLM required, but easy to extend.
- **Filters**: by difficulty, provider keywords, and cost.
- **Interface**: Streamlit.

## Tech Stack
- Python 3.9+
- sentence-transformers
- numpy
- streamlit

## Quickstart

```bash
pip install -U streamlit sentence-transformers numpy
streamlit run streamlit_app.py
```

The app reads `courses.json` and `kb.md` in the same directory and writes feedback to `feedback_store.json`.

## How it Works

1. We construct a user text from profile fields and embed it using `all-MiniLM-L6-v2`.
2. We embed all courses once and cache them. We compute cosine similarity (via normalized dot product).
3. We compute a **base tag score** from your learned tag preferences and min-max normalize it.
4. Final score = `0.8 * similarity + 0.2 * tag_score`. You can tune these weights.
5. On **like**: add course to likes, increment tag preferences by +1.0, and nudge the user vector toward liked courses.
6. On **dislike**: add to dislikes, decrement tag preferences by -0.7, and nudge user vector away.
7. Q&A: retrieve the top 3 KB chunks using embeddings; a rule-based synthesis produces a short, grounded answer with personalization hints.

## Sample Input/Output

**Profile**
- Background: "Final-year CS student"
- Interests: "AI, Data Science, MLOps"
- Goals: "Become an ML Engineer at a product company"
- Skills: "Python: intermediate, Math: beginner, SQL: intermediate"

**Top Recommendations (example)**
- Generative AI with LLMs (DeepLearning.AI)
- ML Ops Specialization (DeepLearning.AI)
- LangChain for LLM Apps (Udemy)
- Mathematics for Machine Learning (Imperial / Coursera)
- Intro to TensorFlow for AI (DeepLearning.AI)

**Q&A Example**
- Q: "Should I learn Python or R for data science?"
- A (summary): If you're targeting ML engineering and deployment, **Python** is the better first choice; pick **R** for statistics-heavy or research workflows.

## Design Choices

- **Local-by-default**: Uses a compact, widely-available embedding model for fast inference and easy setup.
- **Transparent feedback**: Tag preferences are visible and editable by resetting.
- **Sane heuristics**: Gentle vector nudges + tag-based re-ranking keep behavior stable but adaptive.
- **RAG-lite**: No external LLM required; swapping to an API is straightforward (plug your call inside `answer_question_with_context`).

## Possible Improvements
- Swap to FAISS/Chroma for larger datasets or multi-tenant setups.
- Add multi-criteria scoring (cost, duration) with user-set weights.
- Add full LLM Q&A with grounding and citations.
- Cold-start tag inference from profile (e.g., NER on profile text to seed tags).
- Session management, auth, and exporting learning paths as PDFs.

## File Structure
- `streamlit_app.py` — main app
- `courses.json` — course dataset (40 items)
- `kb.md` — knowledge base for Q&A
- `feedback_store.json` — created at runtime

## License
MIT