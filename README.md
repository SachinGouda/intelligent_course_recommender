# Intelligent Course Recommender & Learning Path Q&A 🎓

A mini app that recommends courses and answers learning path questions using **embeddings + a feedback loop**, with optional OpenAI integration for richer Q&A.

> For detailed design and architecture, see [DESIGN_NOTE.md](./DESIGN_NOTE.md).

---

## 🌿 Branches
This repo has **two branches** to demonstrate evolution:

- **`main` (baseline)**  
  - HuggingFace embeddings only (`all-MiniLM-L6-v2`).  
  - Rule-based Q&A, no external API.  
  - Fully local, lightweight.  

- **`openai-enhanced` (extended)**  
  - Adds OpenAI integration:  
    - `text-embedding-ada-002` embeddings.  
    - GPT-4o chat completion for Q&A (with local fallback).  
  - Shows rationale scores and improved feedback loop.  

👉 Use `main` for a local-only.  
👉 Use `openai-enhanced` for the richer version.

---

## ✨ Features
- **User Profile Input**: capture background, interests, goals, and skills.
- **Course Recommendation Engine**: ~40 curated courses (`courses.json`) with provider, tags, difficulty, cost, and duration.
- **Embeddings**:  
  - Local: SentenceTransformers (`all-MiniLM-L6-v2`).  
  - Optional: OpenAI `text-embedding-ada-002` if API key is provided.  
- **Feedback Learning Loop**:  
  1. **Tag Preference Modeling** — like/dislike updates per-tag scores and re-ranks.  
  2. **Vector Adjustment** — user embedding nudged toward liked courses, away from disliked ones.  
  3. **Persistent Storage** — stored in `feedback_store.json` for session-to-session learning.  
- **Learning Path Q&A Agent**:  
  - Local: simple RAG over `kb.md`.  
  - Optional: GPT-4o with profile + KB context.  
- **Filters**: by difficulty, provider, and cost.  
- **Interface**: Streamlit.

---

## 🛠️ Tech Stack
- Python 3.9+  
- [streamlit](https://streamlit.io)  
- [sentence-transformers](https://www.sbert.net)  
- numpy  
- openai (optional)  
- python-dotenv  

---

## 🚀 Quickstart

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app:
```bash
streamlit run streamlit_app.py
```

Files used:
- `courses.json` — course dataset  
- `kb.md` — knowledge base for Q&A  
- `feedback_store.json` — created at runtime  

---

## ⚙️ How it Works
1. Build user text from profile (background, interests, goals, skills).  
2. Embed user profile + all courses.  
3. Compute similarity.
4. On 👍 like: increment tag scores, nudge vector toward course.  
5. On 👎 dislike: decrement tag scores, nudge vector away.  
6. Persist feedback in `feedback_store.json`.  
7. Q&A: retrieve top KB chunks → answer with GPT-4o or rule-based fallback.

---

## 📂 File Structure
- `streamlit_app.py` — main app  
- `courses.json` — course dataset (~40 items)  
- `kb.md` — knowledge base for Q&A  
- `feedback_store.json` — feedback persistence (auto-created)  
- `DESIGN_NOTE.md` — architecture & design details  

---

## 📖 Sample Input/Output

**Profile**
- Background: *"Final-year CS student"*  
- Interests: *"AI, Data Science, MLOps"*  
- Goals: *"Become an ML Engineer at a product company"*  
- Skills: *"Python: intermediate, Math: beginner, SQL: intermediate"*  

**Recommendations (example)**
1. Generative AI with LLMs — Coursera (DeepLearning.AI)  
2. ML Ops Specialization — Coursera (DeepLearning.AI)  
3. LangChain for LLM Apps — Udemy  
4. Mathematics for Machine Learning — Coursera (Imperial)  
5. Intro to TensorFlow for AI — Coursera (DeepLearning.AI)  

**Q&A Example**
- Q: *“Should I learn Python or R for data science?”*  
- A: *Python* is better for ML engineering & deployment; *R* is better for statistics-heavy or research workflows.  

---

## 📌 License
MIT