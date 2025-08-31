
import os
import json
import uuid
import time
import datetime
from typing import List, Dict, Any, Tuple
import httpx

import streamlit as st

# Embeddings
from sentence_transformers import SentenceTransformer, util
import numpy as np


# OpenAI integration
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    client = OpenAI(api_key=OPENAI_API_KEY,http_client = httpx.Client(verify=False))
else:
    client = None

APP_TITLE = "Intelligent Course Recommender & Learning Path Q&A"
COURSE_PATH = os.path.join(os.path.dirname(__file__), "courses.json")
KB_PATH = os.path.join(os.path.dirname(__file__), "kb.md")
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "feedback_store.json")

# ----------------- Utilities -----------------
def load_courses() -> List[Dict[str, Any]]:
    with open(COURSE_PATH, "r") as f:
        return json.load(f)

def load_kb() -> str:
    with open(KB_PATH, "r") as f:
        return f.read()

def load_feedback() -> Dict[str, Any]:
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    with open(FEEDBACK_PATH, "r") as f:
        return json.load(f)

def save_feedback(store: Dict[str, Any]) -> None:
    with open(FEEDBACK_PATH, "w") as f:
        json.dump(store, f, indent=2)

def ensure_user(store: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    if user_id not in store:
        store[user_id] = {
            "tag_preferences": {},  # tag -> score
            "liked_courses": [],
            "disliked_courses": [],
            "profile_history": [],
        }
    return store

# ----------------- Embedding Model -----------------

# Embedding model selection
@st.cache_resource
def get_model():
    if USE_OPENAI:
        return None  # No local model needed
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------- Recommendation Core -----------------

def embed_texts(model, texts: List[str]) -> np.ndarray:
    if USE_OPENAI:
        # Use OpenAI text-embedding-ada-002
        emb_list = []
        for text in texts:
            resp = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            emb_list.append(np.array(resp.data[0].embedding, dtype=np.float32))
        arr = np.stack(emb_list)
        # Normalize embeddings
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms == 0, 1, norms)
        return arr
    else:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

def build_user_text(profile: Dict[str, str]) -> str:
    parts = []
    for k in ["background", "interests", "goals", "skills"]:
        if profile.get(k):
            parts.append(f"{k}: {profile[k]}")
    return " | ".join(parts)

def score_courses(user_vec: np.ndarray, course_vecs: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    # cosine similarity already when normalized -> dot product
    sim = (course_vecs @ user_vec.T).flatten()

    # Combine with base score (e.g., from tag preferences / heuristics)
    # Weighted sum with small weight to base preference (0.2).
    return 0.8 * sim + 0.2 * base_scores

def compute_tag_base_scores(courses: List[Dict[str, Any]], tag_pref: Dict[str, float]) -> np.ndarray:
    scores = []
    for c in courses:
        s = 0.0
        for t in c.get("tags", []):
            s += tag_pref.get(t, 0.0)
        scores.append(s)
    if len(scores) == 0:
        return np.zeros((0,), dtype=np.float32)
    arr = np.array(scores, dtype=np.float32)
    # Normalize to [0,1] to keep scale stable
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr)
    return arr

def update_tag_preferences(tag_pref: Dict[str, float], course: Dict[str, Any], delta: float = 1.0):
    for t in course.get("tags", []):
        tag_pref[t] = tag_pref.get(t, 0.0) + delta

def adjust_user_vector(user_vec: np.ndarray, liked_vecs: List[np.ndarray], disliked_vecs: List[np.ndarray]) -> np.ndarray:
    # Simple vector-based adjustment heuristic
    if len(liked_vecs) > 0:
        like_mean = np.mean(liked_vecs, axis=0)
        user_vec = user_vec + 0.1 * like_mean
    if len(disliked_vecs) > 0:
        dislike_mean = np.mean(disliked_vecs, axis=0)
        user_vec = user_vec - 0.1 * dislike_mean
    # Renormalize
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm
    return user_vec

# ----------------- Q&A (RAG-lite) -----------------
def retrieve_kb_chunks(model, kb_text: str, query: str, k: int = 3) -> List[str]:
    # Fast numpy similarity for both local and OpenAI embeddings
    chunks = [ch.strip() for ch in kb_text.split("\n\n") if ch.strip()]
    chunk_embs = embed_texts(model, chunks)
    q_vec = embed_texts(model, [query])[0]
    sims = chunk_embs @ q_vec.T
    idxs = sims.argsort()[::-1][:k]
    return [chunks[i] for i in idxs]

def answer_question_with_context(query: str, context_chunks: List[str], user_profile: Dict[str, str]) -> str:
    profile_line = build_user_text(user_profile)
    context = "\n---\n".join(context_chunks)
    if USE_OPENAI:
        # Use GPT-4o for Q&A
        prompt = (
            f"User Context: {profile_line}\n"
            f"Knowledge Base:\n{context}\n"
            f"Question: {query}\n"
            "Answer in a detailed, practical, and personalized way. "
            "Provide actionable steps, learning paths, and career advice. "
            "If relevant, recommend specific courses, skills, or resources. "
            "Be concise but thorough, and always ground your answer in the provided knowledge base."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert course recommender, learning path designer, and career advisor. You provide practical, actionable, and personalized advice for learners in data science, machine learning, and related fields. Always ground your answers in the provided knowledge base and user context."},
                          {"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI error: {e}]\nFallback to local answer.\n" + _local_answer_question_with_context(query, context_chunks, user_profile)
    else:
        return _local_answer_question_with_context(query, context_chunks, user_profile)

# Local fallback for Q&A
def _local_answer_question_with_context(query: str, context_chunks: List[str], user_profile: Dict[str, str]) -> str:
    profile_line = build_user_text(user_profile)
    preface = f"User Context: {profile_line}\nQuestion: {query}\n"
    context = "\n---\n".join(context_chunks)
    key_terms = [w.lower() for w in query.split() if len(w) > 3]
    selected_lines = []
    for ch in context_chunks:
        for line in ch.split("\n"):
            if any(kt in line.lower() for kt in key_terms):
                selected_lines.append(line.strip())
    selected = "\n".join(selected_lines[:6]) if selected_lines else context_chunks[0]
    guidance = (
        "Answer (grounded in retrieved notes, personalized to the user's context):\n"
        f"{selected}\n\n"
        "Personalization Tip: prioritize Python and practical projects if aiming for ML engineering; "
        "include MLOps and deployment. For research/statistics-heavy roles, emphasize R and statistical foundations."
    )
    return preface + "\n" + guidance

# ----------------- Streamlit UI -----------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)
    st.caption("Embeddings-powered recommendations with feedback learning + RAG-style Q&A")

    # Sidebar: profile
    with st.sidebar:
        st.header("User Profile")
        user_name = st.text_input("Your name / handle", value="guest")
        user_id = user_name.strip().lower() or "guest"
        background = st.text_input("Background", value="Final-year CS student")
        interests = st.text_input("Interests", value="AI, Data Science, MLOps")
        goals = st.text_input("Career Goals", value="Become an ML Engineer at a product company")
        skills = st.text_input("Self-rated skills (optional)", value="Python: intermediate, Math: beginner, SQL: intermediate")
        difficulty_filter = st.selectbox("Filter by skill level", ["any","beginner","intermediate","advanced"], index=0)
        provider_filter = st.multiselect("Filter by provider", options=["Coursera","Udemy","edX","Udacity","Kaggle Learn","DataTalksClub","IBM","Google","Microsoft"], default=[])
        cost_filter = st.multiselect("Cost", options=["free","paid"], default=[])
        top_k = st.slider("Top K recommendations", min_value=3, max_value=10, value=5, step=1)

    courses = load_courses()
    kb_text = load_kb()
    store = load_feedback()
    store = ensure_user(store, user_id)

    model = get_model()

    # Build profile & embedding
    profile = {"background": background, "interests": interests, "goals": goals, "skills": skills}
    user_text = build_user_text(profile)
    user_vec = embed_texts(model, [user_text])[0]

    # Apply vector adjustments from previous likes/dislikes
    liked_vecs, disliked_vecs = [], []
    if store[user_id]["liked_courses"]:
        liked_ids = set(store[user_id]["liked_courses"])
        liked_vecs = embed_texts(model, [c["description"] for c in courses if c["id"] in liked_ids])
    if store[user_id]["disliked_courses"]:
        disliked_ids = set(store[user_id]["disliked_courses"])
        disliked_vecs = embed_texts(model, [c["description"] for c in courses if c["id"] in disliked_ids])
    user_vec = adjust_user_vector(user_vec, liked_vecs, disliked_vecs)

    # Compute base scores from tag preferences
    base_scores = compute_tag_base_scores(courses, store[user_id]["tag_preferences"])

    # Embed courses once (cache across runs)
    @st.cache_data(show_spinner=False)
    def get_course_matrix(cs):
        texts = [f"{c['title']} {c['description']} {' '.join(c.get('tags', []))}" for c in cs]
        return embed_texts(model, texts)
    course_vecs = get_course_matrix(courses)

    # Optional filters
    def course_filter(c):
        ok = True
        if difficulty_filter != "any":
            ok = ok and c["skill_level"] == difficulty_filter
        if provider_filter:
            ok = ok and any(p.lower() in c["provider"].lower() for p in provider_filter)
        if cost_filter:
            ok = ok and (c["cost"] in cost_filter)
        return ok

    filtered_idxs = [i for i,c in enumerate(courses) if course_filter(c)]
    if not filtered_idxs:
        st.warning("No courses match the current filters. Try relaxing filters.")
        filtered_idxs = list(range(len(courses)))

    filtered_vecs = course_vecs[filtered_idxs]
    filtered_base = base_scores[filtered_idxs]

    # Score & rank
    scores = score_courses(user_vec, filtered_vecs, filtered_base)
    top_order = np.argsort(-scores)[:top_k]

    st.subheader("Recommended Courses")
    cols = st.columns(1)
    for rank, idx in enumerate(top_order, start=1):
        course = courses[filtered_idxs[idx]]
        with st.container(border=True):
            st.markdown(f"**#{rank}. {course['title']}**  \n*{course['provider']}*  •  Level: `{course['skill_level']}`  •  Cost: `{course['cost']}`  •  ~{course['duration_hours']}h")
            st.write(course["description"])
            st.write("Tags:", ", ".join(course["tags"]))
            st.link_button("Open course", course["url"])
            c1, c2, c3 = st.columns([1,1,3])
            with c1:
                if st.button(f"👍 Like {course['id']}", key=f"like_{course['id']}"):
                    if course["id"] not in store[user_id]["liked_courses"]:
                        store[user_id]["liked_courses"].append(course["id"])
                    if course["id"] in store[user_id]["disliked_courses"]:
                        store[user_id]["disliked_courses"].remove(course["id"])
                    update_tag_preferences(store[user_id]["tag_preferences"], course, delta=+1.0)
                    save_feedback(store)
                    st.success("Recorded like & updated preferences.")
                    st.rerun()
            with c2:
                if st.button(f"👎 Dislike {course['id']}", key=f"dislike_{course['id']}"):
                    if course["id"] not in store[user_id]["disliked_courses"]:
                        store[user_id]["disliked_courses"].append(course["id"])
                    if course["id"] in store[user_id]["liked_courses"]:
                        store[user_id]["liked_courses"].remove(course["id"])
                    update_tag_preferences(store[user_id]["tag_preferences"], course, delta=-0.7)
                    save_feedback(store)
                    st.warning("Recorded dislike & updated preferences.")
                    st.rerun()
            with c3:
                st.caption(f"Rationale score: {scores[idx]:.3f}  •  Tag bias: {filtered_base[idx]:.2f}")

    st.divider()
    st.subheader("Q&A: Ask about learning paths")
    user_q = st.text_input("Ask a question (e.g., 'What are the steps to become an ML engineer?')", "")
    if user_q:
        chunks = retrieve_kb_chunks(model, kb_text, user_q, k=3)
        answer = answer_question_with_context(user_q, chunks, profile)
        with st.expander("Retrieved context"):
            for i, ch in enumerate(chunks, 1):
                st.markdown(f"**Chunk {i}**")
                st.write(ch)
        st.markdown("### Answer")
        st.write(answer)

    # Sidebar: debug & memory
    with st.sidebar:
        st.divider()
        st.header("Your Preferences (live)")
        st.json(store[user_id]["tag_preferences"])
        st.write("Likes:", store[user_id]["liked_courses"])
        st.write("Dislikes:", store[user_id]["disliked_courses"])
        if st.button("Reset my feedback"):
            store[user_id] = {
                "tag_preferences": {},
                "liked_courses": [],
                "disliked_courses": [],
                "profile_history": []
            }
            save_feedback(store)
            st.info("Preferences cleared.")
            st.rerun()

if __name__ == "__main__":
    main()
