import os
import json
import uuid
import time
import datetime
from typing import List, Dict, Any, Tuple
import httpx
import re

import streamlit as st

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# OpenAI integration
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI = bool(OPENAI_API_KEY)
if USE_OPENAI:
    # Disabling SSL verification for demo purposes (not recommended for production)
    client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client(verify=False))
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
@st.cache_resource
def get_model():
    if USE_OPENAI:
        return None  # OpenAI handles embeddings
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_texts(model, texts: List[str]) -> np.ndarray:
    if USE_OPENAI:
        emb_list = []
        for text in texts:
            resp = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            emb_list.append(np.array(resp.data[0].embedding, dtype=np.float32))
        arr = np.stack(emb_list)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms == 0, 1, norms)
        return arr
    else:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)


# ----------------- Recommendation Core -----------------
def build_user_text(profile: Dict[str, str]) -> str:
    parts = []
    for k in ["background", "interests", "goals", "skills"]:
        if profile.get(k):
            parts.append(f"{k}: {profile[k]}")
    return " | ".join(parts)

def score_courses(user_vec: np.ndarray, course_vecs: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    sim = (course_vecs @ user_vec.T).flatten()
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
    # Directly use raw tag scores, but penalize negative scores more strongly
    # If all scores are negative, keep them negative
    # Optionally, clip very negative scores to avoid extreme penalties
    arr = np.clip(arr, -10, 10)
    return arr

def update_tag_preferences(tag_pref: Dict[str, float], course: Dict[str, Any], delta: float = 1.0):
    for t in course.get("tags", []):
        tag_pref[t] = tag_pref.get(t, 0.0) + delta

def adjust_user_vector(user_vec: np.ndarray, liked_vecs: List[np.ndarray], disliked_vecs: List[np.ndarray]) -> np.ndarray:
    if len(liked_vecs) > 0:
        like_mean = np.mean(liked_vecs, axis=0)
        user_vec = user_vec + 0.1 * like_mean
    if len(disliked_vecs) > 0:
        dislike_mean = np.mean(disliked_vecs, axis=0)
        user_vec = user_vec - 0.1 * dislike_mean
    norm = np.linalg.norm(user_vec)
    if norm > 0:
        user_vec = user_vec / norm
    return user_vec


# ----------------- Keyword Search -----------------
def keyword_search(courses, query, top_k=5):
    query_terms = re.findall(r'\w+', query.lower())
    scored_courses = []
    for course in courses:
        score = 0
        title = course.get("title", "")
        description = course.get("description", "")
        tags = [t.lower() for t in course.get("tags", [])]
        text = (title + " " + description).lower()
        for term in query_terms:
            # Match whole tag only
            if term in tags:
                score += 2
            if term in text:
                score += 1
        if score > 0:
            scored_courses.append((score, course))
    scored_courses.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored_courses[:top_k]]


# ----------------- Q&A (RAG-lite) -----------------
def retrieve_kb_chunks(model, kb_text: str, query: str, k: int = 3) -> List[str]:
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
        prompt = (
            f"User Context: {profile_line}\n"
            f"Knowledge Base:\n{context}\n"
            f"Question: {query}\n"
            "Answer in a detailed, practical, and personalized way."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are an expert course recommender and career advisor."},
                          {"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI error: {e}]"
    else:
        return _local_answer_question_with_context(query, context_chunks, user_profile)

def _local_answer_question_with_context(query: str, context_chunks: List[str], user_profile: Dict[str, str]) -> str:
    profile_line = build_user_text(user_profile)
    return f"User Context: {profile_line}\nQ: {query}\nA: {context_chunks[0]}"


# ----------------- Streamlit UI -----------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🎓", layout="wide")
    st.title(APP_TITLE)
    st.caption("Embeddings-powered recommendations with feedback learning + RAG-style Q&A")

    with st.sidebar:
        st.header("User Profile")
        user_name = st.text_input("Your name / handle", value="guest")
        user_id = user_name.strip().lower() or "guest"
        background = st.text_input("Background", value="Final-year CS student")
        interests = st.text_input("Interests", value="AI, Data Science, MLOps")
        goals = st.text_input("Career Goals", value="Become an ML Engineer at a product company")
        skills = st.text_input("Skills", value="Python: intermediate, Math: beginner, SQL: intermediate")
        difficulty_filter = st.selectbox("Filter by skill level", ["any","beginner","intermediate","advanced"], index=0)
        provider_filter = st.multiselect("Filter by provider", options=["Coursera","Udemy","edX","Udacity","Google","Microsoft"], default=[])
        cost_filter = st.multiselect("Cost", options=["free","paid"], default=[])
        top_k = st.slider("Top K recommendations", min_value=3, max_value=10, value=5, step=1)
        search_mode = st.radio("Search Mode", ["Embedding-based", "Keyword", "Hybrid"], index=0)

    courses = load_courses()
    kb_text = load_kb()
    store = load_feedback()
    store = ensure_user(store, user_id)

    model = get_model()
    profile = {"background": background, "interests": interests, "goals": goals, "skills": skills}
    user_text = build_user_text(profile)
    user_vec = embed_texts(model, [user_text])[0]

    liked_vecs, disliked_vecs = [], []
    if store[user_id]["liked_courses"]:
        liked_ids = set(store[user_id]["liked_courses"])
        liked_vecs = embed_texts(model, [c["description"] for c in courses if c["id"] in liked_ids])
    if store[user_id]["disliked_courses"]:
        disliked_ids = set(store[user_id]["disliked_courses"])
        disliked_vecs = embed_texts(model, [c["description"] for c in courses if c["id"] in disliked_ids])
    user_vec = adjust_user_vector(user_vec, liked_vecs, disliked_vecs)

    base_scores = compute_tag_base_scores(courses, store[user_id]["tag_preferences"])

    @st.cache_data(show_spinner=False)
    def get_course_matrix(cs):
        texts = [f"{c['title']} {c['description']} {' '.join(c.get('tags', []))}" for c in cs]
        return embed_texts(model, texts)
    course_vecs = get_course_matrix(courses)

    def course_filter(c):
        ok = True
        skill_level = c.get("skill_level", "any")
        provider = c.get("provider", "")
        cost = c.get("cost", "")
        if difficulty_filter != "any":
            ok = ok and skill_level == difficulty_filter
        if provider_filter:
            ok = ok and any(p.lower() == provider.lower() for p in provider_filter)
        if cost_filter:
            ok = ok and (cost in cost_filter)
        return ok

    filtered_idxs = [i for i, c in enumerate(courses) if course_filter(c)]
    if not filtered_idxs:
        st.warning("No courses match the filters. Showing all courses.")
        filtered_idxs = list(range(len(courses)))

    filtered_vecs = course_vecs[filtered_idxs]
    filtered_base = base_scores[filtered_idxs]

    if search_mode == "Embedding-based":
        scores = score_courses(user_vec, filtered_vecs, filtered_base)
        top_order = np.argsort(-scores)[:top_k]
        ranked_courses = [courses[filtered_idxs[idx]] for idx in top_order]

    elif search_mode == "Keyword":
        ranked_courses = keyword_search([courses[i] for i in filtered_idxs], user_text, top_k=top_k)

    elif search_mode == "Hybrid":
        emb_scores = score_courses(user_vec, filtered_vecs, filtered_base)
        top_emb = np.argsort(-emb_scores)[:top_k]
        emb_results = [courses[filtered_idxs[idx]] for idx in top_emb]
        kw_results = keyword_search([courses[i] for i in filtered_idxs], user_text, top_k=top_k)
        combined = {}
        for rank, c in enumerate(emb_results):
            combined[c["id"]] = combined.get(c["id"], 0) + (top_k - rank) * 0.7
        for rank, c in enumerate(kw_results):
            combined[c["id"]] = combined.get(c["id"], 0) + (top_k - rank) * 0.3
        ranked_ids = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        ranked_courses = [next(c for c in courses if c["id"] == cid) for cid, _ in ranked_ids[:top_k]]

    st.subheader("Recommended Courses")
    for rank, course in enumerate(ranked_courses, start=1):
        with st.container(border=True):
            st.markdown(f"**#{rank}. {course['title']}**\n*{course['provider']}* • Level: `{course['skill_level']}` • Cost: `{course['cost']}` • ~{course['duration_hours']}h")
            st.write(course["description"])
            st.write("Tags:", ", ".join(course["tags"]))
            st.link_button("Open course", course["url"])
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button(f"👍 Like {course['id']}", key=f"like_{course['id']}"):
                    if course["id"] not in store[user_id]["liked_courses"]:
                        store[user_id]["liked_courses"].append(course["id"])
                    if course["id"] in store[user_id]["disliked_courses"]:
                        store[user_id]["disliked_courses"].remove(course["id"])
                    update_tag_preferences(store[user_id]["tag_preferences"], course, delta=+1.0)
                    save_feedback(store)
                    st.success("Liked & preferences updated.")
                    st.rerun()
            with c2:
                if st.button(f"👎 Dislike {course['id']}", key=f"dislike_{course['id']}"):
                    if course["id"] not in store[user_id]["disliked_courses"]:
                        store[user_id]["disliked_courses"].append(course["id"])
                    if course["id"] in store[user_id]["liked_courses"]:
                        store[user_id]["liked_courses"].remove(course["id"])
                    update_tag_preferences(store[user_id]["tag_preferences"], course, delta=-0.7)
                    save_feedback(store)
                    st.warning("Disliked & preferences updated.")
                    st.rerun()

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
