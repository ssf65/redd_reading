import os
import json
import math
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from better_profanity import profanity

# Load profanity filter
profanity.load_censor_words()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load JSON data
current_directory = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_directory, "init.json")
with open(json_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

title_to_id = {}
id_to_meta = {}
titles = []
post_ids = []

# Build title-ID mapping and title list
for item in raw_data:
    post_id = item["post_url"].split("/")[-1]
    title = item["title"]
    title_to_id[title] = post_id
    id_to_meta[post_id] = item
    titles.append(title)
    post_ids.append(post_id)

# Utility: tokenize + stop-word removal for Jaccard
stop_words = ENGLISH_STOP_WORDS

def tokenize(text):
    clean = re.sub(r"[^\w\s]", "", text.lower())
    return set(w for w in clean.split() if w and w not in stop_words)

# Precompute token sets for each title
title_token_sets = [tokenize(t) for t in titles]

@app.route("/")
def home():
    return render_template("base.html", title="Reddit Comment Search")

@app.route("/titles")
def get_titles():
    raw_q = request.args.get("q", "").strip()
    no_spoilers = request.args.get("no_spoilers", "false").lower() == "true"

    filtered_titles = []
    filtered_post_ids = []
    filtered_token_sets = []

    for i, title in enumerate(titles):
        if no_spoilers and "[SPOILERS]" in title.upper():
            continue
        filtered_titles.append(title)
        filtered_post_ids.append(post_ids[i])
        filtered_token_sets.append(title_token_sets[i])

    if not raw_q:
        return jsonify([
            {
                "title": filtered_titles[i],
                "post_id": filtered_post_ids[i],
                "num_comments": id_to_meta[filtered_post_ids[i]]["num_comments"]
            }
            for i in range(len(filtered_titles))
        ])

    q_tokens = tokenize(raw_q)
    if not q_tokens:
        return jsonify([])

    jaccard_scores = []
    for ts in filtered_token_sets:
        if not ts:
            jaccard_scores.append(0.0)
        else:
            inter = q_tokens & ts
            union = q_tokens | ts
            jaccard_scores.append(len(inter) / len(union))

    ranked = [
        i
        for i in sorted(range(len(jaccard_scores)), key=lambda i: jaccard_scores[i], reverse=True)
        if jaccard_scores[i] > 0
    ]

    return jsonify([
        {
            "title": filtered_titles[i],
            "post_id": filtered_post_ids[i],
            "num_comments": id_to_meta[filtered_post_ids[i]]["num_comments"],
            "jaccard": round(jaccard_scores[i], 4)
        }
        for i in ranked
    ])


@app.route("/search_comments")
def search_comments():
    post_ids = request.args.getlist("post_ids")
    query = request.args.get("q", "")
    filter_flag = request.args.get("filter_offensive", "false").lower() == "true"

    if not post_ids or not query.strip():
        return jsonify([])

    all_comments = []
    for pid in post_ids:
        if pid in id_to_meta:
            comments = id_to_meta[pid].get("comments", [])
            for comment in comments:
                if "text" in comment and comment["text"].strip():
                    all_comments.append(comment)

    if not all_comments:
        return jsonify([])

    df = pd.DataFrame(all_comments)
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.strip().str.lower().ne("[deleted]")]
    df = df[df["text"].str.strip() != ""]

    texts = df["text"].tolist()
    if not texts:
        return jsonify([])

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        token_pattern=r"(?u)\b\w\w+\b"
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return jsonify([])

    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = tfidf_scores.argsort()[-100:][::-1].copy()

    svd = TruncatedSVD(n_components=5, random_state=42)
    try:
        svd_matrix = svd.fit_transform(tfidf_matrix[top_indices])
    except ValueError:
        return jsonify([])

    components = svd.components_
    feature_names = vectorizer.get_feature_names_out()

    dimension_labels = []
    for comp in components:
        sorted_indices = np.argsort(comp)[::-1]
        filtered_terms = []
        for idx in sorted_indices:
            term = feature_names[idx].lower()
            if term not in ["deleted", "ve", "im", "dont", "didn", "http", "https"]:
                filtered_terms.append(term)
            if len(filtered_terms) == 3:
                break
        dimension_labels.append("/".join(filtered_terms))

    results = []
    for i, idx in enumerate(top_indices[:20]):
        row = df.iloc[idx]
        text = row["text"]
        if filter_flag:
            text = profanity.censor(text)

        doc_vector = svd_matrix[i]
        label_details = {
            f"Dim{d+1}: {dimension_labels[d]}": round(doc_vector[d], 3)
            for d in range(len(dimension_labels))
        }

        length_boost = math.log(1 + len(text.split()))
        final_score = tfidf_scores[idx] * length_boost

        if final_score > 0:
            results.append({
                "author": row.get("user", "unknown"),
                "text": text,
                "depth": 0,
                "tfidf_score": round(tfidf_scores[idx], 4),
                "length_boost": round(length_boost, 4),
                "final_score": round(final_score, 4),
                "label_details": label_details,
                "permalink": row.get("full_permalink", "")
            })


    return jsonify(sorted(results, key=lambda r: r["final_score"], reverse=True))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
