"""
Semantic Similarity-Based Detection of AI-Paraphrased Academic Plagiarism
--------------------------------------------------------------------------
Seminar demo script.

What this does:
1. Takes an ORIGINAL passage and one or more SUSPECT passages (e.g. a
   verbatim copy, an AI-paraphrased version, and an unrelated passage).
2. Computes a simple LEXICAL overlap score (word-matching, like a basic
   plagiarism checker would do).
3. Computes a SEMANTIC similarity score using Sentence-BERT embeddings +
   cosine similarity (the approach this seminar proposes).
4. Prints both side by side, so you can show live that AI-paraphrased text
   can have LOW lexical overlap but HIGH semantic similarity.

HOW TO RUN (VS Code):
1. Open a terminal in this file's folder.
2. Create a virtual environment (recommended):
       python -m venv venv
       venv\\Scripts\\activate        (Windows)
       source venv/bin/activate     (Mac/Linux)
3. Install dependencies:
       pip install -r requirements.txt
4. Run:
       python plagiarism_demo.py
   (First run will download the model, ~90 MB — needs internet once.)

EDIT ME:
Replace the placeholder text in ORIGINAL and SUSPECTS below with your own
real passages before your live demo / guide meeting.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------------------------------------------------------------
# 1. PASSAGES TO EDIT
# ---------------------------------------------------------------------
ORIGINAL = (
    "There is a broad agreement that climate change is accelerating due to human activities."
)

SUSPECTS = {
    "Verbatim copy":
    "There is a broad agreement that climate change is accelerating due to human activities.",


    "AI-paraphrased version":
    "A widespread consensus exists that human actions are driving the acceleration of global climate change.",

    "Unrelated passage":
        "Climate change is why elon musk is existing as the richest guy",
}

THRESHOLD = 0.80  # similarity score above this = flagged

# ---------------------------------------------------------------------
# 2. LEXICAL OVERLAP (simple word-matching baseline, like Turnitin-style checkers)
# ---------------------------------------------------------------------
def lexical_overlap(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"\w+", text_a.lower()))
    words_b = set(re.findall(r"\w+", text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    shared = words_a & words_b
    return len(shared) / len(words_a | words_b)  # Jaccard overlap, 0 to 1


# ---------------------------------------------------------------------
# 3. SEMANTIC SIMILARITY (Sentence-BERT + cosine similarity)
# ---------------------------------------------------------------------
print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_similarity(text_a: str, text_b: str) -> float:
    embeddings = model.encode([text_a, text_b])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)


# ---------------------------------------------------------------------
# 4. RUN THE COMPARISON AND PRINT RESULTS
# ---------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"{'Passage Pair':<28}{'Lexical Overlap':>16}{'Semantic Similarity':>20}{'Flag':>10}")
print("-" * 78)

for label, suspect_text in SUSPECTS.items():
    lex_score = lexical_overlap(ORIGINAL, suspect_text)
    sem_score = semantic_similarity(ORIGINAL, suspect_text)
    flag = "FLAGGED" if sem_score >= THRESHOLD else "-"
    print(f"{label:<28}{lex_score*100:>14.1f}%{sem_score:>19.2f}{flag:>10}")

print("=" * 78)
print(f"\nThreshold used: {THRESHOLD}  (semantic similarity >= threshold -> flagged)")
print("Tip: the 'AI-paraphrased version' row is your key seminar result --")
print("it should show LOW lexical overlap but HIGH semantic similarity.")
