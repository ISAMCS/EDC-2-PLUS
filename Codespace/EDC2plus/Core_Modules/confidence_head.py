import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sentence_transformers import SentenceTransformer
from Codespace.EDC2plus.Core_Modules.answer_postprocess import canonicalize_answer
import re

class ConfidenceHead:
    def __init__(self, embedder_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_name)
        self.calibrator = LogisticRegression()
        self.isotonic = IsotonicRegression(out_of_bounds="clip")
        self.trained = False

    def semantic_entropy(self, answers):
        # Embed answers and compute mean pairwise cosine similarity (lower = higher confidence)
        if len(answers) < 2:
            return 0.0  # No disagreement → low uncertainty
        embs = self.embedder.encode(answers)
        sims = []
        for i in range(len(embs)):
            for j in range(i+1, len(embs)):
                sim = np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j]) + 1e-8)
                sims.append(sim)
        mean_sim = np.mean(sims) if sims else 0.0
        return 1.0 - mean_sim  # Lower entropy = higher confidence

    def faithfulness(self, candidate, quotes, ans_type=None):
        canon = canonicalize_answer(candidate, ans_type)
        support = 0
        for q in quotes:
            # Case-insensitive, word-boundary match
            if canon and re.search(rf"\b{re.escape(canon)}\b", q, re.IGNORECASE):
                support += 1
        return support / max(1, len(quotes))

    def retrieval_sufficiency(self, retrieval_score, coverage):
        # Blend retrieval score and coverage
        return 0.5 * retrieval_score + 0.5 * coverage

    def extract_features(self, candidate, answers, quotes, retrieval_score, coverage, ans_type=None):
        return [
            self.semantic_entropy(answers),
            self.faithfulness(candidate, quotes, ans_type=ans_type),
            self.retrieval_sufficiency(retrieval_score, coverage)
        ]

    def fit(self, X, y):
        # X: feature matrix, y: correctness labels
        self.calibrator.fit(X, y)
        probs = self.calibrator.predict_proba(X)[:, 1]
        self.isotonic.fit(probs, y)
        self.trained = True

    def predict_proba(self, features, ans_type=None):
        # features: [semantic_entropy, faithfulness, retrieval_sufficiency]
        if not self.trained:
            # Use a sensible heuristic fallback
            sem_entropy = features[0]
            faith = features[1]
            # Optionally blend with retrieval_sufficiency
            prob = 0.5 * (1 - sem_entropy) + 0.5 * faith
            return prob
        prob = self.calibrator.predict_proba([features])[0, 1]
        return float(self.isotonic.transform([prob])[0])

    def conformal_abstain(self, prob, error_budget=0.1):
        # Abstain if confidence is below threshold (budget)
        return prob < error_budget
