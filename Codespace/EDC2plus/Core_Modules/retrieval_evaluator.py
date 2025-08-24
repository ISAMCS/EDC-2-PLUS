import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

class RetrievalEvaluator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", alpha=0.3, beta=0.6, gamma=0.1, thresholds=None):
        """
        alpha: lexical overlap weight
        beta: semantic similarity weight
        gamma: coverage weight
        thresholds: dict for confidence levels, e.g. {"high": 0.55, "med": 0.25}
        """
        try:
            self.embedder = SentenceTransformer(model_name)
            self.has_embedder = True
        except Exception:
            self.embedder = None
            self.has_embedder = False
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.thresholds = thresholds or {"high": 0.55, "med": 0.25}

    def lexical_overlap(self, query, passage):
        try:
            q_tokens = set(word_tokenize(query.lower()))
            p_tokens = set(word_tokenize(passage.lower()))
        except Exception:
            q_tokens = set(query.lower().split())
            p_tokens = set(passage.lower().split())
        if not q_tokens or not p_tokens:
            return 0.0
        return len(q_tokens & p_tokens) / len(q_tokens | p_tokens)

    def semantic_similarity(self, query, passage):
        if self.has_embedder:
            embeddings = self.embedder.encode([query, passage])
            return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        # Fallback: cosine on bag-of-words
        def bow(text):
            return np.array([hash(w) % 1000 for w in text.lower().split()])
        q_vec = bow(query)
        p_vec = bow(passage)
        if len(q_vec) == 0 or len(p_vec) == 0:
            return 0.0
        return float(np.dot(q_vec, p_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(p_vec) + 1e-8))

    def hybrid_score(self, query, passage):
        lex = self.lexical_overlap(query, passage)
        sem = self.semantic_similarity(query, passage)
        return self.alpha * lex + self.beta * sem

    def coverage_score(self, query, passages):
        # Coverage: fraction of passages with any lexical or semantic match
        matches = 0
        for p in passages:
            lex = self.lexical_overlap(query, p)
            sem = self.semantic_similarity(query, p)
            if lex > 0.1 or sem > 0.5:
                matches += 1
        return matches / max(1, len(passages))

    def evaluate(self, query, retrieved_passages, dev_scores=None):
        """
        Calibrated continuous confidence score using normalized semantic similarity, consensus, and adaptive thresholds.
        Returns: (confidence_label, score)
        """
        if not retrieved_passages:
            return "low", 0.0
        hybrid_scores = [self.hybrid_score(query, p) for p in retrieved_passages]
        # Normalize hybrid scores to [0,1]
        min_h, max_h = min(hybrid_scores), max(hybrid_scores)
        norm_scores = [(s - min_h) / (max_h - min_h + 1e-8) for s in hybrid_scores]
        best = max(norm_scores)
        avg = np.mean(norm_scores)
        coverage = self.coverage_score(query, retrieved_passages)
        # Salient: 75th percentile of norm_scores
        salient_thresh = np.percentile(norm_scores, 75)
        salient = sum(1 for s in norm_scores if s >= salient_thresh)
        final_score = 0.5 * best + 0.2 * avg + 0.3 * coverage
        # Adaptive thresholds: use dev-set percentiles if provided
        if dev_scores is not None and len(dev_scores) > 10:
            high_th = np.percentile(dev_scores, 70)
            med_th = np.percentile(dev_scores, 40)
        else:
            high_th = self.thresholds.get("high", 0.55)
            med_th = self.thresholds.get("med", 0.25)
        if final_score >= high_th:
            return "high", final_score
        elif final_score >= med_th:
            return "med", final_score
        else:
            return "low", final_score
