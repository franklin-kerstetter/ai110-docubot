"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "at", "by",
    "for", "with", "from", "up", "about", "as", "if", "it",
    "that", "this", "these", "those", "which", "who", "what",
    "when", "where", "why", "how", "all", "each", "every",
    "both", "not", "no", "nor", "only", "same", "then", "just"
}

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Tokenization helpers
    # -----------------------------------------------------------

    def _tokenize_as_set(self, text):
        """Extract tokens from text (lowercase, stopwords removed). Returns set."""
        tokens = re.findall(r"\w+", text.lower())
        return {t for t in tokens if t not in STOPWORDS}

    def _tokenize_as_list(self, text):
        """Extract tokens from text (lowercase, stopwords removed). Returns list."""
        tokens = re.findall(r"\w+", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build an inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": {"AUTH.md", "API_REFERENCE.md"},
            "database": {"DATABASE.md"}
        }

        Tokenizes on word boundaries, lowercases, filters stopwords.
        """
        index = {}
        for filename, text in documents:
            tokens = self._tokenize_as_set(text)
            for token in tokens:
                if token not in index:
                    index[token] = set()
                index[token].add(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.

        Count total occurrences of query tokens in text token list.
        Higher count = higher relevance.
        """
        query_tokens = self._tokenize_as_set(query)
        text_tokens = self._tokenize_as_list(text)
        score = sum(1 for token in text_tokens if token in query_tokens)
        return score

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document snippets.

        Tokenize query, find candidate docs from index, score each, sort descending.
        Return a list of (filename, text) sorted by score descending.
        """
        query_tokens = self._tokenize_as_set(query)

        if not query_tokens:
            return []

        candidate_files = set()
        for token in query_tokens:
            if token in self.index:
                candidate_files.update(self.index[token])

        scored_docs = []
        for filename, text in self.documents:
            if filename in candidate_files:
                score = self.score_document(query, text)
                scored_docs.append((score, filename, text))

        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [(filename, text) for _, filename, text in scored_docs[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
