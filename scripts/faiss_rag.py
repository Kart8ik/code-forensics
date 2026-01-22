"""
FAISS RAG Layer for Code Forensics
Repo-scoped, diff-only embedding with windowed retrieval

Architectural Constraints:
- FAISS embeds ONLY code diffs (no clusters, metrics, or file summaries)
- All documents are scoped to (repo, file, commit)
- Retrieval filters by repo + file BEFORE similarity search
- Cross-repo and cross-file retrieval is forbidden
"""

import json
import csv
import re
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FAISSContextRetriever:
    """
    Diff-only FAISS retriever with strict repo/file scoping.
    
    Epistemic boundaries:
    - FAISS ranks candidate diffs within a pre-selected commit window
    - NOT for discovery, comparison, or similarity across files
    - All retrieval is scoped to (repo_name, file_path)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with embedding model.
        
        Args:
            model_name: HuggingFace model for embeddings
                       all-MiniLM-L6-v2: Fast, 384 dims, good quality
        """
        print("=" * 60)
        print("FAISS RAG Layer Initialization")
        print("Diff-only, repo-scoped retrieval")
        print("=" * 60)
        
        print(f"\nLoading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Embedding dimension: {self.embedding_dim}")
        
        # Storage
        self.index: Optional[faiss.IndexFlatL2] = None
        self.documents: List[str] = []  # Stores actual diff text
        self.metadata: List[Dict[str, Any]] = []  # Stores (repo, file, commit, type)
        
        # Index for fast lookup by (repo, file)
        self._file_index: Dict[tuple, List[int]] = {}
        
        # Load commits for message lookup
        self._commits: Dict[str, Dict] = {}
        
        print("=" * 60 + "\n")
    
    def build_index(self) -> None:
        """
        Build FAISS index from code diffs ONLY.
        
        What gets embedded:
        - Code diffs scoped to (repo_name, file_path, commit_hash)
        
        What is NOT embedded (by design):
        - Cluster descriptions
        - File evolution summaries  
        - Metric summaries
        """
        print("Building FAISS index from diffs only...")
        print("-" * 60)
        
        # Load commits for message context
        with open(DATA_DIR / "commits.csv", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                self._commits[row['commit_hash']] = row
        
        # Load diffs
        diffs = {}
        with open(DATA_DIR / "diffs.csv", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                key = (row['repo_name'], row['commit_hash'], row['file_path'])
                diffs[key] = row['diff_text']
        
        print(f"Loaded {len(self._commits)} commits, {len(diffs)} diffs")
        print("-" * 60)
        
        # Embed ONLY diff contexts (no clusters, no file summaries, no metrics)
        print("\nEmbedding diff contexts (diff-only mode)...")
        diff_count = 0
        skipped_trivial = 0
        
        for (repo, commit_hash, filepath), diff_text in diffs.items():
            # Skip trivial diffs (<50 chars)
            if len(diff_text) < 50:
                skipped_trivial += 1
                continue
            
            commit_msg = self._commits.get(commit_hash, {}).get('message', 'No message')
            doc = self._build_diff_document(
                repo, filepath, commit_hash, commit_msg, diff_text
            )
            
            doc_idx = len(self.documents)
            self.documents.append(doc)
            
            # Strict metadata schema as per architectural spec
            self.metadata.append({
                "repo": repo,
                "file": filepath,
                "commit": commit_hash,
                "type": "diff"
            })
            
            # Build file index for fast lookup
            file_key = (repo, filepath)
            if file_key not in self._file_index:
                self._file_index[file_key] = []
            self._file_index[file_key].append(doc_idx)
            
            diff_count += 1
        
        print(f"  ✓ Added {diff_count} diff documents")
        print(f"  ✓ Skipped {skipped_trivial} trivial diffs (<50 chars)")
        
        # Create embeddings
        print(f"\n{'=' * 60}")
        print(f"Total documents: {len(self.documents)}")
        print("Creating embeddings...")
        
        embeddings = self.model.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Build FAISS index
        print("\nBuilding FAISS index...")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors (diff-only)")
        print("=" * 60 + "\n")
    
    def _build_diff_document(self, repo: str, filepath: str,
                             commit: str, msg: str, diff: str) -> str:
        """Build searchable document for diff."""
        # Truncate diff intelligently (keep context)
        diff_preview = diff[:500] if len(diff) > 500 else diff
        
        return f"""
Change in {filepath}
Repository: {repo}
Commit: {commit[:8]}
Message: {msg[:200]}
Diff preview:
{diff_preview}
        """.strip()
    
    def retrieve_context(self, file_data: Dict, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant diffs for a specific file with repo/file locality enforcement.
        
        Args:
            file_data: File from file_analysis.json
            k: Number of relevant documents to retrieve
        
        Returns:
            List of relevant diff documents with metadata
            
        Constraints:
            - Only retrieves from same repo AND same file
            - Cross-file retrieval is forbidden
        """
        repo = file_data['repo_name']
        filepath = file_data['file_path']
        
        logger.info(f"Retrieving context for {repo}/{filepath}")
        
        # Get valid indices for this repo+file (locality enforcement)
        file_key = (repo, filepath)
        valid_indices = self._file_index.get(file_key, [])
        
        if not valid_indices:
            logger.warning(f"No diffs found for {repo}/{filepath}")
            return []
        
        # Build query from file characteristics
        query = self._build_file_query(file_data)
        query_embedding = self.model.encode([query])[0]
        
        # Search FAISS with k = min(k, len(valid_indices)) 
        # We search globally but filter to valid indices
        search_k = min(k * 5, self.index.ntotal)  # Search more, then filter
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            search_k
        )
        
        # Filter to only valid indices (same repo + same file)
        valid_set = set(valid_indices)
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx in valid_set:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(1 / (1 + dist))  # Convert L2 to similarity
                })
                if len(results) >= k:
                    break
        
        logger.info(f"Retrieved {len(results)} diffs for {filepath}")
        return results
    
    def retrieve_windowed_diffs(
        self,
        file_data: Dict,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve diffs only from inflection commits listed in file_data.
        
        This method:
        - Retrieves diffs only from commits where ML signals indicate metric inflection
        - Returns at most k diffs
        - Skips trivial diffs (already filtered during indexing)
        
        Args:
            file_data: File data containing 'inflection_commits' list
            k: Maximum number of diffs to retrieve (default: 3)
        
        Returns:
            List of diffs from inflection commits, ranked by relevance
        """
        repo = file_data['repo_name']
        filepath = file_data['file_path']
        inflection_commits = file_data.get('inflection_commits', [])
        
        logger.info(f"Windowed retrieval for {filepath} with {len(inflection_commits)} inflection commits")
        
        if not inflection_commits:
            logger.warning(f"No inflection_commits provided for {filepath}, falling back to standard retrieval")
            return self.retrieve_context(file_data, k=k)
        
        # Get valid indices for this repo+file
        file_key = (repo, filepath)
        valid_indices = self._file_index.get(file_key, [])
        
        if not valid_indices:
            logger.warning(f"No diffs indexed for {repo}/{filepath}")
            return []
        
        # Filter to only inflection commits
        inflection_set = set(inflection_commits)
        candidate_indices = [
            idx for idx in valid_indices
            if self.metadata[idx]['commit'] in inflection_set
        ]
        
        if not candidate_indices:
            logger.warning(f"No diffs found for inflection commits in {filepath}")
            return []
        
        # If we have few candidates, return them all (up to k)
        if len(candidate_indices) <= k:
            results = []
            for idx in candidate_indices:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': 1.0  # Direct match to inflection commit
                })
            logger.info(f"Returning {len(results)} diffs from inflection commits")
            return results
        
        # If we have more candidates than k, rank by similarity
        query = self._build_file_query(file_data)
        query_embedding = self.model.encode([query])[0]
        
        # Get embeddings for candidates and rank
        candidate_embeddings = []
        for idx in candidate_indices:
            # Reconstruct embedding (expensive but necessary for ranking)
            doc_embedding = self.model.encode([self.documents[idx]])[0]
            candidate_embeddings.append(doc_embedding)
        
        candidate_embeddings = np.array(candidate_embeddings).astype('float32')
        
        # Compute distances
        distances = np.linalg.norm(candidate_embeddings - query_embedding, axis=1)
        
        # Sort by distance (lower = more similar)
        sorted_indices = np.argsort(distances)[:k]
        
        results = []
        for rank_idx in sorted_indices:
            idx = candidate_indices[rank_idx]
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': float(1 / (1 + distances[rank_idx]))
            })
        
        logger.info(f"Returning {len(results)} ranked diffs from inflection commits")
        return results
    
    def get_commits_for_file(self, repo: str, filepath: str) -> List[str]:
        """Get all indexed commit hashes for a specific file."""
        file_key = (repo, filepath)
        indices = self._file_index.get(file_key, [])
        return [self.metadata[idx]['commit'] for idx in indices]
    
    def _build_file_query(self, file_data: Dict) -> str:
        """Build search query from file data."""
        return f"""
Find code changes related to:
File: {file_data['file_path']}
Repository: {file_data['repo_name']}
Behavior: {file_data.get('churn_label', 'unknown')}, anomaly score {file_data.get('anomaly_score', 0):.3f}
        """.strip()
    
    def save_index(self, output_dir: str = "data") -> None:
        """Save FAISS index and metadata."""
        output_path = DATA_DIR
        output_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_path / "faiss.index"))
        
        # Save documents, metadata, and file index
        with open(output_path / "faiss_documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim,
                'file_index': self._file_index,
                'commits': self._commits
            }, f)
        
        print(f"✓ FAISS index saved to {output_path}/")
    
    def load_index(self, data_dir: str = "data") -> None:
        """Load pre-built FAISS index."""
        data_path = DATA_DIR
        
        # Load FAISS index
        self.index = faiss.read_index(str(data_path / "faiss.index"))
        
        # Load documents, metadata, and file index
        with open(data_path / "faiss_documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
            self._file_index = data.get('file_index', {})
            self._commits = data.get('commits', {})
        
        # Rebuild file index if not present (backward compatibility)
        if not self._file_index:
            logger.info("Rebuilding file index from metadata...")
            for idx, meta in enumerate(self.metadata):
                if meta.get('type') == 'diff':
                    file_key = (meta['repo'], meta['file'])
                    if file_key not in self._file_index:
                        self._file_index[file_key] = []
                    self._file_index[file_key].append(idx)
        
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors (diff-only)")


def build_faiss_index() -> None:
    """Standalone script to build FAISS index."""
    print("\n" + "=" * 60)
    print("BUILDING FAISS INDEX (DIFF-ONLY MODE)")
    print("=" * 60 + "\n")
    
    retriever = FAISSContextRetriever()
    retriever.build_index()
    retriever.save_index()
    
    print("\n" + "=" * 60)
    print("✓ FAISS index built successfully!")
    print("=" * 60)
    print("\nTest retrieval:")
    print("-" * 60)
    
    # Test with first file
    with open(DATA_DIR / "file_analysis.json") as f:
        files = json.load(f)
    
    if files:
        test_file = files[0]
        print(f"\nQuerying: {test_file['file_path']}")
        print(f"Repo: {test_file['repo_name']}")
        
        results = retriever.retrieve_context(test_file, k=3)
        
        print(f"\nTop {len(results)} relevant diffs (same file only):")
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            print(f"\n{i}. Commit: {meta['commit'][:8]}")
            print(f"   File: {meta['file']}")
            print(f"   Repo: {meta['repo']}")
            print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Preview: {result['document'][:100]}...")
        
        # Test windowed retrieval
        print("\n" + "-" * 60)
        print("Testing windowed diff retrieval...")
        
        # Simulate inflection commits (use first 2 available commits for this file)
        available_commits = retriever.get_commits_for_file(
            test_file['repo_name'], 
            test_file['file_path']
        )
        if available_commits:
            test_file_with_inflection = {**test_file, 'inflection_commits': available_commits[:2]}
            windowed_results = retriever.retrieve_windowed_diffs(test_file_with_inflection, k=2)
            
            print(f"\nWindowed results ({len(windowed_results)} diffs from inflection commits):")
            for i, result in enumerate(windowed_results, 1):
                meta = result['metadata']
                print(f"\n{i}. Commit: {meta['commit'][:8]}")
                print(f"   Similarity: {result['similarity']:.3f}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    build_faiss_index()
