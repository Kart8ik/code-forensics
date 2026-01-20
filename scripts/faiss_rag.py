"""
FAISS RAG Layer for Code Forensics
Implements proper context retrieval with embeddings
"""

import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
class FAISSContextRetriever:
    """
    Embeds and indexes all relevant context (diffs, metrics, clusters)
    Retrieves only relevant context per file for LLM
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with embedding model
        
        Args:
            model_name: HuggingFace model for embeddings
                       all-MiniLM-L6-v2: Fast, 384 dims, good quality
        """
        print("="*60)
        print("FAISS RAG Layer Initialization")
        print("="*60)
        
        print(f"\nLoading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Embedding dimension: {self.embedding_dim}")
        
        # Storage
        self.index = None
        self.documents = []  # Stores actual text
        self.metadata = []   # Stores (repo, file, commit, type)
        
        print("="*60 + "\n")
    
    def build_index(self):
        """
        Build FAISS index from all available context
        
        What gets embedded:
        1. Diff summaries (what changed)
        2. Metric deltas (numerical changes)
        3. Cluster descriptions
        4. Commit messages
        """
        data_dir = DATA_DIR
        
        print("Building FAISS index from data...")
        print("-"*60)
        
        # Load source data
        with open(DATA_DIR / "file_analysis.json") as f:
            file_analysis = json.load(f)
        
        with open(data_dir / "cluster_summary.json") as f:
            cluster_data = json.load(f)
            # Handle list format
            if isinstance(cluster_data, list):
                cluster_summary = {
                    str(item.get('cluster_id', item.get('id', i))): item 
                    for i, item in enumerate(cluster_data)
                }
            else:
                cluster_summary = cluster_data
        
        # Load commits and diffs
        import csv
        commits = {}
        with open(data_dir / "commits.csv") as f:
            for row in csv.DictReader(f):
                commits[row['commit_hash']] = row
        
        diffs = {}
        with open(data_dir / "diffs.csv") as f:
            for row in csv.DictReader(f):
                key = (row['repo_name'], row['commit_hash'], row['file_path'])
                diffs[key] = row['diff_text']
        
        print(f"Loaded {len(file_analysis)} files, {len(commits)} commits, {len(diffs)} diffs")
        print("-"*60)
        
        # 1. Embed cluster descriptions (global context)
        print("\n[1/4] Embedding cluster descriptions...")
        for cluster_id, cluster_info in cluster_summary.items():
            doc = self._build_cluster_document(cluster_id, cluster_info)
            self.documents.append(doc)
            self.metadata.append({
                'type': 'cluster',
                'cluster_id': cluster_id,
                'repo': 'global',
                'file': None,
                'commit': None
            })
        print(f"  ✓ Added {len(cluster_summary)} cluster documents")
        
        # 2. Embed file evolution summaries
        print("\n[2/4] Embedding file evolution summaries...")
        for file_data in file_analysis:
            doc = self._build_file_evolution_document(file_data)
            self.documents.append(doc)
            self.metadata.append({
                'type': 'file_evolution',
                'cluster_id': file_data['cluster_id'],
                'repo': file_data['repo_name'],
                'file': file_data['file_path'],
                'commit': None
            })
        print(f"  ✓ Added {len(file_analysis)} file evolution documents")
        
        # 3. Embed diff contexts (code changes)
        print("\n[3/4] Embedding diff contexts...")
        diff_count = 0
        for (repo, commit_hash, filepath), diff_text in diffs.items():
            # Only embed meaningful diffs (skip tiny changes)
            if len(diff_text) < 50:
                continue
            
            commit_msg = commits.get(commit_hash, {}).get('message', 'No message')
            doc = self._build_diff_document(
                repo, filepath, commit_hash, commit_msg, diff_text
            )
            self.documents.append(doc)
            self.metadata.append({
                'type': 'diff',
                'cluster_id': None,
                'repo': repo,
                'file': filepath,
                'commit': commit_hash
            })
            diff_count += 1
        print(f"  ✓ Added {diff_count} diff documents")
        
        # 4. Embed metric change summaries
        print("\n[4/4] Embedding metric changes...")
        for file_data in file_analysis:
            doc = self._build_metrics_document(file_data)
            self.documents.append(doc)
            self.metadata.append({
                'type': 'metrics',
                'cluster_id': file_data['cluster_id'],
                'repo': file_data['repo_name'],
                'file': file_data['file_path'],
                'commit': None
            })
        print(f"  ✓ Added {len(file_analysis)} metric documents")
        
        # Create embeddings
        print(f"\n{'='*60}")
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
        
        print(f"✓ FAISS index built with {self.index.ntotal} vectors")
        print("="*60 + "\n")
    
    def _build_cluster_document(self, cluster_id: str, info: Dict) -> str:
        """Build searchable document for cluster"""
        return f"""
Cluster {cluster_id}: {info.get('description', 'N/A')}
Size: {info.get('size', 0)} files
Average churn rate: {info.get('avg_churn_rate', 0):.1%}
Average complexity: {info.get('avg_complexity', 0):.1f}
Average anomaly score: {info.get('avg_anomaly_score', 0):.3f}
Characteristics: {info.get('characteristics', 'Standard evolutionary pattern')}
        """.strip()
    
    def _build_file_evolution_document(self, file_data: Dict) -> str:
        """Build searchable document for file evolution"""
        trends = file_data.get('historical_trends', {})
        
        return f"""
File: {file_data['file_path']}
Repository: {file_data['repo_name']}
Cluster: {file_data['cluster_id']}
Total commits: {file_data['n_commits']}
Churn: {file_data['churn_label']}
Complexity trend: {trends.get('cc', {}).get('label', 'unknown')}
Maintainability trend: {trends.get('mi', {}).get('label', 'unknown')}
Anomaly score: {file_data['anomaly_score']:.3f}
        """.strip()
    
    def _build_diff_document(self, repo: str, filepath: str, 
                            commit: str, msg: str, diff: str) -> str:
        """Build searchable document for diff"""
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
    
    def _build_metrics_document(self, file_data: Dict) -> str:
        """Build searchable document for metrics"""
        raw = file_data.get('raw_features', {})
        
        return f"""
Metrics for {file_data['file_path']}
Churn rate: {raw.get('churn_rate', 0):.1%}
Mean complexity: {raw.get('mean_cc_after', 0):.2f}
Complexity volatility: {raw.get('std_cc_after', 0):.2f}
Mean maintainability: {raw.get('mean_mi_after', 0):.2f}
Mean LOC delta: {raw.get('mean_abs_loc_delta', 0):.1f}
        """.strip()
    
    def retrieve_context(self, file_data: Dict, k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a specific file
        
        Args:
            file_data: File from file_analysis.json
            k: Number of relevant documents to retrieve
        
        Returns:
            List of relevant documents with metadata
        """
        # Build query from file characteristics
        query = self._build_file_query(file_data)
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Search FAISS
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        # Gather results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadata[idx],
                'similarity': float(1 / (1 + dist))  # Convert L2 to similarity
            })
        
        return results
    
    def _build_file_query(self, file_data: Dict) -> str:
        """Build search query from file data"""
        return f"""
Find context related to:
File: {file_data['file_path']}
Repository: {file_data['repo_name']}
Cluster: {file_data['cluster_id']}
Behavior: {file_data['churn_label']}, anomaly score {file_data['anomaly_score']:.3f}
        """.strip()
    
    def save_index(self, output_dir: str = "data"):
        """Save FAISS index and metadata"""
        output_dir = DATA_DIR
        output_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_dir / "faiss.index"))
        
        # Save documents and metadata
        with open(output_dir / "faiss_documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✓ FAISS index saved to {output_dir}/")
    
    def load_index(self, data_dir: str = "data"):
        """Load pre-built FAISS index"""
        data_dir = DATA_DIR
        
        # Load FAISS index
        self.index = faiss.read_index(str(data_dir / "faiss.index"))
        
        # Load documents and metadata
        with open(data_dir / "faiss_documents.pkl", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']
        
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")


def build_faiss_index():
    """Standalone script to build FAISS index"""
    print("\n" + "="*60)
    print("BUILDING FAISS INDEX")
    print("="*60 + "\n")
    
    retriever = FAISSContextRetriever()
    retriever.build_index()
    retriever.save_index()
    
    print("\n" + "="*60)
    print("✓ FAISS index built successfully!")
    print("="*60)
    print("\nTest retrieval:")
    print("-"*60)
    
    # Test with first file
    with open(DATA_DIR / "file_analysis.json") as f:

        files = json.load(f)
    
    if files:
        test_file = files[0]
        print(f"\nQuerying: {test_file['file_path']}")
        results = retriever.retrieve_context(test_file, k=5)
        
        print(f"\nTop 5 relevant contexts:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Type: {result['metadata']['type']}")
            print(f"   Similarity: {result['similarity']:.3f}")
            print(f"   Preview: {result['document'][:150]}...")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    build_faiss_index()