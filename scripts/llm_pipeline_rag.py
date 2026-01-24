"""
LLM Pipeline for Code Forensics - Neuro-Symbolic Architecture
Two-stage explanatory system with grounded interpretation

Architectural Constraints:
- LLM acts ONLY as an explainer and synthesizer
- LLM must NEVER recommend refactors, predict bugs/risks, assign severity, or introduce new labels
- Stage 1: File-level explanation (N API calls) using windowed diffs
- Stage 2: Repository-level synthesis (1 API call) using only Stage 1 outputs

Epistemic Boundaries:
- ML = detection of patterns (WHAT)
- Diffs = evidence anchors (WHERE)
- LLM = interpretation only (HOW / WHY)
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import os

import numpy as np
from scipy.stats import percentileofscore
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import FAISS retriever
try:
    from faiss_rag import FAISSContextRetriever
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS module not found - windowed diff retrieval unavailable")

# Configurable top-K for file selection
DEFAULT_TOP_K = 10


def select_top_k_files(
    file_analysis: List[Dict],
    criterion: str = "anomaly_score",
    k: int = DEFAULT_TOP_K
) -> List[Dict]:
    """
    Select top-K files based on a criterion (attention mechanism).
    
    Only anomalous/atypical files are explained to focus LLM resources.
    
    Args:
        file_analysis: List of file analysis dicts from file_analysis.json
        criterion: Field to sort by (default: anomaly_score)
        k: Number of files to select
    
    Returns:
        Top-K files sorted by criterion (descending)
    """
    return sorted(
        file_analysis,
        key=lambda x: x.get(criterion, 0),
        reverse=True
    )[:k]


def strip_ungrounded_commits(explanation: str, valid_commits: List[str]) -> str:
    """
    Remove any commit hash references not in valid_commits list.
    
    This ensures grounding: if an explanation references a commit hash,
    it must exist in the retrieved diff list.
    
    Args:
        explanation: LLM-generated explanation text
        valid_commits: List of commit hashes that were actually retrieved
    
    Returns:
        Explanation with ungrounded commit references removed
    """
    if not valid_commits:
        return explanation
    
    # Create set of valid short and full hashes
    valid_set: Set[str] = set()
    for commit in valid_commits:
        valid_set.add(commit)
        valid_set.add(commit[:7])
        valid_set.add(commit[:8])
    
    # Find all commit-like patterns (7-40 hex chars)
    commit_pattern = r'\b([a-f0-9]{7,40})\b'
    
    def replace_if_ungrounded(match):
        commit_ref = match.group(1)
        # Check if this reference is grounded
        if commit_ref in valid_set:
            return commit_ref
        # Check if any valid commit starts with this reference
        for valid in valid_commits:
            if valid.startswith(commit_ref) or commit_ref.startswith(valid[:7]):
                return commit_ref
        # Ungrounded - remove it
        logger.warning(f"Stripped ungrounded commit reference: {commit_ref}")
        return "[commit]"
    
    return re.sub(commit_pattern, replace_if_ungrounded, explanation)


def extract_referenced_commits(explanation: str, valid_commits: List[str]) -> List[str]:
    """
    Extract commit hashes that are actually referenced in the explanation.
    
    Args:
        explanation: LLM-generated explanation text
        valid_commits: List of valid commit hashes
    
    Returns:
        List of commit hashes referenced in the explanation
    """
    if not valid_commits:
        return []
    
    referenced = []
    commit_pattern = r'\b([a-f0-9]{7,40})\b'
    
    for match in re.finditer(commit_pattern, explanation):
        commit_ref = match.group(1)
        for valid in valid_commits:
            if valid.startswith(commit_ref) or commit_ref.startswith(valid[:7]):
                if valid not in referenced:
                    referenced.append(valid)
                break
    
    return referenced


class CodeForensicsLLM:
    """
    Neuro-symbolic LLM pipeline for code evolution interpretation.
    
    The LLM is constrained to interpretation only:
    - Characterize evolutionary trajectories
    - Highlight metric/code discrepancies
    - Never speculate beyond provided evidence
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_faiss: bool = True,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.3
    ):
        """
        Initialize with Groq API and FAISS retriever.
        
        Args:
            api_key: Groq API key (or from GROQ_API_KEY env var)
            use_faiss: Whether to use FAISS for windowed diff retrieval
            top_k: Number of top anomalous files to explain
            temperature: LLM temperature (default 0.3 for determinism)
        """
        print("=" * 60)
        print("Code Forensics LLM Pipeline")
        print("Neuro-Symbolic Architecture")
        print("=" * 60)
        
        # Configuration
        self.top_k = top_k
        self.temperature = temperature
        
        # Load .env and get API key
        load_dotenv(BASE_DIR / ".env")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found!\n"
                "Get your free API key at: https://console.groq.com\n"
                "Then set it: export GROQ_API_KEY='your-key-here'"
            )
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        
        # Initialize FAISS retriever
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.retriever: Optional[FAISSContextRetriever] = None
        
        if self.use_faiss:
            print("\n✓ Initializing FAISS RAG layer (diff-only)...")
            self.retriever = FAISSContextRetriever()
            try:
                self.retriever.load_index()
                print("✓ FAISS index loaded")
            except FileNotFoundError:
                print("⚠️  FAISS index not found - run build_faiss_index() first")
                self.use_faiss = False
        
        print(f"\n✓ Groq API initialized")
        print(f"✓ Model: {self.model}")
        print(f"✓ Temperature: {self.temperature}")
        print(f"✓ Top-K files: {self.top_k}")
        print(f"✓ FAISS Mode: {'ENABLED (diff-only)' if self.use_faiss else 'DISABLED'}")
        print("=" * 60 + "\n")

    def _contextualize_anomaly_score(self, file_score: float, all_scores: List[float]) -> str:
        """Convert raw anomaly score to percentile rank."""
        if not all_scores:
            return "N/A"
        percentile = int(percentileofscore(all_scores, file_score))
        top_percent = 100 - percentile
        return f"{percentile}th percentile (top {top_percent}% most atypical)"

    def _compute_key_signals(self, file_data: Dict, cluster_stats: Dict) -> Dict[str, str]:
        """Compute prioritized metrics for attention-directing."""
        cluster_id = str(file_data['cluster_id'])
        cluster_median_churn = cluster_stats.get(cluster_id, {}).get('median_churn', 0)

        # Churn comparison
        file_churn = file_data['raw_features']['churn_rate']
        churn_ratio = file_churn / cluster_median_churn if cluster_median_churn > 0 else 1.0
        churn_vs_cluster = f"{churn_ratio:.1f}× cluster median"

        # Complexity trend
        cc_slope = file_data['historical_trends']['cc']['value']
        cc_label = file_data['historical_trends']['cc']['label']
        complexity_trend = f"{cc_label} (slope: {cc_slope:.4f})"

        return {
            'churn_vs_cluster': churn_vs_cluster,
            'complexity_trend': complexity_trend,
            'maintainability_trend': file_data['historical_trends']['mi']['label'],
            'import_volatility': file_data['imports_volatility_label']
        }

    def _validate_file_explanation(self, explanation: str, provided_commits: List[str]) -> Dict[str, Any]:
        """Validate file-level output quality."""
        violations = []

        # 1. Word count
        word_count = len(explanation.split())
        if word_count > 250:
            violations.append(f"Exceeded word limit: {word_count} words (max 250)")

        # 2. Required sections
        required_sections = [
            'Why This File Stands Out',
            'Evidence from Metrics',
            'Evidence from Code Changes',
            'What Makes This Unusual'
        ]
        for section in required_sections:
            if section not in explanation:
                violations.append(f"Missing required section: '{section}'")

        # 3. Banned patterns (case-insensitive)
        banned_patterns = [
            'has undergone', 'has been modified', 'has experienced',
            'suggests that', 'appears to', 'seems to indicate',
            'significant changes', 'substantial modifications',
            'development focused on', 'efforts were made'
        ]
        explanation_lower = explanation.lower()
        for pattern in banned_patterns:
            if pattern in explanation_lower:
                violations.append(f"Contains banned pattern: '{pattern}'")

        # 4. Commit grounding (check referenced commits exist in provided diffs)
        commit_refs = re.findall(r'\b[0-9a-f]{7,40}\b', explanation)
        for commit_ref in commit_refs:
            if not any(commit_ref in commit for commit in provided_commits):
                violations.append(f"References unknown commit: {commit_ref[:8]}")

        return {
            'valid': len(violations) == 0,
            'violations': violations,
            'word_count': word_count
        }

    def _extract_attention_signals(self, file_explanations: List[Dict]) -> str:
        """Extract only the 'Why This File Stands Out' section from each file."""
        signals = []
        for file_exp in file_explanations:
            file_path = file_exp.get('file', 'unknown')
            explanation = file_exp.get('explanation', '')

            # Extract first section (assumes markdown heading structure)
            lines = explanation.split('\n')
            why_section = []
            in_section = False

            for line in lines:
                if '### Why This File Stands Out' in line or '## Why This File Stands Out' in line:
                    in_section = True
                    continue
                if line.startswith('###') or line.startswith('##'):
                    if in_section:
                        break
                elif in_section and line.strip():
                    why_section.append(line.strip())

            signal = ' '.join(why_section) if why_section else "(No signal extracted)"
            signals.append(f"- **{file_path}**: {signal}")

        return '\n'.join(signals)
    
    def load_data(self) -> None:
        """Load required data files."""
        print("Loading data files from:", DATA_DIR.absolute())
        print("-" * 60)
        
        required = [
            DATA_DIR / "file_analysis.json",
            DATA_DIR / "cluster_summary.json",
        ]
        if self.use_faiss:
            required.append(DATA_DIR / "faiss.index")
        
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required files:\n" +
                "\n".join(f"  - {p}" for p in missing)
            )
        
        # Load ML outputs
        with open(DATA_DIR / "file_analysis.json") as f:
            self.file_analysis = json.load(f)
        print(f"✓ file_analysis.json ({len(self.file_analysis)} files)")

        # Precompute anomaly score list for percentile context
        self.all_anomaly_scores = [f.get('anomaly_score', 0) for f in self.file_analysis]

        # Precompute cluster churn medians for key signals
        churn_by_cluster: Dict[str, List[float]] = {}
        for f in self.file_analysis:
            cluster_id = str(f.get('cluster_id'))
            churn_by_cluster.setdefault(cluster_id, []).append(
                f.get('raw_features', {}).get('churn_rate', 0)
            )

        self.cluster_stats = {}
        for cluster_id, churn_values in churn_by_cluster.items():
            median_churn = float(np.median(churn_values)) if churn_values else 0.0
            self.cluster_stats[cluster_id] = {
                'median_churn': median_churn
            }
        
        # Load cluster summary
        with open(DATA_DIR / "cluster_summary.json") as f:
            cluster_data = json.load(f)
        
        # Convert list to dict if needed
        if isinstance(cluster_data, list):
            self.cluster_summary = {}
            for item in cluster_data:
                cluster_id = item.get('cluster_id', item.get('id', 'unknown'))
                self.cluster_summary[str(cluster_id)] = item
            print(f"✓ cluster_summary.json ({len(self.cluster_summary)} clusters)")
        else:
            self.cluster_summary = cluster_data
            print(f"✓ cluster_summary.json ({len(self.cluster_summary)} clusters)")
        
        print("-" * 60 + "\n")
    
    def build_file_prompt(
        self,
        file_data: Dict[str, Any],
        diffs: List[Dict],
        anomaly_context: str,
        key_signals: Dict[str, str]
    ) -> str:
        """
        Build attention-worthy prompt for file explanation.
        """
        # Format diffs for prompt
        diff_text = ""
        if diffs:
            # Hard limit: show max 2 diffs only
            for i, diff in enumerate(diffs[:2], 1):
                meta = diff['metadata']
                diff_text += f"\n### Diff {i} (Commit: {meta['commit'][:8]})\n"
                diff_text += f"```\n{diff['document']}\n```\n"
        else:
            diff_text = "No diffs available for this file."

        # Get cluster label
        cluster_id = str(file_data['cluster_id'])
        cluster_info = self.cluster_summary.get(cluster_id, {})
        cluster_label = cluster_info.get('derived_cluster_label', f'Cluster {cluster_id}')

        prompt = f"""Given the metrics and code diffs for a single file, explain why this file stands out compared to other files in the same repository.

Answer only this question:
Why should a developer pay attention to this file?

You are provided:
- ML-detected metric trends (churn, complexity, volatility, anomaly score)
- Cluster assignment (relative behavior group)
- Retrieved code diffs (commit-level evidence)

Use only this information.

OUTPUT FORMAT (STRICT — FOLLOW EXACTLY):

### Why This File Stands Out
(1–2 sentences, plain English, no numbers)

### Evidence from Metrics
- Max 3 bullet points
- Each bullet must compare this file to:
  • other files in the repo, or
  • files in its cluster

### Evidence from Code Changes
- Max 2 bullet points
- Each bullet must reference a specific commit hash
- Explain what changed and how it explains the metric behavior

### What Makes This Unusual
(1 sentence describing tension, deviation, or contradiction)

HARD CONSTRAINTS:
- Max 220 words total (will be validated)
- Do NOT use: “has undergone”, “suggests that”, “appears to”, “development focused on”
- Reference ONLY commits shown in diffs above
- If no commit materially explains a metric trend, say so explicitly
- Reference at most 2 commits, and only if they are provided
- Do NOT use passive voice, hedging, or process narration
- Do NOT give recommendations, predict risks, or assign severity
- Do NOT restate cluster definitions
- Each metric bullet MUST compare to cluster or repo average

## FILE INFORMATION

**File:** {file_data['file_path']}
**Repository:** {file_data['repo_name']}
**Cluster:** {cluster_label}

## KEY SIGNALS (Use these first)

**Anomaly Level:** {anomaly_context}
**Churn vs Cluster:** {key_signals['churn_vs_cluster']}
**Complexity Trend:** {key_signals['complexity_trend']}

## SUPPORTING CONTEXT (Use if relevant)

- Total commits analyzed: {file_data['n_commits']}
- Maintainability trend: {key_signals['maintainability_trend']}
- Import volatility: {key_signals['import_volatility']}
- Cluster assignment: {cluster_label}

## CODE DIFFS (Evidence Anchors)
{diff_text}
"""
        return prompt
    
    def generate_file_explanation(
        self,
        file_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate grounded explanation for a single file.
        
        Returns structured output with:
        - repo, file, cluster, anomaly_score
        - referenced_commits (grounded)
        - explanation (with ungrounded refs stripped)
        """
        # Retrieve windowed diffs via FAISS
        diffs = []
        valid_commits = []
        
        if self.use_faiss and self.retriever:
            diffs = self.retriever.retrieve_windowed_diffs(file_data, k=3)
            diffs = diffs[:2]
            valid_commits = [d['metadata']['commit'] for d in diffs]
            logger.info(f"Retrieved {len(diffs)} diffs for {file_data['file_path']}")
            logger.info(f"Commits: {[c[:8] for c in valid_commits]}")
        
        # Preprocessing: anomaly context + key signals
        anomaly_context = self._contextualize_anomaly_score(
            file_data.get('anomaly_score', 0),
            getattr(self, 'all_anomaly_scores', [])
        )
        key_signals = self._compute_key_signals(
            file_data,
            getattr(self, 'cluster_stats', {})
        )

        # Build prompt
        prompt = self.build_file_prompt(file_data, diffs, anomaly_context, key_signals)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a code evolution analyst writing for busy developers. "
                            "Your role is to flag files that deserve attention based on historical behavior. "
                            "Rules for language: "
                            "- Avoid passive or hedging constructions (e.g., 'has been', 'appears', 'suggests') "
                            "- Avoid vague intensifiers ('significant', 'substantial', 'notable') "
                            "- Avoid process narration ('efforts were made', 'development focused on') "
                            "Preferred style: "
                            "- Active voice with specific comparisons ('2x higher than cluster average') "
                            "- Concrete contrasts ('only file in cluster with rising complexity') "
                            "- Evidence-backed statements citing commit hashes "
                            "Never diagnose bugs, predict risk, or give recommendations."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=1200,
                top_p=0.9,
            )
            
            explanation = response.choices[0].message.content.strip()

            # Validate output
            provided_commits = [d['metadata']['commit'] for d in diffs]
            validation = self._validate_file_explanation(explanation, provided_commits)

            if not validation['valid']:
                logger.warning(f"Output validation failed for {file_data['file_path']}:")
                for violation in validation['violations']:
                    logger.warning(f"  - {violation}")

                if validation['word_count'] > 250:
                    logger.info("Attempting regeneration with stricter word limit...")
                    stricter_prompt = (
                        prompt
                        + "\n\nCRITICAL: Stay under 200 words or output will be rejected."
                    )
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a code evolution analyst writing for busy developers. "
                                    "Your role is to flag files that deserve attention based on historical behavior. "
                                    "Rules for language: "
                                    "- Avoid passive or hedging constructions (e.g., 'has been', 'appears', 'suggests') "
                                    "- Avoid vague intensifiers ('significant', 'substantial', 'notable') "
                                    "- Avoid process narration ('efforts were made', 'development focused on') "
                                    "Preferred style: "
                                    "- Active voice with specific comparisons ('2x higher than cluster average') "
                                    "- Concrete contrasts ('only file in cluster with rising complexity') "
                                    "- Evidence-backed statements citing commit hashes "
                                    "Never diagnose bugs, predict risk, or give recommendations."
                                )
                            },
                            {
                                "role": "user",
                                "content": stricter_prompt
                            }
                        ],
                        temperature=self.temperature,
                        max_tokens=1200,
                        top_p=0.9,
                    )
                    explanation = response.choices[0].message.content.strip()
                    validation = self._validate_file_explanation(explanation, provided_commits)
            
            # Grounding check: strip ungrounded commit references
            explanation = strip_ungrounded_commits(explanation, valid_commits)
            
            # Extract actually referenced commits
            referenced = extract_referenced_commits(explanation, valid_commits)
            
            # Log the mapping
            logger.info(f"File: {file_data['file_path']}")
            logger.info(f"Commits retrieved: {valid_commits}")
            logger.info(f"Commits referenced: {referenced}")
            
            return {
                "repo": file_data['repo_name'],
                "file": file_data['file_path'],
                "cluster": str(file_data['cluster_id']),
                "anomaly_score": file_data['anomaly_score'],
                "referenced_commits": referenced,
                "explanation": explanation,
                "validation": validation
            }
            
        except Exception as e:
            logger.error(f"API Error for {file_data['file_path']}: {e}")
            return {
                "repo": file_data['repo_name'],
                "file": file_data['file_path'],
                "cluster": str(file_data['cluster_id']),
                "anomaly_score": file_data['anomaly_score'],
                "referenced_commits": [],
                "explanation": f"Error generating explanation: {e}"
            }
    
    def build_repo_synthesis_prompt(
        self,
        file_explanations: List[Dict[str, Any]]
    ) -> str:
        """
        Build repository-level synthesis prompt.
        
        Inputs (ONLY these, no FAISS, no raw diffs):
        - cluster_summary.json
        - file_explanations.json (Stage 1 outputs)
        
        The prompt explicitly forbids advice language.
        """
        # Cluster overview
        cluster_text = "## CLUSTER CHARACTERISTICS\n\n"
        for cluster_id, cluster_info in self.cluster_summary.items():
            cluster_text += f"""**Cluster {cluster_id}:** {cluster_info.get('derived_cluster_label', 'N/A')}
- Size: {cluster_info.get('size', 0)} files
- Avg churn rate: {cluster_info.get('avg_churn_rate', 0):.1%}
- Avg complexity: {cluster_info.get('avg_cc_after', 0):.2f}
- Churn distribution: {cluster_info.get('churn_label_distribution', {})}
- Complexity trend distribution: {cluster_info.get('cc_trend_label_distribution', {})}

"""
        
        # Compressed file-level signals
        signals_text = "## FILE-LEVEL ATTENTION SIGNALS\n\n"
        signals_text += self._extract_attention_signals(file_explanations)
        
        prompt = f"""Using the cluster summaries and file-level outputs, produce a repository-level synthesis that highlights where evolution concentrates and where it behaves unexpectedly.

Do NOT restate file summaries verbatim.

OUTPUT FORMAT (STRICT — FOLLOW EXACTLY):

### Dominant Evolutionary Modes
- 2–3 labeled themes (e.g., “Refinement Zones”, “Complexity Accretion”)
- Each label must be justified using metric relationships

### Where Change Concentrates
- Identify files or clusters that absorb a disproportionate share of churn or complexity change

### Files That Defy Their Peers
- Short list (2–4 files)
- One-line explanation of how each deviates from its cluster

### What This Says About the Repository
- 2-3 sentences max
- No fluff, no repetition

HARD CONSTRAINTS:
- No recommendations, action items, or risk language
- No repeating cluster definitions
- Prefer contrasts over descriptions
- If no cross-file insight exists, say so explicitly

{cluster_text}

{signals_text}
"""
        return prompt
    
    def generate_repo_synthesis(
        self,
        file_explanations: List[Dict[str, Any]]
    ) -> str:
        """
        Generate repository-level synthesis using ONLY Stage 1 outputs.
        
        No FAISS retrieval.
        No raw diffs.
        No re-analysis.
        """
        prompt = self.build_repo_synthesis_prompt(file_explanations)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are synthesizing cross-file evolution patterns for a repository. "
                            "Your role is to surface attention-directing signals, not recommendations or risk assessments. "
                            "Theme quality rules: "
                            "GOOD themes: "
                            "- Describe a relationship between metrics (e.g., 'high churn with declining complexity') "
                            "- Explain how multiple files behave similarly "
                            "BAD themes: "
                            "- Restate a single metric ('High Complexity Files') "
                            "- Rename clusters ('Cluster 0 Behavior') "
                            "- Use generic labels ('Active Development', 'Maintenance Mode') "
                            "Be concise and comparative."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=2000,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"API Error for repo synthesis: {e}")
            return f"Error generating repository synthesis: {e}"
    
    def run_pipeline(self, output_dir: str = "output", limit: Optional[int] = None) -> None:
        """
        Run complete two-stage pipeline.\n\n
        
        Stage 0: Select top-K anomalous files\n
        Stage 1: Generate file-level explanations (N API calls)\n
        Stage 2: Generate repository-level synthesis (1 API call)
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Stage 0: File selection (attention mechanism)
        print("=" * 60)
        print("STAGE 0: File Selection")
        print("=" * 60)
        
        k = limit if limit else self.top_k
        files = select_top_k_files(
            self.file_analysis,
            criterion="anomaly_score",
            k=k
        )
        
        print(f"Selected top {len(files)} files by anomaly_score:")
        for i, f in enumerate(files[:5], 1):
            print(f"  {i}. {f['file_path']} (score: {f['anomaly_score']:.4f})")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        print("=" * 60 + "\n")
        
        # Stage 1: File-level explanations
        print("=" * 60)
        print("STAGE 1: File-level Explanations")
        print(f"Processing {len(files)} files")
        print(f"Temperature: {self.temperature}")
        print("=" * 60 + "\n")
        
        file_explanations = []
        
        for i, file_data in enumerate(files, 1):
            file_path = file_data['file_path']
            print(f"[{i}/{len(files)}] {file_path}")
            
            result = self.generate_file_explanation(file_data)
            file_explanations.append(result)
            
            print(f"           ✓ Generated ({len(result['explanation'])} chars)")
            print(f"           Commits: {result['referenced_commits']}\n")
            
            # Save checkpoint every 5 files
            if i % 5 == 0:
                with open(output_path / "checkpoint.json", 'w') as f:
                    json.dump(file_explanations, f, indent=2)
                logger.info(f"Checkpoint saved at {i} files")
        
        # Save file explanations
        with open(output_path / "file_explanations.json", 'w') as f:
            json.dump(file_explanations, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print(f"✓ Saved {len(file_explanations)} file explanations")
        print(f"{'=' * 60}\n")
        
        # Stage 2: Repository-level synthesis
        print("=" * 60)
        print("STAGE 2: Repository Synthesis")
        print("Input: cluster_summary.json + file_explanations.json")
        print("NO FAISS | NO raw diffs | NO re-analysis")
        print("=" * 60 + "\n")
        
        print("Generating repository-level synthesis...")
        repo_synthesis = self.generate_repo_synthesis(file_explanations)
        
        # Save report
        with open(output_path / "repository_report.md", 'w') as f:
            f.write("# Repository Evolution Synthesis\n\n")
            f.write("*This report describes observed evolutionary patterns without recommendations or risk assessments.*\n\n")
            f.write("---\n\n")
            f.write(repo_synthesis)
        
        with open(output_path / "repository_report.json", 'w') as f:
            json.dump({
                'synthesis': repo_synthesis,
                'metadata': {
                    'files_analyzed': len(file_explanations),
                    'clusters': len(self.cluster_summary),
                    'model': self.model,
                    'temperature': self.temperature,
                    'top_k': self.top_k,
                    'architecture': 'neuro-symbolic'
                }
            }, f, indent=2)
        
        print(f"\n✓ Repository synthesis saved\n")
        
        print("=" * 60)
        print("Pipeline Complete")
        print("=" * 60)
        print(f"\nOutputs saved to: {output_path.absolute()}")
        print("\nFiles created:")
        print(f"  - file_explanations.json ({len(file_explanations)} explanations)")
        print(f"  - repository_report.md (synthesis)")
        print(f"  - repository_report.json (structured)")
        print("\nArchitectural compliance:")
        print("  - LLM acted as interpreter only (no advice)")
        print("  - Stage 2 used only Stage 1 outputs")
        print("  - All commit references are grounded")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    import sys
    
    print("\n" + "=" * 60)
    print("CODE FORENSICS - NEURO-SYMBOLIC PIPELINE")
    print("LLM as interpreter, not advisor")
    print("=" * 60 + "\n")
    
    # Parse arguments
    test_mode = "--test" in sys.argv
    
    # Get top_k from args if provided
    top_k = DEFAULT_TOP_K
    for arg in sys.argv:
        if arg.startswith("--top-k="):
            try:
                top_k = int(arg.split("=")[1])
            except ValueError:
                pass
    
    if test_mode:
        print(f"TEST MODE: Processing only {min(5, top_k)} files\n")
        top_k = min(5, top_k)
    
    try:
        # Initialize pipeline
        pipeline = CodeForensicsLLM(
            use_faiss=True,
            top_k=top_k,
            temperature=0.3
        )
        
        # Load data
        pipeline.load_data()
        
        # Run pipeline
        pipeline.run_pipeline()
        
    except ValueError as e:
        print(f"\n{'=' * 60}")
        print("CONFIGURATION ERROR")
        print("=" * 60)
        print(f"\n{e}\n")
        print("=" * 60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n{'=' * 60}")
        print("MISSING FILES")
        print("=" * 60)
        print("\nRequired files not found. Please ensure:")
        print("  - data/file_analysis.json exists")
        print("  - data/cluster_summary.json exists")
        print("  - data/faiss.index exists (run faiss_rag.py first)")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n{'=' * 60}")
        print("ERROR")
        print("=" * 60)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
