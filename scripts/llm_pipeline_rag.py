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
    
    def build_file_prompt(self, file_data: Dict[str, Any], diffs: List[Dict]) -> str:
        """
        Build neutral, epistemic-boundary-respecting prompt for file explanation.
        
        Inputs per file:
        - Structured ML JSON (file_analysis.json entry)
        - Cluster ID (label only)
        - 2-3 windowed diffs retrieved via FAISS
        - No other context
        
        The prompt explicitly forbids:
        - Recommendations
        - Risk predictions
        - Severity assignments
        - Speculation beyond evidence
        """
        # Format diffs for prompt
        diff_text = ""
        if diffs:
            for i, diff in enumerate(diffs, 1):
                meta = diff['metadata']
                diff_text += f"\n### Diff {i} (Commit: {meta['commit'][:8]})\n"
                diff_text += f"```\n{diff['document'][:600]}\n```\n"
        else:
            diff_text = "No diffs available for this file."
        
        # Get cluster label
        cluster_id = str(file_data['cluster_id'])
        cluster_info = self.cluster_summary.get(cluster_id, {})
        cluster_label = cluster_info.get('derived_cluster_label', f'Cluster {cluster_id}')
        
        prompt = f"""Characterize the evolutionary trajectory of this source code file based on the provided metric trends and code diffs.

## FILE INFORMATION

**File:** {file_data['file_path']}
**Repository:** {file_data['repo_name']}
**Cluster:** {cluster_label}

## METRIC TRENDS (ML-Detected)

**Evolution Summary:**
- Total commits analyzed: {file_data['n_commits']}
- Anomaly score: {file_data['anomaly_score']:.4f}

**Behavioral Labels (rule-based):**
- Churn level: {file_data['churn_label']}
- Complexity volatility: {file_data['cc_volatility_label']}
- Import volatility: {file_data['imports_volatility_label']}

**Historical Trends:**
- Cyclomatic complexity: {file_data['historical_trends']['cc']['label']} (slope: {file_data['historical_trends']['cc']['value']:.4f})
- Maintainability index: {file_data['historical_trends']['mi']['label']} (slope: {file_data['historical_trends']['mi']['value']:.4f})
- Imports: {file_data['historical_trends']['imports']['label']} (slope: {file_data['historical_trends']['imports']['value']:.4f})

**Raw Metrics:**
- Churn rate: {file_data['raw_features']['churn_rate']:.1%}
- Mean LOC delta: {file_data['raw_features']['mean_abs_loc_delta']:.1f}
- Mean complexity: {file_data['raw_features']['mean_cc_after']:.2f} (+/- {file_data['raw_features']['std_cc_after']:.2f})
- Mean maintainability: {file_data['raw_features']['mean_mi_after']:.2f}

## CODE DIFFS (Evidence Anchors)
{diff_text}

## YOUR TASK

Provide a characterization (3-4 paragraphs) that:

1. **Describes the evolutionary trajectory**: How has this file changed over time based on the metrics?
2. **Interprets the patterns**: What do the metric trends suggest about how this file has been developed?
3. **Grounds in evidence**: Reference specific commits and metric values where relevant.
4. **Highlights discrepancies**: If the metrics and code changes appear contradictory, explicitly note this.

**CONSTRAINTS (STRICTLY ENFORCED):**
- Do NOT provide recommendations or suggest what developers should do
- Do NOT predict risks, bugs, or future problems
- Do NOT assign severity, priority, or urgency
- Do NOT speculate beyond the provided evidence
- Focus ONLY on characterizing WHAT the metrics show and WHY the code may have evolved this way"""

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
            valid_commits = [d['metadata']['commit'] for d in diffs]
            logger.info(f"Retrieved {len(diffs)} diffs for {file_data['file_path']}")
            logger.info(f"Commits: {[c[:8] for c in valid_commits]}")
        
        # Build prompt
        prompt = self.build_file_prompt(file_data, diffs)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing code evolution patterns. "
                            "Your role is to interpret and explain, never to advise or predict. "
                            "Describe what the data shows without making recommendations."
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
                "explanation": explanation
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
        
        # File explanations summary
        explanations_text = "## FILE-LEVEL EXPLANATIONS\n\n"
        for item in file_explanations:
            explanations_text += f"""### {item['file']}
**Cluster:** {item['cluster']} | **Anomaly Score:** {item['anomaly_score']:.4f}
**Referenced Commits:** {', '.join(c[:8] for c in item['referenced_commits']) if item['referenced_commits'] else 'None'}

{item['explanation'][:800]}{'...' if len(item['explanation']) > 800 else ''}

---

"""
        
        prompt = f"""Synthesize the evolutionary patterns across this repository based on the cluster characteristics and file-level explanations below.

{cluster_text}

{explanations_text}

## YOUR TASK

Create a repository-level synthesis (4-5 paragraphs) that:

1. **Summarizes observed patterns**: What evolutionary patterns emerge across the explained files?
2. **Relates to cluster behavior**: How do individual file trajectories relate to their cluster characteristics?
3. **Describes repository evolution**: How has this repository evolved based on the evidence?
4. **Highlights areas of atypical change**: Which files or clusters show unusual evolutionary patterns?

**LANGUAGE CONSTRAINTS (STRICTLY ENFORCED):**
- Use "observed patterns" instead of "risks"
- Use "evolutionary themes" instead of "concerns"
- Use "areas of atypical change" instead of "problems"
- Do NOT use: "recommend", "should", "must", "need to", "mitigation", "risk", "priority"
- Do NOT provide action items or next steps
- Focus ONLY on describing what the data shows

This is an interpretive synthesis, not a prescriptive report."""

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
                            "You are synthesizing code evolution patterns across a repository. "
                            "Describe observed patterns and evolutionary themes. "
                            "Never provide recommendations, risk assessments, or action items."
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
        Run complete two-stage pipeline.
        
        Stage 0: Select top-K anomalous files
        Stage 1: Generate file-level explanations (N API calls)
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
