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
from collections import defaultdict

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

    def _collect_all_anomaly_scores(self, file_data_list: List[Dict]) -> None:
        """Collect anomaly scores for percentile normalization."""
        self.all_anomaly_scores = [f.get('anomaly_score', 0) for f in file_data_list]

    def _prepare_interpretation_context(self, file_data: Dict, cluster_stats: Dict) -> Dict[str, Any]:
        """
        Deterministically contextualize ML outputs for explanation.
        No learning, no thresholds beyond simple comparisons.
        """
        cluster_id = str(file_data['cluster_id'])
        anomaly_score = file_data.get('anomaly_score', 0)

        percentile = int(percentileofscore(self.all_anomaly_scores, anomaly_score))
        top_percent = 100 - percentile

        if top_percent <= 5:
            anomaly_context = f"Top {top_percent}% most atypical (extreme outlier)"
        elif top_percent <= 20:
            anomaly_context = f"Top {top_percent}% most atypical"
        else:
            anomaly_context = f"{percentile}th percentile"

        cluster_median_churn = cluster_stats.get(cluster_id, {}).get('median_churn', 0.0)
        file_churn = file_data['raw_features']['churn_rate']
        churn_ratio = (
            file_churn / cluster_median_churn
            if cluster_median_churn > 0 else 1.0
        )

        # Add semantic interpretation to prevent LLM contradictions
        if churn_ratio < 0.8:
            churn_interpretation = "lower than cluster median (more stable)"
        elif churn_ratio < 1.2:
            churn_interpretation = "similar to cluster median"
        elif churn_ratio < 2.0:
            churn_interpretation = "moderately higher than cluster median"
        else:
            churn_interpretation = "significantly higher than cluster median"

        churn_comparison = f"{churn_ratio:.2f}× cluster median ({churn_interpretation})"

        cluster_cc_slope = cluster_stats.get(cluster_id, {}).get('median_cc_slope', 0.0)
        file_cc_slope = file_data['historical_trends']['cc']['value']

        if abs(cluster_cc_slope) > 1e-3:
            cc_ratio = round(abs(file_cc_slope / cluster_cc_slope), 2)
            direction = "faster" if abs(file_cc_slope) > abs(cluster_cc_slope) else "slower"
            complexity_context = f"{cc_ratio}x {direction} than cluster median"
        else:
            complexity_context = f"slope {file_cc_slope:.4f}"

        return {
            "anomaly_context": anomaly_context,
            "churn_comparison": churn_comparison,
            "complexity_comparison": complexity_context,
            "n_commits": file_data.get("n_commits", 0)
        }

    def _compute_key_signals(self, file_data: Dict, cluster_stats: Dict) -> Dict[str, str]:
        """Compute prioritized metrics for attention-directing."""
        cluster_id = str(file_data['cluster_id'])
        cluster_median_churn = cluster_stats.get(cluster_id, {}).get('median_churn', 0)

        # Churn comparison
        file_churn = file_data['raw_features']['churn_rate']
        churn_ratio = file_churn / cluster_median_churn if cluster_median_churn > 0 else 1.0
        churn_vs_cluster = f"{churn_ratio:.1f}x cluster median"

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

    def _build_component_summary(self, file_explanations: List[Dict[str, Any]]) -> str:
        components: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for f in file_explanations:
            path = f.get("file", "")
            parts = path.split("/")
            component = parts[1] if len(parts) > 1 else "root"
            components[component].append(f)

        lines = []
        for comp, files in components.items():
            avg_anomaly = sum(f.get("anomaly_score", 0) for f in files) / len(files)
            clusters = {f.get("cluster") for f in files}
            lines.append(
                f"{comp}/: {len(files)} files, avg anomaly {avg_anomaly:.2f}, clusters {sorted(clusters)}"
            )

        return "\n".join(lines)

    def compute_grounding_metrics(self, explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(explanations)
        counters = {
            "valid_commits": 0,
            "quantified": 0,
            "within_limit": 0,
            "violations": 0
        }

        for e in explanations:
            text = e.get("explanation", "")
            val = e.get("validation", {})

            if not val.get("violations"):
                counters["valid_commits"] += 1

            if "x" in text:
                counters["quantified"] += 1

            if val.get("word_count", 999) <= 250:
                counters["within_limit"] += 1

            if val.get("violations"):
                counters["violations"] += 1

        if total == 0:
            return {
                "commit_grounding_rate": 0,
                "quantified_comparison_rate": 0,
                "word_limit_compliance": 0,
                "violation_rate": 0,
                "total_files": 0
            }

        return {
            "commit_grounding_rate": counters["valid_commits"] / total * 100,
            "quantified_comparison_rate": counters["quantified"] / total * 100,
            "word_limit_compliance": counters["within_limit"] / total * 100,
            "violation_rate": counters["violations"] / total * 100,
            "total_files": total
        }
    
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
        cc_slope_by_cluster: Dict[str, List[float]] = {}
        for f in self.file_analysis:
            cluster_id = str(f.get('cluster_id'))
            churn_by_cluster.setdefault(cluster_id, []).append(
                f.get('raw_features', {}).get('churn_rate', 0)
            )
            cc_slope_by_cluster.setdefault(cluster_id, []).append(
                f.get('historical_trends', {}).get('cc', {}).get('value', 0)
            )

        self.cluster_stats = {}
        for cluster_id, churn_values in churn_by_cluster.items():
            median_churn = float(np.median(churn_values)) if churn_values else 0.0
            cc_values = cc_slope_by_cluster.get(cluster_id, [])
            median_cc_slope = float(np.median(cc_values)) if cc_values else 0.0
            self.cluster_stats[cluster_id] = {
                'median_churn': median_churn,
                'median_cc_slope': median_cc_slope
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
        signals: Dict[str, Any]
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

        # Determine if we should include commit evidence section
        has_meaningful_diffs = len(diffs) > 0 and any(
            len(d.get('document', '')) > 100 for d in diffs
        )

        if has_meaningful_diffs:
            commit_section = """### Evidence from Code Changes
- Cite 0-2 commits ONLY from the diffs shown above
- Cite ONLY commits with concrete evidence of metric impact
- If no commit clearly explains a metric trend, write: "Shown commits do not directly explain [metric] behavior"

Example of GOOD citation:
- Commit abc123d: Removed 3 nested if-statements, extracted 2 helper functions → explains complexity drop from 15 to 8 (47 percent reduction)

Example of BAD citation:
- Commit xyz789: Fixed merge conflicts → may have contributed to churn (too vague)"""
        else:
            commit_section = """### Evidence from Code Changes
Insufficient commit context available for causal analysis."""

        # Get cluster label
        cluster_id = str(file_data['cluster_id'])
        cluster_info = self.cluster_summary.get(cluster_id, {})
        cluster_label = cluster_info.get('derived_cluster_label', f'Cluster {cluster_id}')

        prompt = f"""You are explaining why this specific file deserves developer attention based on its evolutionary behavior.

## CONTEXT

**File:** {file_data['file_path']}
**Repository:** {file_data['repo_name']}
**Cluster:** {cluster_label}

## KEY SIGNALS (Use these first)

**Anomaly Level:** {signals['anomaly_context']}
**Churn Rate:** {signals['churn_comparison']} 
**Complexity Trend:** {file_data['historical_trends']['cc']['label']}, {signals['complexity_comparison']}
**Commits Analyzed:** {signals['n_commits']}

## SUPPORTING CONTEXT (Use if relevant)

- Maintainability trend: {file_data['historical_trends']['mi']['label']}
- Import volatility: {file_data.get('imports_volatility_label', 'N/A')}
- Raw churn rate: {file_data['raw_features']['churn_rate']:.1%}

## CODE DIFFS (Evidence Anchors)
{diff_text}

---

## INTERNAL ANALYSIS (Think through this first, do not output)

Before generating your response, answer these internally:

1. What is the MOST SPECIFIC unusual thing about this file?
    - NOT: "unusual combination of metrics"
    - YES: "changes frequently but simplifies" OR "grows complexity while reducing dependencies"

2. What are the EXACT numerical comparisons?
    - Calculate ratios from the signals above
    - Every metric bullet needs a number

3. Do the shown commits contain ANY of these concrete changes?
    - Function/class additions or removals
    - Control flow changes (if/switch/loop modifications)
    - Dependency additions or removals
    - If NO clear evidence, state it explicitly

4. What is the tension or deviation in ONE sentence?
    - Format: "X paired with Y is unusual because typical files show Z"

---

## BANNED OPENING PATTERNS (Do not use these)

❌ "This file stands out due to its unusual combination of..."
❌ "This file has a unique set of characteristics..."
❌ "This file exhibits interesting patterns..."
❌ "This file shows attention-worthy behavior..."
❌ "This file warrants investigation because..."

✅ REQUIRED OPENING PATTERN:

Start with a SPECIFIC behavioral description:
- "This file changes frequently but simplifies with each modification"
- "This file accumulates complexity while shedding dependencies"
- "This file remains stable despite being in a high-churn cluster"

## OUTPUT FORMAT (STRICT - FOLLOW EXACTLY)

### Why This File Stands Out
Write 1-2 sentences in plain English explaining what makes this file attention-worthy.
Do NOT use numbers or percentages here - save those for the next section.

### Evidence from Metrics
Provide exactly 3 bullet points. Each MUST:
- Include a specific quantified comparison (e.g., "2.3x cluster median", "3x faster growth")
- Compare this file to either its cluster or the repository average
- Be concrete, not vague

Example of GOOD bullets:
- Churn rate is 2.8x cluster median while complexity drops 1.5x faster - unusual inverse relationship
- Anomaly score places it in top 5% of repository - extreme outlier behavior
- Complexity decreased from ~15 to ~8 across 12 commits - 47% reduction

Example of BAD bullets:
- This file has high churn (no comparison, no number)
- Complexity is decreasing (no quantification)
- Metrics show unusual patterns (too vague)

{commit_section}

### What Makes This Unusual
Write exactly 1 sentence that:
1. Names the specific pattern (e.g., "high churn + declining complexity")
2. Explains what this pattern REFLECTS or IS CONSISTENT WITH (not what it "suggests" or "indicates")
3. States why this differs from typical behavior

Template: "[Pattern] reflects [interpretation], which differs from typical files that [normal behavior]"

Examples:
✅ "High churn paired with declining complexity reflects iterative simplification, which differs from typical high-activity files that accumulate complexity"
✅ "Stable complexity despite 80% churn reflects maintenance-focused changes rather than feature development"
✅ "Growing complexity with declining imports reflects logic consolidation, which differs from typical complexity growth that comes with new dependencies"

❌ "This combination suggests the file is worth investigating" (too vague)
❌ "High churn and low complexity indicate unusual patterns" (no interpretation)

---

## HARD CONSTRAINTS

❌ FORBIDDEN:
- Passive voice hedging ("has undergone", "has been modified", "has experienced")
- Vague qualifiers ("significant", "substantial", "notable", "considerable")
- Uncertainty markers ("suggests that", "appears to", "seems to indicate", "may have")
- Process narration ("development focused on", "efforts were made to")
- Recommendations ("should refactor", "needs attention", "consider")
- Risk predictions ("likely to cause bugs", "potential issues")
- Severity labels ("critical", "high risk", "problematic")

✅ REQUIRED:
- Active voice with specific numbers
- Quantified comparisons in EVERY metric bullet
- Commit citations ONLY when evidence is clear
- Max 220 words total
- Reference ONLY commits shown in diffs above

If you cannot make a specific, quantified comparison, do not include that bullet.
If commits don't explain metrics, explicitly say so - honesty over speculation.
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
        
        # Preprocessing: interpretation context
        signals = self._prepare_interpretation_context(
            file_data,
            getattr(self, 'cluster_stats', {})
        )

        # Build prompt
        prompt = self.build_file_prompt(file_data, diffs, signals)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a code evolution analyst writing for developers who need to prioritize attention.

## Core Responsibility
Describe what evolutionary patterns REFLECT or ARE CONSISTENT WITH, without claiming to know intent or predicting outcomes.

## Language Rules

FORBIDDEN phrases/patterns:
- Passive hedging: "has been", "has undergone", "has experienced"
- Uncertainty markers: "suggests that", "appears to", "seems to indicate", "may have"
- Vague intensifiers: "significant", "substantial", "considerable", "notable"
- Generic openers: "unusual combination", "unique characteristics", "interesting patterns"
- Recommendations: "should", "needs", "requires", "recommend"
- Risk language: "problematic", "concerning", "risky", "technical debt"

REQUIRED patterns:
- Active, specific descriptions: "changes 2.3× more than cluster median"
- Interpretive framing: "reflects [pattern]" not "suggests [speculation]"
- Concrete contrasts: "differs from typical files that..."
- Causal connections: "explains [metric change]" when citing commits

## Interpretive vs Speculative (Critical Distinction)

✅ INTERPRETIVE (allowed):
- "This pattern reflects iterative simplification"
- "Behavior is consistent with maintenance-focused development"
- "Differs from typical high-churn files that accumulate complexity"

❌ SPECULATIVE (forbidden):
- "Suggests the team is refactoring" (claims intent)
- "Likely to cause bugs" (predicts outcomes)
- "Indicates technical debt" (diagnoses problems)

## Evidence Standards

Every metric bullet MUST include:
- A specific number (2.3×, 47%, top 5%)
- A comparison (vs cluster, vs repository, vs typical)
- Context (what this number means)

Example: "Complexity decreased 47 percent over 8 months (slope: -0.095 vs cluster median +0.012) - only file showing decline while peers grow"

Commit citations MUST:
- Identify specific code change (what was modified)
- Connect change to metric behavior (how it explains the number)
- OR explicitly state "Shown commits do not explain [metric]"

## Core Principle
You describe patterns and what they reflect. You do not diagnose issues, predict risks, or recommend actions. Your outputs must be falsifiable using only the metrics and commits provided."""

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

            # Post-process to remove banned patterns that slipped through
            banned_replacements = {
                "suggests that": "reflects",
                "appears to": "shows",
                "seems to indicate": "shows",
                "may have": "likely",
                "has undergone": "experienced",
                "has been modified": "changed",
                "unusual combination of": "combines",
                "unique set of characteristics": "unique behavior",
                "interesting patterns": "patterns",
            }

            explanation_lower = explanation.lower()
            for banned, replacement in banned_replacements.items():
                if banned in explanation_lower:
                    # Find the actual case-sensitive occurrence
                    pattern = re.compile(re.escape(banned), re.IGNORECASE)
                    explanation = pattern.sub(replacement, explanation)
                    logger.warning(f"Auto-replaced banned phrase: '{banned}' → '{replacement}'")

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
        component_summary = self._build_component_summary(file_explanations)
        file_attention_snippets = self._extract_attention_signals(file_explanations)

        prompt = f"""You are synthesizing cross-file evolutionary patterns for a repository.

Your goal: Identify patterns that only become visible when comparing multiple files together.

## COMPONENT BREAKDOWN
{component_summary}

## FILE ATTENTION SIGNALS
(Only the "Why This File Stands Out" section from each file)

{file_attention_snippets}

---

## OUTPUT FORMAT (STRICT)

### Dominant Evolutionary Modes
Identify 2-3 themes that describe RELATIONSHIPS between metrics or components.

Each theme must:
- Name a specific pattern (not just label files)
- Include quantified evidence (percentages, ratios, file counts)
- Explain what the relationship means

✅ GOOD theme:
"UI Simplification vs Data Complexity: 5 files in pages/*.tsx reducing complexity by avg 35% (slopes: -0.06 to -0.12) while 3 files in components/*.tsx growing complexity by avg 28% (slopes: +0.04 to +0.08). Creates architectural divergence - presentation layer simplifying while data handling becomes more tangled."

❌ BAD themes:
- "High Churn Files" (just a label, no relationship)
- "Cluster 1 Behavior" (just renaming a cluster)
- "Active Development Zone" (too generic, no specifics)
- "Files undergoing significant changes" (vague, no numbers)

### Where Change Concentrates
List specific files or directories with disproportionate activity.

Include numbers:
- "pages/Dashboard.tsx and components/TopNavbar.tsx absorb 45% of total commits despite being 12% of codebase"
- "src/utils/ (8 files) accounts for 62% of high-anomaly files"

### Files That Defy Their Peers
List 2-4 files maximum. For each, write ONE LINE explaining how it deviates from its cluster or component.

Format: `**filename**: deviation in one sentence`

Example:
- **src/App.tsx**: Only file in stable-complexity cluster showing 40% growth - contradicts cluster pattern
- **pages/Login.tsx**: Churn rate 3.2x higher than other pages/ files while maintaining stable metrics

### What This Reveals
Answer these 3 questions in 2-3 sentences total:

1. **Architectural implication:** What does the pattern mean for code organization?
    - Use "reflects" or "is consistent with" framing
    - Example: "Reflects divergence between UI and data layers"
   
2. **Development pattern:** What does change concentration indicate?
    - Example: "Consistent with hotspot development rather than distributed evolution"
   
3. **Non-obvious insight:** What becomes visible only by comparing files?
    - Example: "ExampleChart becoming self-contained while other components remain modular - suggests different design philosophy"

Do NOT:
- Just restate the patterns listed above
- Make recommendations ("should refactor")
- Predict outcomes ("will cause issues")
- Claim intent ("team is focusing on...")

DO:
- Connect patterns to architectural consequences
- Use "reflects" or "consistent with" framing
- Provide falsifiable interpretations

---

## CONSTRAINTS

❌ FORBIDDEN:
- Recommendations ("should refactor", "needs attention")
- Risk language ("problematic", "concerning", "technical debt")
- Generic labels without evidence
- Restating individual file summaries verbatim
- Vague statements without numbers

✅ REQUIRED:
- Every theme must describe a relationship, not label files
- Include specific numbers (percentages, ratios, counts)
- Cross-file comparisons (component A vs component B)
- Concrete, falsifiable statements

If you cannot identify a genuine cross-file pattern, say: "Files do not exhibit clear cross-component patterns - evolution is file-specific rather than architectural."

Be honest. Be specific. Be concise.
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
                        "content": """You are synthesizing cross-file evolution patterns for a repository.

## Core Responsibility
Identify architectural and organizational patterns that emerge from comparing multiple files, using interpretive language that avoids speculation about intent or outcomes.

## Language Requirements

FORBIDDEN:
- Intent claims: "team is focusing on", "developers are prioritizing"
- Recommendations: "should", "needs to", "must", "consider"
- Risk predictions: "will cause", "likely to", "may lead to"
- Generic labels: "Active Development", "Maintenance Mode", "Legacy Code"
- Speculation: "suggests", "indicates", "implies"

REQUIRED:
- Architectural framing: "reflects divergence", "consistent with concentration"
- Comparative statements: "UI layer vs data layer", "component A differs from component B"
- Quantified evidence: percentages, file counts, slope comparisons
- Falsifiable claims: "5 files reducing complexity by avg 35%"

## Theme Quality Standards

GOOD theme structure:
"[Component/Pattern name]: [Quantified observation] while [Contrasting observation]. Reflects [architectural consequence]."

Example:
"UI Simplification: 5 files in pages/*.tsx reducing complexity by avg 38% (slopes: -0.06 to -0.12) while 3 files in components/*.tsx growing by avg 28% (slopes: +0.04 to +0.08). Reflects architectural divergence where presentation simplifies as data handling complexifies."

BAD themes to avoid:
- "High Churn Files" (just a label)
- "Active Development Zone" (generic)
- "Cluster 1 Behavior" (restates ML output)

## Synthesis Principle

You are NOT summarizing individual files.
You ARE identifying patterns that only become visible when viewing files together.

Focus on:
- Divergence (components evolving in opposite directions)
- Concentration (change focused in specific areas)
- Deviation (files behaving unlike their peers)
- Architectural consequences (what patterns mean for structure)

Never diagnose problems. Never recommend changes. Describe what patterns reflect about the codebase's evolutionary trajectory."""
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

        # Collect anomaly scores for percentile normalization
        self._collect_all_anomaly_scores(self.file_analysis)
        
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

        # Grounding & compliance metrics (for paper)
        grounding_metrics = self.compute_grounding_metrics(file_explanations)
        with open(output_path / "grounding_metrics.json", 'w') as f:
            json.dump(grounding_metrics, f, indent=2)
        
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
                    'architecture': 'neuro-symbolic',
                    'grounding_metrics': grounding_metrics
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
