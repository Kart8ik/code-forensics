"""
LLM-Only Baseline for Code Forensics
Realistic snapshot-based analysis without ML signals

Key characteristics:
- Uses only local commit snapshot inputs (file content + single commit diff + metadata)
- No FAISS/RAG, no ML signals, no clusters, no anomaly scores, no historical trends
- Produces descriptive explanations similar in format to hybrid outputs
- Two-stage structure: per-file explanations → repo synthesis
- Same Groq client, model (llama-3.3-70b-versatile), temperature (0.3)
- No recommendations, judgments, predictions, or quality labels

This represents a strong, fair baseline for LLM-based code review.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import os
from collections import defaultdict, Counter

from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "llm_only_output"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 10


class CodeForensicsLLMOnly:
    """
    LLM-only baseline for code evolution analysis.
    
    Operates on commit snapshots without ML signals or historical aggregation.
    Represents realistic LLM-based code review usage.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        temperature: float = 0.3,
        repo_filter: Optional[str] = None
    ):
        """
        Initialize LLM-only pipeline.
        
        Args:
            api_key: Groq API key (or from GROQ_API_KEY env var)
            top_k: Number of top files to explain (by commit frequency)
            temperature: LLM temperature (default 0.3 for determinism)
            repo_filter: Optional repo name to filter snapshots (e.g., 'flask', 'fastapi')
        """
        print("=" * 60)
        print("Code Forensics LLM-Only Baseline")
        print("NO ML SIGNALS – SNAPSHOT ANALYSIS ONLY")
        print("=" * 60)
        
        # Configuration
        self.top_k = top_k
        self.temperature = temperature
        self.repo_filter = repo_filter
        
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
        
        print(f"\n✓ Groq API initialized")
        print(f"✓ Model: {self.model}")
        print(f"✓ Temperature: {self.temperature}")
        print(f"✓ Top-K files: {self.top_k}")
        if self.repo_filter:
            print(f"✓ Repository filter: {self.repo_filter}")
        print(f"✓ Mode: LLM-ONLY (no FAISS, no ML signals)")
        print("=" * 60 + "\n")
    
    def load_data(self) -> None:
        """Load commit snapshots data."""
        print("Loading data files from:", DATA_DIR.absolute())
        print("-" * 60)
        
        snapshot_path = DATA_DIR / "commit_snapshots.json"
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Missing required file: {snapshot_path}\n"
                "Run: python scripts/generate_commit_snapshots.py"
            )
        
        with open(snapshot_path) as f:
            all_snapshots = json.load(f)
        
        # Filter by repository if specified
        if self.repo_filter:
            self.snapshots = [s for s in all_snapshots if s.get('repo_name') == self.repo_filter]
            print(f"✓ commit_snapshots.json ({len(all_snapshots)} total, {len(self.snapshots)} for {self.repo_filter})")
            if not self.snapshots:
                available_repos = set(s.get('repo_name') for s in all_snapshots)
                raise ValueError(
                    f"No snapshots found for repo '{self.repo_filter}'\n"
                    f"Available repos: {', '.join(sorted(available_repos))}"
                )
        else:
            self.snapshots = all_snapshots
            print(f"✓ commit_snapshots.json ({len(self.snapshots)} snapshots)")
        
        # Count files with content available
        with_content = sum(1 for s in self.snapshots if s.get('content_available', False))
        print(f"  Files with content: {with_content} ({with_content/len(self.snapshots)*100:.1f}%)")
        
        print("-" * 60 + "\n")
    
    def select_files(self) -> List[Dict[str, Any]]:
        """
        Select top-K files by commit frequency (deterministic).
        
        Returns:
            List of snapshot dicts for selected files
        """
        # Count commits per file
        file_commit_counts = Counter()
        file_to_snapshot = {}
        
        for snapshot in self.snapshots:
            file_path = snapshot['file_path']
            file_commit_counts[file_path] += 1
            # Keep the most recent snapshot (explicit date comparison)
            existing = file_to_snapshot.get(file_path)
            if not existing or snapshot['commit_date'] > existing['commit_date']:
                file_to_snapshot[file_path] = snapshot
        
        # Get top-K most frequently changed files
        top_files = [
            file_path for file_path, count 
            in file_commit_counts.most_common(self.top_k)
        ]
        
        # Return snapshots for top files
        selected = [file_to_snapshot[fp] for fp in top_files]
        
        logger.info(f"Selected {len(selected)} files by commit frequency")
        for i, snapshot in enumerate(selected[:5], 1):
            count = file_commit_counts[snapshot['file_path']]
            logger.info(f"  {i}. {snapshot['file_path']} ({count} commits)")
        if len(selected) > 5:
            logger.info(f"  ... and {len(selected) - 5} more")
        
        return selected
    
    def build_file_prompt(self, snapshot: Dict[str, Any]) -> str:
        """
        Build snapshot-only prompt for file explanation.
        
        Args:
            snapshot: Commit snapshot dict
        
        Returns:
            Prompt string
        """
        file_path = snapshot['file_path']
        commit_date = snapshot['commit_date']
        author = snapshot['author']
        commit_message = snapshot['commit_message']
        diff = snapshot['diff']
        current_content = snapshot['current_content']
        
        # Truncate content if too long
        max_content_chars = 4000
        if len(current_content) > max_content_chars:
            content_display = current_content[:max_content_chars] + "\n\n[... content truncated ...]"
        else:
            content_display = current_content if current_content else "[Content not available]"
        
        # Truncate diff if extremely long
        max_diff_chars = 3000
        if len(diff) > max_diff_chars:
            diff_display = diff[:max_diff_chars] + "\n\n[... diff truncated ...]"
        else:
            diff_display = diff if diff else "[No diff available]"
        
        prompt = f"""You are analyzing a single commit in a software repository.

## COMMIT METADATA

**File:** {file_path}
**Date:** {commit_date}
**Author:** {author}
**Message:** {commit_message}

## COMMIT DIFF

```
{diff_display}
```

## CURRENT FILE CONTENT (post-commit)

```
{content_display}
```

---

## YOUR TASK

Describe what changed in this commit to this file.

Focus on:
- What modifications were made (additions, deletions, refactorings)
- Apparent intent from diff and commit message
- Immediate code implications visible in the diff

## OUTPUT FORMAT

Write a structured explanation with these sections:

### Change Summary
2-3 sentences describing what was changed at a high level.

### Specific Modifications
List 2-4 key changes made in this commit:
- Be specific about what code was added/removed/modified
- Reference line changes visible in the diff
- Identify structural changes (new functions, removed classes, etc.)

### Apparent Intent
1-2 sentences on what the commit message and diff suggest about the purpose of this change.

### Code Implications
1-2 sentences on immediate technical implications visible in the diff (e.g., new dependencies, changed interfaces, simplified logic).

---

## CONSTRAINTS

❌ FORBIDDEN:
- Long-term trend analysis ("this file has been...")
- Comparisons to other files or repository patterns
- Quality judgments ("this is good/bad code")
- Recommendations ("should refactor", "needs improvement")
- Risk predictions ("might cause bugs", "potential issues")
- Speculation beyond what's visible in the diff

✅ REQUIRED:
- Descriptive only
- Grounded in visible changes
- Specific about what changed
- Neutral tone
- Max 200 words

Be precise. Be factual. Be concise.
"""
        return prompt
    
    def generate_file_explanation(
        self,
        snapshot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanation for a single file snapshot.
        
        Args:
            snapshot: Commit snapshot dict
        
        Returns:
            Dict with file, explanation, and metadata
        """
        file_path = snapshot['file_path']
        
        prompt = self.build_file_prompt(snapshot)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a code analyst describing what changed in a specific commit.

Your role: Describe visible changes objectively without making judgments or predictions.

Language rules:
- Be specific about what was added/removed/modified
- Reference actual code changes from the diff
- Avoid speculation about long-term trends or quality
- Avoid recommendations or advice
- Avoid predicting future issues or risks

Focus: What changed in THIS commit to THIS file, based on the diff and commit message.

Keep explanations under 200 words."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=800,
                top_p=0.9,
            )
            
            explanation = response.choices[0].message.content.strip()
            
            return {
                "repo": snapshot['repo_name'],
                "file": file_path,
                "commit_hash": snapshot['commit_hash'],
                "commit_date": snapshot['commit_date'],
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"API Error for {file_path}: {e}")
            return {
                "repo": snapshot['repo_name'],
                "file": file_path,
                "commit_hash": snapshot['commit_hash'],
                "commit_date": snapshot['commit_date'],
                "explanation": f"Error generating explanation: {e}"
            }
    
    def build_repo_synthesis_prompt(
        self,
        file_explanations: List[Dict[str, Any]]
    ) -> str:
        """
        Build repository-level synthesis prompt.
        
        Args:
            file_explanations: List of file explanation dicts
        
        Returns:
            Prompt string
        """
        # Extract summaries from each file
        summaries = []
        for exp in file_explanations:
            file_path = exp['file']
            explanation = exp['explanation']
            
            # Extract just the Change Summary section
            lines = explanation.split('\n')
            summary_lines = []
            in_summary = False
            
            for line in lines:
                if '### Change Summary' in line or '## Change Summary' in line:
                    in_summary = True
                    continue
                elif line.startswith('###') or line.startswith('##'):
                    if in_summary:
                        break
                elif in_summary and line.strip():
                    summary_lines.append(line.strip())
            
            if not summary_lines:
                logger.debug(f"No explicit Change Summary found for {file_path}, using truncation fallback")
            
            summary = ' '.join(summary_lines) if summary_lines else explanation[:200]
            summaries.append(f"**{file_path}**: {summary}")
        
        summaries_text = '\n'.join(summaries)
        
        prompt = f"""You are synthesizing common change patterns across multiple files.

## FILE CHANGE SUMMARIES

{summaries_text}

---

## YOUR TASK

Based ONLY on the per-file descriptions above, summarize common types of changes observed across these commits.

## OUTPUT FORMAT

### Common Change Patterns
Identify 2-3 recurring themes across the files:
- What types of modifications appear multiple times?
- Are there common intents visible in commit messages?
- Do changes concentrate in specific areas?

Use quantified statements when possible (e.g., "3 files modified imports", "2 files refactored error handling").

### Repository Change Profile
In 2-3 sentences, characterize the overall nature of changes in this analysis:
- What kinds of work do these commits represent?
- Are changes localized or distributed?
- What appears to be the focus of recent development?

---

## CONSTRAINTS

❌ FORBIDDEN:
- Temporal claims ("this repository is evolving towards...")
- Trend predictions ("will likely continue...")
- Quality judgments ("technical debt", "well-architected")
- Recommendations ("should", "needs", "consider")
- Speculation beyond what's stated in the summaries

✅ REQUIRED:
- Descriptive only
- Patterns visible across multiple files
- Grounded in the summaries provided
- Neutral tone
- Max 250 words

You are summarizing observed changes, not inferring repository-wide trends.
Be factual. Be specific. Be concise.
"""
        return prompt
    
    def generate_repo_synthesis(
        self,
        file_explanations: List[Dict[str, Any]]
    ) -> str:
        """
        Generate repository-level synthesis.
        
        Args:
            file_explanations: List of file explanation dicts
        
        Returns:
            Synthesis text
        """
        prompt = self.build_repo_synthesis_prompt(file_explanations)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are synthesizing patterns across multiple file changes.

Your role: Identify common themes and patterns without making temporal claims or quality judgments.

Language rules:
- Describe patterns visible across files
- Use quantified statements ("3 files", "most changes", etc.)
- Avoid trend predictions or evolution claims
- Avoid recommendations or advice
- Stay grounded in provided summaries

Focus: What patterns appear across these specific commits?

Keep synthesis under 250 words."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=1000,
                top_p=0.9,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"API Error for repo synthesis: {e}")
            return f"Error generating repository synthesis: {e}"
    
    def run_pipeline(self) -> None:
        """
        Run complete LLM-only pipeline.
        
        Stage 1: Generate per-file explanations (N API calls)
        Stage 2: Generate repository-level synthesis (1 API call)
        """
        output_path = OUTPUT_DIR
        output_path.mkdir(exist_ok=True)
        
        # Stage 1: File selection
        print("=" * 60)
        print("STAGE 1: File Selection")
        print("=" * 60)
        
        selected_files = self.select_files()
        
        print(f"\nSelected {len(selected_files)} files by commit frequency")
        print("=" * 60 + "\n")
        
        # Stage 2: File-level explanations
        print("=" * 60)
        print("STAGE 2: File-level Explanations")
        print(f"Processing {len(selected_files)} files")
        print(f"Temperature: {self.temperature}")
        print("=" * 60 + "\n")
        
        file_explanations = []
        
        for i, snapshot in enumerate(selected_files, 1):
            file_path = snapshot['file_path']
            print(f"[{i}/{len(selected_files)}] {file_path}")
            
            result = self.generate_file_explanation(snapshot)
            file_explanations.append(result)
            
            print(f"           ✓ Generated ({len(result['explanation'])} chars)\n")
            
            # Save checkpoint every 5 files
            if i % 5 == 0:
                with open(output_path / "checkpoint_llm_only.json", 'w') as f:
                    json.dump(file_explanations, f, indent=2)
                logger.info(f"Checkpoint saved at {i} files")
        
        # Save file explanations
        with open(output_path / "file_explanations_llm_only.json", 'w') as f:
            json.dump(file_explanations, f, indent=2)
        
        print(f"\n{'=' * 60}")
        print(f"✓ Saved {len(file_explanations)} file explanations")
        print(f"{'=' * 60}\n")
        
        # Stage 3: Repository-level synthesis
        print("=" * 60)
        print("STAGE 3: Repository Synthesis")
        print("Input: file_explanations_llm_only.json")
        print("NO ML signals | NO trends | NO structure")
        print("=" * 60 + "\n")
        
        print("Generating repository-level synthesis...")
        repo_synthesis = self.generate_repo_synthesis(file_explanations)
        
        # Save report
        with open(output_path / "repository_report_llm_only.md", 'w') as f:
            f.write("# Repository Change Summary (LLM-Only Baseline)\n\n")
            f.write("*This report describes observed changes without temporal trends or quality assessments.*\n\n")
            f.write("---\n\n")
            f.write(repo_synthesis)
        
        with open(output_path / "repository_report_llm_only.json", 'w') as f:
            json.dump({
                'synthesis': repo_synthesis,
                'metadata': {
                    'files_analyzed': len(file_explanations),
                    'model': self.model,
                    'temperature': self.temperature,
                    'top_k': self.top_k,
                    'mode': 'llm-only',
                    'architecture': 'snapshot-based'
                }
            }, f, indent=2)
        
        print(f"\n✓ Repository synthesis saved\n")
        
        print("=" * 60)
        print("Pipeline Complete")
        print("=" * 60)
        print(f"\nOutputs saved to: {output_path.absolute()}")
        print("\nFiles created:")
        print(f"  - file_explanations_llm_only.json ({len(file_explanations)} explanations)")
        print(f"  - repository_report_llm_only.md (synthesis)")
        print(f"  - repository_report_llm_only.json (structured)")
        print("\nBaseline characteristics:")
        print("  - LLM operates on commit snapshots only")
        print("  - No ML signals, clusters, or trends")
        print("  - No FAISS/RAG or historical context")
        print("  - Same model and temperature as hybrid")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    import sys
    
    print("\n" + "=" * 60)
    print("CODE FORENSICS - LLM-ONLY BASELINE")
    print("Snapshot-based analysis without ML signals")
    print("=" * 60 + "\n")
    
    # Parse arguments
    test_mode = "--test" in sys.argv
    
    # Get top_k from args if provided
    top_k = DEFAULT_TOP_K
    repo_filter = None
    
    for arg in sys.argv:
        if arg.startswith("--top-k="):
            try:
                top_k = int(arg.split("=")[1])
            except ValueError:
                pass
        elif arg.startswith("--repo="):
            repo_filter = arg.split("=")[1]
    
    if test_mode:
        print(f"TEST MODE: Processing only {min(5, top_k)} files\n")
        top_k = min(5, top_k)
    
    try:
        # Initialize pipeline
        pipeline = CodeForensicsLLMOnly(
            top_k=top_k,
            temperature=0.3,
            repo_filter=repo_filter
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
        print("  - data/commit_snapshots.json exists")
        print("\nTo generate:")
        print("  python scripts/generate_commit_snapshots.py")
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
