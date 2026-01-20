"""
LLM Pipeline for Code Forensics - WITH FAISS RAG
100% Architecture Compliant Version
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import os
from groq import Groq
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Import FAISS retriever (assumes faiss_rag.py is in same directory)
try:
    from faiss_rag import FAISSContextRetriever
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS module not found - will fall back to direct loading")


class CodeForensicsLLM:
    def __init__(self, api_key: str = None, use_faiss: bool = True):
        """Initialize with Groq API and optional FAISS"""
        print("="*60)
        print("Code Forensics LLM Pipeline")
        print("Using LLaMA-3 70B via Groq API")
        if use_faiss and FAISS_AVAILABLE:
            print("WITH FAISS RAG LAYER ✨")
        print("="*60)
        
        # Get API key
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
        if self.use_faiss:
            print("\n✓ Initializing FAISS RAG layer...")
            self.retriever = FAISSContextRetriever()
            try:
                self.retriever.load_index()
                print("✓ FAISS index loaded")
            except FileNotFoundError:
                print("⚠️  FAISS index not found - run build_faiss_index() first")
                print("   Falling back to direct loading")
                self.use_faiss = False
        
        print("\n✓ Groq API initialized")
        print(f"✓ Model: {self.model}")
        print(f"✓ RAG Mode: {'ENABLED' if self.use_faiss else 'DISABLED (direct load)'}")
        print("="*60 + "\n")
    
    def load_data(self):
        data_dir = DATA_DIR
        
        print("Loading data files from:", data_dir.absolute())
        print("-"*60)
        required = [
        data_dir / "file_analysis.json",
        data_dir / "cluster_summary.json",
        ]
        if self.use_faiss:
            required.append(data_dir / "faiss.index")

        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required files:\n" +
                "\n".join(f"  • {p}" for p in missing)
            )
        # Load ML outputs
        with open(data_dir / "file_analysis.json") as f:
            self.file_analysis = json.load(f)
        print(f"✓ file_analysis.json ({len(self.file_analysis)} files)")
        
        # Load cluster summary
        with open(data_dir / "cluster_summary.json") as f:
            cluster_data = json.load(f)
            
        # Convert list to dict if needed
        if isinstance(cluster_data, list):
            self.cluster_summary = {}
            for item in cluster_data:
                cluster_id = item.get('cluster_id', item.get('id', 'unknown'))
                self.cluster_summary[str(cluster_id)] = item
            print(f"✓ cluster_summary.json ({len(self.cluster_summary)} clusters) [converted from list]")
        else:
            self.cluster_summary = cluster_data
            print(f"✓ cluster_summary.json ({len(self.cluster_summary)} clusters)")
        
        # Only load commits/diffs if NOT using FAISS
        if not self.use_faiss:
            # Load commits
            self.commits = {}
            with open(data_dir / "commits.csv") as f:
                for row in csv.DictReader(f):
                    self.commits[row['commit_hash']] = row
            print(f"✓ commits.csv ({len(self.commits)} commits)")
            
            # Load diffs
            self.diffs = {}
            with open(data_dir / "diffs.csv") as f:
                for row in csv.DictReader(f):
                    key = (row['repo_name'], row['commit_hash'], row['file_path'])
                    self.diffs[key] = row['diff_text']
            print(f"✓ diffs.csv ({len(self.diffs)} diffs)")
            
            # Load file changes
            self.file_changes = defaultdict(list)
            with open(data_dir / "file_changes.csv") as f:
                for row in csv.DictReader(f):
                    self.file_changes[(row['repo_name'], row['file_path'])].append(row)
            print(f"✓ file_changes.csv")
        else:
            print("✓ Using FAISS for context retrieval (commits/diffs indexed)")
        
        print("-"*60 + "\n")
    
    def build_file_prompt_with_rag(self, file_data: Dict[str, Any]) -> str:
        """Build prompt using FAISS-retrieved context"""
        
        # Retrieve relevant context
        contexts = self.retriever.retrieve_context(file_data, k=8)
        
        # Separate by type
        cluster_context = ""
        diff_context = ""
        metrics_context = ""
        
        for ctx in contexts:
            ctx_type = ctx['metadata']['type']
            similarity = ctx['similarity']
            
            # Only include highly relevant contexts (>0.3 similarity)
            if similarity < 0.3:
                continue
            
            if ctx_type == 'cluster':
                cluster_context += f"\n{ctx['document']}\n"
            elif ctx_type == 'diff':
                diff_context += f"\n**Relevance: {similarity:.2f}**\n{ctx['document']}\n"
            elif ctx_type == 'metrics':
                metrics_context += f"\n{ctx['document']}\n"
        
        # Build prompt
        prompt = f"""Analyze the evolution of this source code file based on pre-computed metrics and retrieved relevant context.

FILE: {file_data['file_path']}
REPOSITORY: {file_data['repo_name']}

## METRICS & ANALYSIS

**Evolution Summary:**
- Total commits: {file_data['n_commits']}
- Cluster: {file_data['cluster_id']}
- Anomaly score: {file_data['anomaly_score']:.3f} (higher = more unusual patterns)

**Labels (rule-based):**
- Churn level: {file_data['churn_label']}
- Complexity volatility: {file_data['cc_volatility_label']}
- Import volatility: {file_data['imports_volatility_label']}

**Historical Trends:**
- Cyclomatic complexity: {file_data['historical_trends']['cc']['label']} (slope: {file_data['historical_trends']['cc']['value']:.4f})
- Maintainability index: {file_data['historical_trends']['mi']['label']} (slope: {file_data['historical_trends']['mi']['value']:.4f})
- Imports: {file_data['historical_trends']['imports']['label']} (slope: {file_data['historical_trends']['imports']['value']:.4f})

**Average Metrics:**
- Churn rate: {file_data['raw_features']['churn_rate']:.1%}
- Mean LOC delta: {file_data['raw_features']['mean_abs_loc_delta']:.1f}
- Mean complexity: {file_data['raw_features']['mean_cc_after']:.2f} (±{file_data['raw_features']['std_cc_after']:.2f})
- Mean maintainability: {file_data['raw_features']['mean_mi_after']:.2f}

## RELEVANT CLUSTER CONTEXT
{cluster_context if cluster_context else "No highly relevant cluster context found."}

## RELEVANT CODE CHANGES (Retrieved via RAG)
{diff_context if diff_context else "No highly relevant diffs found."}

## RELATED METRICS CONTEXT
{metrics_context if metrics_context else "No related metrics context."}

## YOUR TASK

Provide a natural language explanation (3-5 paragraphs) that:

1. **Interprets the metrics**: What do these numbers tell us about the file's health and evolution?
2. **Explains the patterns**: Why might the file be changing this way? What development patterns are evident?
3. **Identifies concerns**: Are there red flags? (e.g., high churn + rising complexity)
4. **Grounds in evidence**: Reference the retrieved contexts and metrics where relevant
5. **Provides recommendations**: What should developers do? Refactor? Add tests? Monitor closely?

Be specific, technical, and actionable. Focus on insights that would help engineering teams make decisions."""

        return prompt
    
    def build_file_prompt_direct(self, file_data: Dict[str, Any]) -> str:
        """Build prompt with direct data loading (fallback)"""
        
        repo_name = file_data['repo_name']
        file_path = file_data['file_path']
        
        # Get recent diffs for context
        changes = self.file_changes.get((repo_name, file_path), [])
        recent_changes = ""
        
        for change in changes[:2]:  # Last 2 commits
            commit_hash = change.get('commit_hash', '')
            diff_key = (repo_name, commit_hash, file_path)
            
            if diff_key in self.diffs:
                diff_text = self.diffs[diff_key]
                if len(diff_text) > 800:
                    diff_text = diff_text[:800] + "\n... (truncated)"
                
                commit_msg = self.commits.get(commit_hash, {}).get('message', 'N/A')
                recent_changes += f"\nCommit {commit_hash[:7]}: {commit_msg[:100]}\n"
                recent_changes += f"```diff\n{diff_text}\n```\n"
        
        prompt = f"""Analyze the evolution of this source code file based on pre-computed metrics and recent changes.

FILE: {file_path}
REPOSITORY: {repo_name}

## METRICS & ANALYSIS

**Evolution Summary:**
- Total commits: {file_data['n_commits']}
- Cluster: {file_data['cluster_id']}
- Anomaly score: {file_data['anomaly_score']:.3f}

**Labels (rule-based):**
- Churn level: {file_data['churn_label']}
- Complexity volatility: {file_data['cc_volatility_label']}
- Import volatility: {file_data['imports_volatility_label']}

**Historical Trends:**
- Cyclomatic complexity: {file_data['historical_trends']['cc']['label']} (slope: {file_data['historical_trends']['cc']['value']:.4f})
- Maintainability index: {file_data['historical_trends']['mi']['label']} (slope: {file_data['historical_trends']['mi']['value']:.4f})

**Average Metrics:**
- Churn rate: {file_data['raw_features']['churn_rate']:.1%}
- Mean complexity: {file_data['raw_features']['mean_cc_after']:.2f}

## RECENT CHANGES
{recent_changes if recent_changes else "No recent changes available."}

## YOUR TASK

Provide a natural language explanation (3-5 paragraphs) that interprets these metrics and provides actionable insights."""

        return prompt
    
    def generate_file_explanation(self, file_data: Dict[str, Any]) -> str:
        """Generate explanation using Groq API with RAG"""
        
        # Build prompt (RAG or direct)
        if self.use_faiss:
            prompt = self.build_file_prompt_with_rag(file_data)
        else:
            prompt = self.build_file_prompt_direct(file_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior software architect analyzing code evolution patterns. Provide detailed, technical, and actionable insights based on metrics and code changes."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=0.9,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"      ✗ API Error: {e}")
            return f"Error generating explanation: {e}"
    
    def build_repo_prompt(self, file_explanations: List[Dict[str, Any]]) -> str:
        """Build repository-level report prompt"""
        
        # Cluster overview
        cluster_text = "## CLUSTER PATTERNS\n\n"
        for cluster_id, cluster_info in self.cluster_summary.items():
            cluster_text += f"""**Cluster {cluster_id}: {cluster_info.get('description', 'N/A')}**
- Size: {cluster_info.get('size', 0)} files
- Avg churn: {cluster_info.get('avg_churn_rate', 0):.1%}
- Avg complexity: {cluster_info.get('avg_complexity', 0):.1f}
- Avg anomaly: {cluster_info.get('avg_anomaly_score', 0):.3f}

"""
        
        # Top anomalous files
        sorted_files = sorted(
            file_explanations,
            key=lambda x: x['file_analysis']['anomaly_score'],
            reverse=True
        )
        
        top_files = "\n## TOP ANOMALOUS FILES\n\n"
        for i, item in enumerate(sorted_files[:5], 1):
            top_files += f"""### {i}. {item['file_path']}
**Anomaly Score:** {item['file_analysis']['anomaly_score']:.4f}
**Cluster:** {item['file_analysis']['cluster_id']}
**Churn:** {item['file_analysis']['churn_label']}

**Analysis:**
{item['explanation'][:500]}...

---

"""
        
        prompt = f"""Generate a comprehensive repository-level evolution report based on the analysis below.

{cluster_text}

{top_files}

## YOUR TASK

Create a detailed executive report with the following sections:

### 1. Executive Summary (3-4 paragraphs)
- Overall codebase health assessment
- Key evolutionary trends
- Critical areas of concern
- Strategic recommendations

### 2. Cluster Analysis
For each cluster:
- What characterizes files in this cluster?
- What are the dominant patterns?
- Is this cluster healthy or concerning?

### 3. Risk Assessment
- Identify top 5 files that need immediate attention
- Explain why they're risky
- Provide specific mitigation strategies

### 4. Recommendations
- Short-term actions (next sprint)
- Medium-term improvements (next quarter)
- Long-term architectural considerations

Make this actionable for engineering leadership. Be specific with file names, metrics, and concrete next steps."""

        return prompt
    
    def generate_repo_report(self, file_explanations: List[Dict[str, Any]]) -> str:
        """Generate repository-level report using Groq API"""
        
        prompt = self.build_repo_prompt(file_explanations)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a CTO analyzing a codebase evolution report. Provide strategic, actionable insights for engineering leadership."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3000,
                top_p=0.9,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"✗ API Error: {e}")
            return f"Error generating report: {e}"
    
    def run_pipeline(self, output_dir: str = "output", limit: int = None):
        """Run complete pipeline"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        files = self.file_analysis[:limit] if limit else self.file_analysis
        
        print("="*60)
        print("LAYER 1: File-level Explanations")
        print(f"Processing {len(files)} files")
        if self.use_faiss:
            print("Using FAISS RAG for context retrieval ✨")
        print("="*60 + "\n")
        
        file_explanations = []
        
        for i, file_data in enumerate(files, 1):
            file_path = file_data['file_path']
            print(f"[{i}/{len(files)}] {file_path}")
            
            try:
                explanation = self.generate_file_explanation(file_data)
                
                file_explanations.append({
                    'repo_name': file_data['repo_name'],
                    'file_path': file_path,
                    'cluster_id': file_data['cluster_id'],
                    'file_analysis': file_data,
                    'explanation': explanation
                })
                
                print(f"           ✓ Generated ({len(explanation)} chars)\n")
                
                # Save checkpoint every 10 files
                if i % 10 == 0:
                    with open(output_dir / "checkpoint.json", 'w') as f:
                        json.dump(file_explanations, f, indent=2)
                    print(f"           💾 Checkpoint saved\n")
                
            except Exception as e:
                print(f"           ✗ Error: {e}\n")
                continue
        
        # Save file explanations
        with open(output_dir / "file_explanations.json", 'w') as f:
            json.dump(file_explanations, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Saved {len(file_explanations)} file explanations")
        print(f"{'='*60}\n")
        
        # Generate repo report
        print("="*60)
        print("LAYER 2: Repository Report")
        print("="*60 + "\n")
        
        print("Generating repository-level report...")
        repo_report = self.generate_repo_report(file_explanations)
        
        # Save report
        with open(output_dir / "repository_report.md", 'w') as f:
            f.write("# Repository Evolution Report\n\n")
            f.write(repo_report)
        
        with open(output_dir / "repository_report.json", 'w') as f:
            json.dump({
                'report': repo_report,
                'metadata': {
                    'files_analyzed': len(file_explanations),
                    'clusters': len(self.cluster_summary),
                    'model': self.model,
                    'provider': 'Groq',
                    'rag_enabled': self.use_faiss
                }
            }, f, indent=2)
        
        print(f"\n✓ Repository report saved\n")
        
        print("="*60)
        print("🎉 Pipeline Complete!")
        print("="*60)
        print(f"\nOutputs saved to: {output_dir.absolute()}")
        print("\nFiles created:")
        print(f"  • file_explanations.json - {len(file_explanations)} analyses")
        print(f"  • repository_report.md - Executive summary")
        print(f"  • repository_report.json - Structured data")
        if self.use_faiss:
            print(f"  • RAG-enhanced with scoped context retrieval ✨")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    import sys
    
    print("\n" + "="*60)
    print("CODE FORENSICS - LLM PIPELINE")
    print("WITH FAISS RAG LAYER")
    print("="*60 + "\n")
    
    # Check for test mode
    test_mode = "--test" in sys.argv
    limit = 5 if test_mode else None
    
    if test_mode:
        print("🧪 TEST MODE: Processing only 5 files\n")
    
    try:
        # Initialize pipeline with FAISS
        pipeline = CodeForensicsLLM(use_faiss=True)
        
        # Load data
        pipeline.load_data()
        
        # Run pipeline
        pipeline.run_pipeline(limit=limit)
        
    except ValueError as e:
        print(f"\n{'='*60}")
        print("❌ CONFIGURATION ERROR")
        print("="*60)
        print(f"\n{e}\n")
        print("="*60 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n{'='*60}")
        print("❌ MISSING FILES")
        print("="*60)
        print("\nRequired files not found. Please ensure:")
        print("  • data/file_analysis.json exists")
        print("  • data/cluster_summary.json exists")
        print("  • data/faiss.index exists (run build_faiss_index() first)")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("❌ ERROR")
        print("="*60)
        print(f"\n{e}\n")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()