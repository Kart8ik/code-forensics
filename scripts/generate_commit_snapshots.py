"""
Generate commit_snapshots.json for LLM-only baseline.

Reads:
- commits.csv
- file_changes.csv  
- diffs.csv

Produces:
- data/commit_snapshots.json

For each file, selects the MOST RECENT commit and extracts:
- file_path
- commit_hash
- commit_date
- author
- commit_message
- diff
- current_content (post-commit file from git repo)
"""

import json
import csv
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPOS_DIR = BASE_DIR / "repos"

def load_commits() -> Dict[str, Dict]:
    """Load commits.csv into dict keyed by (repo_name, commit_hash)."""
    commits = {}
    
    with open(DATA_DIR / "commits.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['repo_name'], row['commit_hash'])
            commits[key] = {
                'repo_name': row['repo_name'],
                'commit_hash': row['commit_hash'],
                'author': row['author'],
                'timestamp': row['timestamp'],
                'message': row['message']
            }
    
    logger.info(f"Loaded {len(commits)} commits")
    return commits


def load_file_changes() -> Dict[str, List[Dict]]:
    """Load file_changes.csv grouped by file_path."""
    file_changes = defaultdict(list)
    
    with open(DATA_DIR / "file_changes.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_key = (row['repo_name'], row['file_path'])
            file_changes[file_key].append({
                'repo_name': row['repo_name'],
                'commit_hash': row['commit_hash'],
                'file_path': row['file_path'],
                'change_type': row['change_type']
            })
    
    logger.info(f"Loaded file changes for {len(file_changes)} unique files")
    return file_changes


def load_diffs() -> Dict[str, str]:
    """Load diffs.csv keyed by (repo_name, commit_hash, file_path)."""
    diffs = {}
    
    with open(DATA_DIR / "diffs.csv", 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row['repo_name'], row['commit_hash'], row['file_path'])
            diffs[key] = row['diff_text']
    
    logger.info(f"Loaded {len(diffs)} diffs")
    return diffs


def get_file_content_from_git(
    repo_name: str,
    commit_hash: str,
    file_path: str
) -> Optional[str]:
    """
    Retrieve post-commit file content using git show.
    
    Args:
        repo_name: Name of repository
        commit_hash: Commit hash
        file_path: File path relative to repo root
    
    Returns:
        File content as string, or None if unavailable
    """
    repo_path = REPOS_DIR / repo_name
    
    if not repo_path.exists():
        logger.warning(f"Repository not found: {repo_path}")
        return None
    
    try:
        result = subprocess.run(
            ["git", "show", f"{commit_hash}:{file_path}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid UTF-8 chars instead of crashing
            check=True,
            timeout=10
        )
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Could not load {file_path} at {commit_hash[:8]} in {repo_name}: "
            f"git returned {e.returncode}"
        )
        return None
    
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout loading {file_path} at {commit_hash[:8]}")
        return None
    
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO 8601 timestamp with timezone."""
    # Handle format: 2023-03-11T08:19:22-08:00
    try:
        # Try parsing with timezone
        from dateutil import parser
        return parser.parse(timestamp_str)
    except:
        # Fallback to basic parsing
        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            logger.warning(f"Could not parse timestamp: {timestamp_str}")
            return datetime.min


def generate_snapshots(
    commits: Dict,
    file_changes: Dict,
    diffs: Dict,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Generate commit snapshots for most recent commit per file.
    
    Args:
        commits: Dict of commit metadata
        file_changes: Dict of file changes grouped by file
        diffs: Dict of diffs
        limit: Optional limit on number of files to process
    
    Returns:
        List of snapshot dicts
    """
    snapshots = []
    files_processed = 0
    files_with_content = 0
    
    logger.info("Generating commit snapshots...")
    logger.info("=" * 60)
    
    # Process each unique file
    for file_key, changes in file_changes.items():
        repo_name, file_path = file_key
        
        # Sort changes by timestamp (most recent first)
        # This ensures we select the MOST RECENT commit per file for snapshot analysis
        changes_with_time = []
        for change in changes:
            commit_key = (change['repo_name'], change['commit_hash'])
            if commit_key in commits:
                timestamp_str = commits[commit_key]['timestamp']
                timestamp = parse_timestamp(timestamp_str)
                changes_with_time.append((timestamp, change))
        
        if not changes_with_time:
            continue
        
        # Select most recent commit
        changes_with_time.sort(reverse=True, key=lambda x: x[0])
        most_recent_change = changes_with_time[0][1]
        
        commit_hash = most_recent_change['commit_hash']
        commit_key = (repo_name, commit_hash)
        
        if commit_key not in commits:
            continue
        
        commit_meta = commits[commit_key]
        
        # Get diff
        diff_key = (repo_name, commit_hash, file_path)
        diff_text = diffs.get(diff_key, "")
        
        # Get post-commit file content
        current_content = get_file_content_from_git(
            repo_name,
            commit_hash,
            file_path
        )
        
        content_available = current_content is not None
        if content_available:
            files_with_content += 1
            # Truncate long files to prevent token overflow
            if len(current_content) > 8000:
                current_content = current_content[:8000] + "\n\n[... content truncated ...]"
        else:
            current_content = ""
        
        snapshot = {
            "file_path": file_path,
            "repo_name": repo_name,
            "commit_hash": commit_hash,
            "commit_date": commit_meta['timestamp'],
            "author": commit_meta['author'],
            "commit_message": commit_meta['message'],
            "diff": diff_text,
            "current_content": current_content,
            "content_available": content_available
        }
        
        snapshots.append(snapshot)
        files_processed += 1
        
        if files_processed % 100 == 0:
            logger.info(f"Processed {files_processed} files ({files_with_content} with content)")
        
        if limit and files_processed >= limit:
            logger.info(f"Reached limit of {limit} files")
            break
    
    logger.info("=" * 60)
    logger.info(f"Generated {len(snapshots)} snapshots")
    logger.info(f"Files with content: {files_with_content} ({files_with_content/len(snapshots)*100:.1f}%)")
    
    return snapshots


def main():
    """Main entry point."""
    import sys
    
    print("\n" + "=" * 60)
    print("GENERATE COMMIT SNAPSHOTS")
    print("For LLM-Only Baseline")
    print("=" * 60 + "\n")
    
    # Check for test mode
    test_mode = "--test" in sys.argv
    limit = 50 if test_mode else None
    
    if test_mode:
        print("TEST MODE: Processing only 50 files\n")
    
    # Load data
    print("Loading input data...")
    commits = load_commits()
    file_changes = load_file_changes()
    diffs = load_diffs()
    
    # Generate snapshots
    snapshots = generate_snapshots(commits, file_changes, diffs, limit=limit)
    
    # Save output
    output_path = DATA_DIR / "commit_snapshots.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ SUCCESS")
    print("=" * 60)
    print(f"\nOutput saved to: {output_path}")
    print(f"Total snapshots: {len(snapshots)}")
    print(f"Snapshots with content: {sum(1 for s in snapshots if s['content_available'])}")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
