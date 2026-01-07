from pathlib import Path
import csv
from git import Repo

# -----------------------
# Path setup (robust)
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

REPO_NAME = "flask"    #change this before running for each repo
REPO_PATH = PROJECT_ROOT / "repos" / REPO_NAME

OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

COMMITS_CSV = OUTPUT_DIR / "commits.csv"
FILE_CHANGES_CSV = OUTPUT_DIR / "file_changes.csv"
DIFFS_CSV = OUTPUT_DIR / "diffs.csv"

# -----------------------
# Config
# -----------------------
MAX_COMMITS = 500
SOURCE_EXTENSIONS = (".py", ".java")
MAX_DIFF_CHARS = 8000

# -----------------------
# Helpers
# -----------------------
def init_csv(path, header):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

# -----------------------
# Main extraction logic
# -----------------------
def main():
    if not REPO_PATH.exists():
        raise FileNotFoundError(f"Repo path does not exist: {REPO_PATH}")

    repo = Repo(REPO_PATH)

    init_csv(COMMITS_CSV,
             ["repo_name", "commit_hash", "parent_hash",
              "author", "timestamp", "message"])

    init_csv(FILE_CHANGES_CSV,
             ["repo_name", "commit_hash", "file_path", "change_type"])

    init_csv(DIFFS_CSV,
             ["repo_name", "commit_hash", "file_path", "diff_text"])

    with open(COMMITS_CSV, "a", newline="", encoding="utf-8") as commits_f, \
         open(FILE_CHANGES_CSV, "a", newline="", encoding="utf-8") as files_f, \
         open(DIFFS_CSV, "a", newline="", encoding="utf-8") as diffs_f:

        commits_writer = csv.writer(commits_f)
        files_writer = csv.writer(files_f)
        diffs_writer = csv.writer(diffs_f)

        for i, commit in enumerate(repo.iter_commits()):
            if i >= MAX_COMMITS:
                break

            # Skip merge commits
            if len(commit.parents) != 1:
                continue

            parent = commit.parents[0]

            # ---- Commit metadata ----
            commits_writer.writerow([
                REPO_NAME,
                commit.hexsha,
                parent.hexsha,
                commit.author.name,
                commit.committed_datetime.isoformat(),
                commit.message.strip().replace("\n", " ")
            ])

           # ---- File changes ----
            diffs = parent.diff(commit)

            for d in diffs:
                file_path = d.b_path or d.a_path
                change_type = d.change_type

                if not file_path:
                    continue

                files_writer.writerow([
                    REPO_NAME,
                    commit.hexsha,
                    file_path,
                    change_type
                ])

                # Only extract diffs for source files
                if change_type in ("A", "M") and file_path.endswith(SOURCE_EXTENSIONS):
                    try:
                        diff_text = repo.git.show(
                            commit.hexsha,
                            "--",
                            file_path
                        )

                        diff_text = diff_text[:MAX_DIFF_CHARS]

                        if diff_text.strip():
                            diffs_writer.writerow([
                                REPO_NAME,
                                commit.hexsha,
                                file_path,
                                diff_text
                            ])

                    except Exception:
                        # Some files genuinely have no textual diff
                        continue


if __name__ == "__main__":
    main()
