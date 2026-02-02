import git
import logging

logger = logging.getLogger(__name__)


def get_git_info_gitpython():
    try:
        repo = git.Repo(search_parent_directories=True)

        commit_hash = repo.head.commit.hexsha
        short_hash = repo.git.rev_parse(commit_hash, short=7)

        branch = repo.active_branch.name if not repo.head.is_detached else "DETACHED"
        is_dirty = repo.is_dirty()

        tags = [tag.name for tag in repo.tags if tag.commit == repo.head.commit]

        return {
            "commit_hash": commit_hash,
            "short_hash": short_hash,
            "branch": branch,
            "is_dirty": is_dirty,
            "tags": tags,
            "commit_message": repo.head.commit.message.strip(),
            "author": str(repo.head.commit.author),
            "commit_date": repo.head.commit.committed_datetime.isoformat(),
        }

    except git.InvalidGitRepositoryError:
        logger.warning("Not in a git repository")
        return None
    except Exception as e:
        logger.error(f"Error with gitpython: {e}")
        return None
