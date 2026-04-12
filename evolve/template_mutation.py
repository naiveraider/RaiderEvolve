from __future__ import annotations

import random


def template_mutate(code: str, task: str, rng: random.Random) -> tuple[str, str]:
    """Replace known substrings with task-specific alternatives (no new imports)."""
    if task == "pacman":
        pairs = [
            ("# pacman-agent", "# pacman-agent (template)"),
            ("nearest_food", "nearest_food"),
        ]
    else:
        pairs = [
            ("def matmul", "def matmul"),
            ("range(3)", "range(3)"),
        ]
    # Actual edits: comment injection / spacing tweaks
    lines = code.splitlines()
    if not lines:
        return code, "template_skip"
    idx = rng.randrange(len(lines))
    if task == "pacman":
        lines.insert(idx, "    # template: prefer survival")
        return "\n".join(lines), "template_insert_comment"
    lines.insert(idx, "    # template: reduce ops")
    return "\n".join(lines), "template_insert_comment_matrix"
