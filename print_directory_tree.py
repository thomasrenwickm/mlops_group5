from pathlib import Path

# Folders to skip (non-essential, logs, metadata)
EXCLUDE_DIRS = {
    "mlruns", ".git", ".dvc", "wandb", "__pycache__", ".idea", ".vscode", ".pytest_cache"
}

# Files to skip (log files, temp files, compiled Python, etc.)
EXCLUDE_FILES_EXT = {".log", ".tmp", ".pyc", ".pyo"}

def print_tree(directory: Path, prefix: str = "") -> list:
    lines = []
    if directory.name in EXCLUDE_DIRS:
        return lines

    contents = sorted([p for p in directory.iterdir() if not p.name.startswith(".")])
    contents = [p for p in contents if p.name not in EXCLUDE_DIRS]

    pointers = ["├── "] * (len(contents) - 1) + ["└── "]

    for pointer, path in zip(pointers, contents):
        if path.is_file() and path.suffix in EXCLUDE_FILES_EXT:
            continue
        lines.append(prefix + pointer + path.name)
        if path.is_dir():
            extension = "│   " if pointer == "├── " else "    "
            lines += print_tree(path, prefix + extension)
    return lines

if __name__ == "__main__":
    root = Path(".").resolve()
    tree_lines = [root.name] + print_tree(root)
    output_path = root / "project_tree.txt"

    with open(output_path, "w") as f:
        f.write("\n".join(tree_lines))

    print(f"\n✅ Project tree saved to: {output_path}")
