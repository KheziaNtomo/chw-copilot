from pathlib import Path

cwd = Path.cwd()

DIRS = [
    "data_raw",
    "data_synth",
    "schemas",
    "prompts",
    "src",
    "app",
    "notebooks",
]

SRC_FILES = [
    "io.py",
    "validate.py",
    "extract.py",
    "checklist.py",
    "tag.py",
    "aggregate.py",
    "detect.py",
    "sitrep.py",
    "__init__.py",
]

OTHER_FILES = {
    "README.md": "# CHW Copilot\n\nRepo scaffold.\n",
    "requirements.txt": "pandas\nnumpy\npydantic\njsonschema\n",
    ".gitignore": "__pycache__/\n.ipynb_checkpoints/\n.env\n*.pyc\ndata_raw/\n",
}

def main():


    # directories
    for d in DIRS:
        (cwd / d).mkdir(parents=True, exist_ok=True)

    # src stubs
    for f in SRC_FILES:
        path = cwd / "src" / f
        if not path.exists():
            path.write_text(f'"""Stub: {f}"""\n', encoding="utf-8")

    # other top-level files
    for fname, content in OTHER_FILES.items():
        path = cwd / fname
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    # simple runner entrypoint
    run_py = cwd / "src" / "run_pipeline.py"
    if not run_py.exists():
        run_py.write_text(
            "def main():\n"
            "    print('Pipeline stub: repo scaffold is ready.')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n",
            encoding="utf-8"
        )

    print(f"Created repo scaffold at: {cwd}")

if __name__ == "__main__":
    main()
