"""Lightweight entrypoint that demonstrates calling pipeline pieces.

This module is intentionally minimal — the notebook contains the end-to-end runnable demo.
"""
from .extract import stub_extract
from .validate import enforce_evidence
from .tag import stub_tag


def main():
    print("Pipeline stub: repo scaffold is ready.")


if __name__ == '__main__':
    main()
