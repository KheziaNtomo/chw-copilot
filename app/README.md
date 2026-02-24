---
title: CHW Copilot
emoji: 🏥
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.41.0"
app_file: app.py
pinned: false
license: apache-2.0
hardware: t4-small
---

# CHW Copilot — Agentic Syndromic Surveillance

MedGemma-powered tool that transforms unstructured Community Health Worker field notes into structured, evidence-grounded syndromic surveillance data.

## Features
- **CHW Field View**: Process free-text clinical notes through a 6-agent pipeline
- **District Dashboard**: Aggregated surveillance trends with anomaly detection
- **Live MedGemma**: When GPU is available, runs MedGemma 1.5 4B-IT for extraction, tagging, and checklist generation
- **Offline Mode**: Falls back to deterministic rules when no GPU is available

## Setup
Set `HF_TOKEN` in Space secrets for gated model access.
Optionally set `PASSWORD` in secrets to protect the app.
