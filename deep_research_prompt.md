# Deep Research Prompt — CHW Copilot for The MedGemma Impact Challenge

Copy everything below this line into a deep-research-capable LLM (Gemini Deep Research, ChatGPT Deep Research, Perplexity, etc.)

---

## Context

I am entering **The MedGemma Impact Challenge** — a $100,000 featured hackathon on Kaggle hosted by **Google Research**. Deadline is **February 24, 2026** (10 days from now). There are 6,041 entrants, 142 teams have submitted so far.

The competition asks participants to **build human-centered AI applications** using MedGemma and other open models from Google's Health AI Developer Foundations (HAI-DEF).

I have built a project called **CHW Copilot** and I need deep research on how to improve it to maximize my competition score. I need to understand what I'm missing, what the strongest possible version of this project looks like, and what concrete steps I should take in the next 10 days.

---

## Competition Details

### Description
> AI is already reshaping medicine, but many clinical environments can't rely on large, closed models that require constant internet access or centralized infrastructure. They need adaptable, privacy-focused tools that can run anywhere care is delivered. Google has released open-weight models specifically designed for healthcare — MedGemma and the rest of the HAI-DEF collection. In this competition, you'll use these models to build full-fledged demonstration applications.

### Submission Requirements
- **Kaggle Writeup** (3 pages or less) — attached to the competition page
- **Video demo** (3 minutes or less) — mandatory
- **Public code repository** — mandatory
- **Bonus**: Public interactive live demo app
- **Bonus**: Open-weight Hugging Face model tracing to a HAI-DEF model

### Required Writeup Template
```
### Project name
[A concise name for your project.]

### Your team
[Name your team members, their speciality and the role they played.]

### Problem statement
[Your answer to the "Problem domain" & "Impact potential" criteria]

### Overall solution
[Your answer to "Effective use of HAI-DEF models" criterion]

### Technical details
[Your answer to "Product feasibility" criterion]
```

The writeup should be high-level — the video should convey most concepts.

### Judging Criteria (exact wording from judges)

**1. Effective Use of HAI-DEF Models (20%)**
> Are HAI-DEF models used appropriately? You will be assessed on: whether the submission proposes an application that uses HAI-DEF models to their fullest potential, where other solutions would likely be less effective. Note: Use of at least one of HAI-DEF models such as MedGemma is mandatory.

**2. Problem Domain (15%)**
> How important is this problem to solve and how plausible is it that AI is the right solution? You will be assessed on: storytelling, clarity of problem definition, clarity on whether there is an unmet need, the magnitude of the problem, who the user is and their improved journey given your solution.

**3. Impact Potential (15%)**
> If the solution works, what impact would it have? You will be assessed on: clear articulation of real or anticipated impact of your application within the given problem domain and description of how you calculated your estimates.

**4. Product Feasibility (20%)**
> Is the technical solution clearly feasible? You will be assessed on: technical documentation detailing model fine-tuning, model's performance analysis, your user-facing application stack, deployment challenges and how you plan on overcoming them. Consideration of how a product might be used in practice, rather than only for benchmarking.

**5. Execution & Communication (30%)**
> What is the quality of your project's execution and your clear and concise communication of your work? Your main submission package follows the provided template and includes a mandatory video demo and a write-up with links to your source material. You will be assessed on: the clarity, polish, and effectiveness of your video demonstration; the completeness and readability of your technical write-up; and the quality of your source code (e.g., organization, comments, reusability). Judges will look for a cohesive and compelling narrative across all submitted materials.

### Prize Tracks

| Track | Prize | Description |
|-------|-------|-------------|
| **Main Track** | $75,000 (1st: $30K, 2nd: $20K, 3rd: $15K, 4th: $10K) | Best overall projects — vision, technical execution, real-world impact |
| **Agentic Workflow Prize** | $10,000 ($5K × 2) | Most effective use of HAI-DEF models as intelligent agents or callable tools to reimagine a complex workflow |
| **Novel Task Prize** | $10,000 ($5K × 2) | Most impressive fine-tuned model adapting HAI-DEF for a task it wasn't originally trained on |
| **Edge AI Prize** | $5,000 | Best adaptation of HAI-DEF to run on local/edge devices (phone, scanner, lab instrument) |

Each submission can enter the Main Track + one special prize track.

### Judges
All from Google Research / Google Health AI / Google DeepMind — including ML engineers, research scientists, product managers, and clinical research scientists.

---

## My Project: CHW Copilot

**One-line description:** An offline-first tool that turns Community Health Worker (CHW) field notes into structured medical encounters and syndromic surveillance alerts, using MedGemma as the sole LLM.

### Architecture

```
CHW Note (typed text)
    → [MedGemma-4b-it] → Structured Encounter JSON (schema-validated, evidence-grounded)
    → [MedGemma-4b-it] → Syndrome Tag (respiratory_fever | acute_watery_diarrhea | other | unclear)
    → [MedGemma-4b-it] → Checklist of missing clinical questions for CHW to ask
    → Deterministic aggregation → Weekly counts by location/syndrome
    → Deterministic anomaly detection (z-score, baseline from prior 4 weeks)
    → [MedGemma-4b-it] → Weekly SITREP (situation report) narrative
```

### What MedGemma Does in My Pipeline (4 distinct tasks)

1. **Structured extraction**: Free-text CHW note → JSON with symptoms (fever, cough, diarrhea, etc.), patient demographics, red flags, treatments, referral info. Every "yes" symptom requires a verbatim `evidence_quote` from the note. A post-processing layer downgrades hallucinated claims to "unknown".

2. **Syndrome tagging**: Extracted encounter → classified into respiratory_fever, acute_watery_diarrhea, other, or unclear. Uses only syndromic language (never diagnostic terms). Includes confidence and trigger quotes.

3. **Checklist generation**: Identifies missing clinical info → generates up to 5 plain-language follow-up questions for the CHW, prioritized by clinical importance (danger signs first).

4. **SITREP generation**: Weekly counts + anomaly alerts → structured situation report with narrative, alerts, data quality notes, and limitations.

### Key Technical Features
- **Evidence grounding enforcement**: Post-processing checks every `evidence_quote` is a verbatim substring of the source. Hallucinated claims are downgraded. Hard safety guardrail.
- **JSON Schema validation**: 4 formal schemas (encounter, syndrome, checklist, SITREP)
- **Deterministic fallbacks**: Every MedGemma stage has a rule-based fallback if parsing fails
- **Prompt optimizer agents**: Templates for sensitivity/specificity improvement (not yet run)
- **Synthetic gold data**: 60 CHW-style notes with gold syndrome labels (inspired by AfriMed-QA from sub-Saharan Africa)
- **Simulation data**: 672 events across 8 locations/8 weeks with injected outbreaks

### What I'm Missing
- **No working UI** — `app/` directory is empty, no Streamlit or other demo exists
- **No video demo** — mandatory for submission
- **No Kaggle Writeup** — mandatory for submission
- **No saved model metrics** — haven't run full pipeline on Kaggle with recorded results
- **Only 1 test file** — minimal coverage
- **No deployment discussion** — "offline-first" claim but pipeline needs GPU + internet
- **No impact numbers** — no quantified real-world reach estimates
- **Text-only** — not using MedGemma's multimodal image capabilities
- **No fine-tuning** — using base MedGemma-4b-it with prompt engineering only

### Tech Stack
- Python, PyTorch, HuggingFace Transformers
- MedGemma-4b-it (google/medgemma-4b-it) — the only model
- Kaggle T4 GPU for inference
- pandas, jsonschema
- (Planned) Streamlit for demo app

---

## What I Need You to Research

Do deep research on each area below and provide specific, actionable recommendations with sources.

### A. MedGemma & HAI-DEF Ecosystem
1. What models are in the HAI-DEF collection beyond MedGemma? What are their specialties? Should I use multiple?
2. What are MedGemma's documented strengths, weaknesses, and published benchmarks?
3. Are there published examples, discussion threads, or Kaggle notebooks from OTHER participants in THIS competition that show effective MedGemma use patterns?
4. MedGemma is multimodal — could I creatively use image capabilities for CHW workflows (wound photos, rash ID, MUAC screening, medication identification)?
5. Has anyone published on fine-tuning MedGemma for low-resource or African health contexts?
6. What is the difference between MedGemma-4b-it and other MedGemma variants? Am I using the right one?

### B. Which Prize Track Should I Target?
Given my project architecture (multi-stage pipeline with MedGemma as 4 distinct agents):
1. My pipeline is inherently **agentic** — should I emphasize this and target the **Agentic Workflow Prize**?
2. Could I adapt the project for the **Edge AI Prize** by quantizing MedGemma?
3. Could I add a fine-tuning component to target the **Novel Task Prize**?
4. What's the strategic play — which track gives me the best chance given 10 days left?

### C. Problem Domain & Impact Research
1. Latest WHO/UNICEF numbers on CHW workforce globally and in sub-Saharan Africa
2. What existing digital tools do CHWs use? (CommCare, DHIS2, Open Data Kit) — what are their gaps?
3. What is IDSR (Integrated Disease Surveillance and Response) and how does my project fit?
4. Most impactful syndromic surveillance success stories (cholera, Ebola, COVID early detection)
5. Real-world delays: How long does a disease signal from a CHW take to reach a district health officer today?
6. Published papers on NLP/LLM for CHW notes or syndromic surveillance in LMICs?
7. Help me build a credible impact model with real numbers (CHWs × notes/week × detection delay improvement → lives saved)

### D. Product Feasibility & Deployment
1. Realistic options for running MedGemma offline on edge devices (quantization, GGUF, TensorRT, on-device inference)
2. What hardware exists in African district health offices that could realistically run this?
3. How do existing mHealth apps handle offline-first + data sync?
4. Privacy/regulatory: Kenya Data Protection Act, Nigeria NDPR, etc.
5. What would a realistic deployment architecture look like?

### E. Execution & Communication Strategy
1. What makes an exceptional 3-minute hackathon demo video? Structure, pacing, what to show vs. tell?
2. What should the 3-page writeup cover that most submissions miss?
3. Examples of winning health AI competition submissions?
4. The judges are all Google Research engineers and PMs — what would THEY specifically care about? What impresses ML engineers vs. clinical researchers vs. product managers?
5. With only 10 days left, what's the highest-impact use of my time?

### F. Competitive Analysis
1. With 142 submissions so far, what kinds of projects are others likely building? (Search Kaggle discussions, notebooks, and writeups for this competition)
2. What are the most common mistakes in health AI hackathon entries?
3. What would make my CHW surveillance angle stand out from medical imaging or diagnostic projects?
4. Are CHW/community health/surveillance projects underrepresented? (Could be an advantage)

---

## Desired Output Format

For each section (A through F), provide:
1. **Key findings** — most important facts and data points (with sources/URLs)
2. **Specific recommendations** — exactly what I should change, add, or emphasize
3. **Priority ranking** — which recommendations have the biggest score impact

End with a **consolidated 10-day action plan** — the top 10–15 things I should do, ranked by score impact, with estimated time per item and which judging criteria each action addresses.
