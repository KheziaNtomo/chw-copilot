# CHW Copilot — 3-Minute Demo Script

> **Format:** Recorded screen walkthrough with voiceover.
> **Audience:** MedGemma Impact Challenge judges.
> **Tone:** Confident, clear, evidence-driven. No filler.

---

## INTRO — The Problem (0:00 – 0:25)

**[Screen: Title slide or landing page]**

> "Every day, community health workers across sub-Saharan Africa visit patients in their homes — documenting symptoms in shorthand notes like *'child 3yo fever cough rash not eating.'*
>
> These notes hold critical surveillance signals, but they're unstructured, noisy, and never aggregated. Outbreaks hide in plain sight.
>
> CHW Copilot uses MedGemma to turn these raw field notes into structured, evidence-grounded syndromic surveillance — running on-device, offline-first, and privacy-preserving."

---

## CHW FIELD VIEW — The Pipeline in Action (0:25 – 1:55)

**[Screen: Navigate to CHW Field View → Demo Cases tab]**

### Case 1: Severe Pneumonia — Infant (0:25 – 0:55)

**[Click "Severe pneumonia — infant"]**

> "Here's a note from a CHW: *'Baby 11 months, fever 2 days, cough, pulling in of chest when breathing, not breastfeeding well, unable to drink.'*
>
> MedGemma extracts structured symptoms — fever, cough — and critically identifies **three red flags**: chest indrawing, unable to drink, and poor feeding.
>
> The syndrome tagger classifies this as **respiratory fever, lower-respiratory sub-type** with high confidence. The ICCM recommendations engine immediately flags **REFER IMMEDIATELY** and advises a first dose of amoxicillin before transport.
>
> Every claim is grounded — you can see the green highlights linking each extraction back to the original text."

### Case 2: Cholera-like AWD Cluster (0:55 – 1:20)

**[Click "Cholera-like AWD — adult cluster"]**

> "Now a different scenario: *'Male 25, sudden diarrhea rice-water type, vomiting, co-workers also affected.'*
>
> The pipeline tags this as **acute watery diarrhea** — and importantly, detects a **cluster pattern**: co-workers are also affected. The recommendations don't just guide individual case management — they trigger an **outbreak alert**, telling the CHW to notify the district health team and investigate the common food or water source.
>
> This is where individual care meets population-level surveillance."

### Case 3: Unclear Presentation (1:20 – 1:40)

**[Click "Unclear presentation — fatigue"]**

> "Not every note maps cleanly. Here: *'Woman 23, dizziness and fatigue, no fever, no cough, no diarrhea.'*
>
> The system correctly tags this as **unclear** — low confidence. It doesn't force a diagnosis. Instead, it recommends follow-up and screening for anaemia or pregnancy. Knowing what you *don't* know is just as important for surveillance integrity."

### Failure Mode: Hallucination Detection (1:40 – 1:55)

**[Click "Failure Mode"]**

> "Safety is non-negotiable. Here, MedGemma reported 'rash: yes' even though the note says *'no rash observed.'* The evidence quote exists in the text — so a naive check passes — but our hallucination detector catches that the quote actually **contradicts** the claim. The budget gap of 9.7 bits flags it clearly.
>
> This is why we built a multi-agent pipeline, not a single model call."

---

## DISTRICT DASHBOARD — Population-Level Surveillance (1:55 – 2:40)

**[Screen: Switch to District Dashboard via sidebar]**

> "Everything we just processed feeds into this district-level dashboard.
>
> Two locations — Kibera Health Post and Mathare Community Center — across eight epi weeks.
>
> Watch the trend line for respiratory fever at Kibera: weeks 1 through 6 show a stable baseline of 3 to 6 cases. Then week 7 — a spike to **18 cases, 4.3 times baseline**. The system triggers an anomaly alert automatically.
>
> Week 8 shows 14 cases — still elevated. The SITREP narrative explains what this syndrome covers — malaria, pneumonia, ILI, URTI, TB — and recommends specific actions: deploy additional CHWs, conduct RDT testing, ensure ACT stock."

**[Scroll to SITREP section]**

> "This situation report is generated from the aggregated data — structured, actionable, and ready for the district health officer."

---

## CLOSE — Impact Statement (2:40 – 3:00)

**[Screen: Return to title or show pipeline diagram]**

> "CHW Copilot demonstrates that a small, on-device model — MedGemma 4B — embedded in a well-designed agentic pipeline can transform unstructured CHW notes into structured, evidence-grounded syndromic surveillance.
>
> It runs offline, preserves patient privacy, and scales to the contexts where surveillance is needed most — and where connectivity is the least reliable.
>
> Thank you."

---

## Timing Reference

| Section | Duration | Cumulative |
|---------|----------|------------|
| Intro | 25s | 0:25 |
| Severe Pneumonia | 30s | 0:55 |
| Cholera AWD Cluster | 25s | 1:20 |
| Unclear Case | 20s | 1:40 |
| Failure Mode | 15s | 1:55 |
| District Dashboard | 45s | 2:40 |
| Close | 20s | 3:00 |
