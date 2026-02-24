# CHW Copilot — 3-Minute Video Script

> **Format:** Screen recording (OBS/Loom) + voiceover — no face cam needed
> **Audience:** Google Research judges (MedGemma Impact Challenge)
> **Tone:** Confident, clear, evidence-driven. This is a pitch, not a tutorial.
> **Tracks:** Main Track + Agentic Workflow Prize
> **Timed read:** ~2:50 — leaves ~10s buffer for transitions
> **Key reminder:** "Less is more" — the video conveys concepts, the write-up has the detail.

---

## Pre-Recording Checklist

- [ ] Streamlit app running and loaded (HF Spaces or local) — HF URL visible in browser bar
- [ ] Demo case pre-selected (severe pneumonia, 11-month-old)
- [ ] District Dashboard tab ready with Kibera Health Post visible
- [ ] Browser zoom at 150%, terminal font large — judges watch on small screens
- [ ] 2–3 slides ready: title, stats, pipeline diagram, closing with links
- [ ] Quiet room, headset mic or external mic
- [ ] OBS/Loom set to capture browser window only (no desktop chrome)

---

## SECTION 1 — THE PROBLEM (0:00 – 0:40)

**🖥️ ON SCREEN:** Title slide (green gradient, project name, team, track) → fade to stats slide (3.8M CHWs, 44% timeliness, cholera 50→5000 stat). 2–3 clean slides, not cluttered.

> "When an outbreak starts, the first people to see it aren't doctors in hospitals — they're community health workers going door to door in villages and urban slums. There are 3.8 million of them across 98 countries, and collectively they are the largest and earliest source of disease signals in the world.
>
> But their observations are written as shorthand paper notes — *'child 3 years, fever, cough, not eating.'* These notes never reach surveillance systems in time. In Uganda, less than half of weekly disease reports arrive by the deadline.
>
> Why does speed matter? Because with cholera, detecting a cluster on day 2 versus day 14 can be the difference between 50 cases and 5,000. With pneumonia — the leading infectious killer of children — recognising a danger sign like chest indrawing triggers immediate referral that saves lives.
>
> We know this is solvable. Nigeria piloted an electronic disease surveillance system and saw reporting timeliness jump from 43% to 73%. Eighty percent of verified outbreak alerts came from the electronic sites. And just this month, a study published in *Nature Health* evaluated LLMs supporting 101 community health workers in Rwanda — LLMs outperformed local clinicians on every single quality metric, at one five-hundredth of the cost. The bottleneck isn't the CHWs — it's the gap between their observations and the structured data that surveillance systems need.
>
> CHW Copilot closes that gap. It uses MedGemma to read a raw field note and produce a structured, evidence-grounded encounter — in about one minute."

---

## SECTION 2 — WHY MEDGEMMA (0:40 – 1:10)

**🖥️ ON SCREEN:** Split view — raw CHW note (messy shorthand) on left, clean MedGemma JSON output on right. This is the "wow" moment. Can use a static screenshot from the Streamlit app or notebook output.

> "Why MedGemma, specifically? These notes are messy, abbreviated, and full of clinical shorthand. *'Hot body'* means fever. *'Pulling in of chest'* means chest indrawing — a WHO danger sign. *'Rice-water stool'* means cholera-like diarrhoea.
>
> General-purpose LLMs miss these. I tested them — they hallucinate clinical terms and miss the shorthand. MedGemma's medical pre-training means it understands this vocabulary immediately, without fine-tuning.
>
> I also built a full keyword-based fallback and tested it against 60 gold-standard CHW encounters. The keyword system reaches 95% syndrome accuracy, but it can't extract patient demographics, can't identify severity or red flags, and can't capture illness onset duration. MedGemma does all of this from a single prompt — achieving 85% syndrome accuracy and 100% evidence grounding on the full 60-note gold-standard evaluation set."

---

## SECTION 3 — THE AGENTIC PIPELINE (1:10 – 1:40)

**🖥️ ON SCREEN:** Pipeline diagram slide (~10s) showing the 6-agent flow → then cut to live Streamlit app. Click pre-selected demo case. Show extraction result, evidence highlights, syndrome tag, agent trace. Use cursor to circle key outputs.

> "The pipeline isn't a single model call — it's an agentic workflow with multiple specialised steps.
>
> MedGemma extracts the structured encounter. Then a deterministic Evidence Grounder checks that every claim has a verbatim quote from the original note — if it doesn't, the claim is downgraded. Then MedGemma runs a self-consistency hallucination check — it re-reads its own evidence quotes and flags contradictions. For example, if it claims 'rash: yes' but the evidence quote says 'no rash observed', it catches that.
>
> After that, syndrome tagging classifies the encounter — respiratory fever, acute watery diarrhoea, or other. And a checklist generator identifies what clinical information is still missing.
>
> This multi-agent design is the safety architecture. No single model output reaches the surveillance layer without verification."

**🖥️ ON SCREEN:** Streamlit app — demo case already processed (no waiting for inference on camera). Slowly scroll through the extraction output.

> "Here's this running live. I'll process a severe pneumonia case — an 11-month-old with fever, cough, and chest indrawing. Watch how the pipeline identifies three danger signs and recommends immediate referral."

---

## SECTION 4 — OUTBREAK DETECTION (1:40 – 2:20)

**🖥️ ON SCREEN:** Switch to District Dashboard tab. Let it breathe — this is the strongest visual. Slowly scroll through trend lines. Use cursor to circle the spike at week 7. Then scroll to SITREP.

> "But the real impact isn't processing one note — it's what happens when you aggregate hundreds of structured encounters across a district.
>
> The dashboard tracks syndrome counts by location and epi-week. Here at Kibera Health Post, respiratory fever runs at a baseline of 3 to 6 cases for weeks 1 through 6. Then week 7 — a spike to 18 cases, 4.3 times baseline. The system triggers an anomaly alert automatically.
>
> For this demo I'm using z-score anomaly detection against a rolling baseline, which is the same method used in DHIS2. In production, more sophisticated approaches like CUSUM or the Farrington algorithm would give higher specificity.
>
> The system also generates a SITREP — a situation report explaining what the syndrome covers, what could be causing it, and what actions to take. This goes directly to the district health officer."

**🖥️ ON SCREEN:** Scroll down to the SITREP section. Pause briefly so judges can read the top line.

> "This takes outbreak detection from *weeks of manual tally review* to *same-day automated alerts.*"

---

## SECTION 5 — FEASIBILITY AND CLOSE (2:20 – 2:50)

**🖥️ ON SCREEN:** Closing slide — key numbers (bfloat16, ~8GB VRAM, ~26s/note, 85% accuracy), deployment plan summary, GitHub + HF Spaces links. Keep it clean.

> "The demo runs on Hugging Face Spaces with a T4 GPU. MedGemma loads in bfloat16 precision — about 8 gigabytes of VRAM. Each note processes in roughly 26 seconds.
>
> For real-world deployment, the model would run on-device on an Android app — CHWs get immediate clinical feedback, red-flag alerts, and treatment recommendations even without connectivity. Structured encounters sync to the district server when connectivity returns. This is the same 'process locally, aggregate centrally' pattern that DHIS2 mobile already uses.
>
> Future input modalities would include scanning handwritten registers with OCR and processing voice recordings — MedGemma handles the noisy output from both.
>
> CHW Copilot demonstrates that a small, on-device medical model — embedded in a well-designed agentic pipeline — can turn unstructured field notes into actionable surveillance intelligence. A peer-reviewed study this month in *Nature Health* confirmed that LLMs can deliver higher-quality clinical advice to CHWs than local clinicians, at a fraction of the cost. CHW Copilot takes this further — from answering individual questions to processing entire encounter notes into structured surveillance data at scale. It addresses a critical gap in global health infrastructure, and it does it with an open model that anyone can deploy."

**🖥️ ON SCREEN:** Title slide — "Thank you" + GitHub link + HF Spaces link

> "Thank you."

---

## Production Tips

1. **Pre-load everything** — app running, demo case processed, dashboard loaded. No waiting on camera.
2. **Highlight with cursor** — slowly circle key numbers/outputs as you mention them (the spike, the SITREP, the danger signs).
3. **No live typing** — everything pre-computed, you're just clicking and scrolling through results.
4. **Show the live URL** — HF Spaces URL visible in the browser bar proves it's deployed and running.
5. **Simple transitions** — fade or hard cut between slides and screen recording. Don't waste time on effects.
6. **Audio first** — record voiceover separately if possible, then screen-record visuals to match. Consistent volume > mic quality.
7. **If you fumble** — keep going. Cut it in post. One clean take per section is easier than one perfect take for the whole video.

---

## Timing Reference

| Section | Focus | Screen | Duration | Cumulative |
|---------|-------|--------|----------|------------|
| The Problem | Problem domain + impact stats | Title + stats slides | 40s | 0:40 |
| Why MedGemma | HAI-DEF model usage + comparison | Raw note ↔ JSON split view | 30s | 1:10 |
| Agentic Pipeline | Workflow + live demo | Pipeline diagram → Streamlit app | 30s | 1:40 |
| Outbreak Detection | Dashboard + anomaly detection | District Dashboard + SITREP | 40s | 2:20 |
| Feasibility & Close | Deployment + future + impact | Closing slide with metrics + links | 30s | 2:50 |

---

## Criteria Coverage

| Criterion (%) | Where in Video |
|---------------|----------------|
| Effective use of HAI-DEF models (20%) | Section 2 + 3: MedGemma vs alternatives, agentic multi-agent usage |
| Problem domain (15%) | Section 1: storytelling, mortality stats, user journey |
| Impact potential (15%) | Section 1 + 4: weeks→minutes, outbreak detection, population reach |
| Product feasibility (20%) | Section 3 + 5: live demo, bfloat16, deployment plan |
| Execution & communication (30%) | Overall: polished narrative, live app, clear structure |

## Tracks

- **Main Track**: Full submission
- **Agentic Workflow Prize**: The multi-agent pipeline (Extract → Ground → Hallucinate-check → Tag → Checklist → Validate) is a textbook agentic workflow that reimagines the CHW reporting process
