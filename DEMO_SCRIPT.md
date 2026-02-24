# CHW Copilot — 3-Minute Video Script

> **Format:** Screen recording (OBS/Loom) + voiceover — no face cam needed
> **Audience:** Google Research judges (MedGemma Impact Challenge)
> **Tone:** Conversational, confident, evidence-driven. Pitch, not tutorial.
> **Tracks:** Main Track + Agentic Workflow Prize
> **Timed read:** ~2:50 — leaves ~10s buffer for transitions

---

## Pre-Recording Checklist

- [ ] Streamlit app running and loaded (HF Spaces or local) — HF URL visible in browser bar
- [ ] Demo case pre-selected (severe pneumonia, 11-month-old)
- [ ] District Dashboard tab ready with Kibera Health Post visible
- [ ] Browser zoom at 150%, terminal font large
- [ ] Slides ready: title, stats, pipeline diagram, closing
- [ ] Quiet room, headset mic or external mic
- [ ] OBS/Loom set to capture browser window only (no desktop chrome)

---

## PART 1 — THE PROBLEM + WHAT IT DOES (0:00 – 0:50)

**🖥️ ON SCREEN:** Title slide (green gradient, "CHW Copilot", name, track) → fade to stats slide (3.8M CHWs, outbreak speed stat). Keep slides clean, 2–3 max.

> [thoughtful] When an outbreak starts, the first people to see it aren't doctors in hospitals — they're community health workers, going door to door in villages and urban slums. [short pause] They spot the earliest signals: fever clusters, diarrhoea, breathing problems.
>
> [sighs] But their observations are usually written as shorthand notes… [short pause] and they don't reach surveillance systems in time.
>
> [serious] That delay matters. [short pause] With fast-moving outbreaks like cholera, finding a cluster early can be the difference between a contained flare-up… and thousands of cases.

**🖥️ ON SCREEN:** Transition to split view — raw CHW note (messy shorthand) on left → clean structured JSON on right. This is the "wow" moment.

> [confident] CHW Copilot closes the gap between those raw notes and the structured data surveillance systems need.
>
> [explaining] It uses MedGemma to read a field note and turn it into a structured, evidence-grounded encounter in about a minute: [short pause] symptoms, danger signs, key details, and a syndrome label — like respiratory fever or acute watery diarrhoea.

---

## PART 2 — SAFETY + LIVE DEMO (0:50 – 1:30)

**🖥️ ON SCREEN:** Pipeline diagram slide (~8s) showing evidence grounding flow → then cut to live Streamlit app with demo case already processed.

> [serious] The key is safety. [short pause] The system doesn't just "summarise." [emphatic] Every extracted claim has to be backed by a verbatim quote from the original note. [short pause] If it can't point to the evidence, it doesn't get passed forward.

**🖥️ ON SCREEN:** Streamlit app — severe pneumonia case. Slowly scroll through extraction output. Use cursor to circle the danger signs and referral recommendation.

> [short pause] Let me show you a real example. [excited] Here's a severe pneumonia case — an eleven-month-old with fever and cough… [short pause] and the note mentions chest indrawing. [serious] The pipeline flags that as a danger sign… and recommends immediate referral.

---

## PART 3 — AGGREGATION + OUTBREAK DETECTION (1:30 – 2:15)

**🖥️ ON SCREEN:** Switch to District Dashboard. Let it breathe — this is the strongest visual. Use cursor to circle the spike at week 7, then scroll to SITREP.

> [short pause] Now zoom out. [confident] The real power is aggregation.
>
> [explaining] As hundreds of these encounters come in, the dashboard tracks syndrome counts by location and week. [short pause] When a clinic jumps from a steady baseline to a spike… [emphatic] the system triggers an alert automatically… [short pause] and generates a short situation report for the district health officer: what's increasing, where, and what actions to consider.

**🖥️ ON SCREEN:** Scroll to SITREP section. Pause briefly so judges can read the headline.

> [emphatic] This takes outbreak detection from manual tallies reviewed weeks later… to same-day automated signals.

---

## PART 4 — FEASIBILITY + CLOSE (2:15 – 2:50)

**🖥️ ON SCREEN:** Closing slide — key metrics (bfloat16, ~8GB, ~26s/note, 85% accuracy), deployment summary, GitHub + HF Spaces links.

> [thoughtful] Today this runs as a demo on a small GPU. [short pause] In deployment, it can run offline on an Android phone… give community health workers immediate clinical feedback… and sync structured encounters when connectivity returns.
>
> [confident] CHW Copilot turns messy field notes into verified, structured signals — so health systems can spot outbreaks earlier, respond faster, and prevent small clusters from becoming crises.

**🖥️ ON SCREEN:** Title slide — "Thank you" + links

---

## Production Tips

1. **Pre-load everything** — app running, demo case processed, dashboard loaded. No waiting on camera.
2. **Highlight with cursor** — slowly circle key outputs as you mention them (danger signs, the spike, the SITREP).
3. **No live typing** — everything pre-computed, just clicking and scrolling.
4. **Show the live URL** — HF Spaces URL in the browser bar proves it's deployed.
5. **Simple transitions** — fade or hard cut. Don't waste time on effects.
6. **Audio first** — record voiceover separately, then screen-record visuals to match.
7. **If you fumble** — keep going, cut in post. One clean take per section.

---

## Timing Reference

| Part | Focus | Screen | Duration | Cumulative |
|------|-------|--------|----------|------------|
| The Problem + What It Does | Problem domain, intro to solution | Title + stats slides → raw note/JSON split | 50s | 0:50 |
| Safety + Live Demo | Evidence grounding, live app | Pipeline diagram → Streamlit demo case | 40s | 1:30 |
| Aggregation + Outbreak | Dashboard, anomaly alerts, SITREP | District Dashboard + SITREP | 45s | 2:15 |
| Feasibility + Close | Deployment, impact, close | Closing slide with metrics + links | 35s | 2:50 |

---

## Criteria Coverage

| Criterion (%) | Where in Video |
|---------------|----------------|
| Effective use of HAI-DEF models (20%) | Part 1 + 2: MedGemma extraction, evidence grounding, agentic pipeline |
| Problem domain (15%) | Part 1: storytelling, outbreak speed, CHW context |
| Impact potential (15%) | Part 1 + 3: weeks→minutes, outbreak detection, population reach |
| Product feasibility (20%) | Part 2 + 4: live demo, bfloat16, offline deployment plan |
| Execution & communication (30%) | Overall: polished narrative, live app, clear structure |

## Tracks

- **Main Track**: Full submission
- **Agentic Workflow Prize**: Multi-agent pipeline (Extract → Ground → Hallucinate-check → Tag → Checklist → Validate)
