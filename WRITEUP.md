# CHW Copilot — Turning the First Mile of Care into the First Mile of Surveillance

## The Problem: Outbreak Signals Buried in Paper Notes

Community Health Workers (CHWs) sit at the "first mile" of case finding and primary care at enormous scale. The Global Fund estimates **3.8 million+ CHWs across 98+ countries**, and their household visit notes can contain the earliest hints of outbreaks — diarrhoeal clusters, febrile rash syndromes, respiratory symptom waves. Yet these notes are unstructured, noisy, and written in shorthand: *"child 3yo fever cough rash not eating"*. They cannot be ingested by digital surveillance systems without manual transcription — a process that introduces days of delay, transcription errors, and data loss.

WHO surveillance guidance stresses early detection through both indicator-based and event-based surveillance (WHO, 2018), but real-world timeliness is often poor. In Uganda's DHIS2 weekly epidemic-prone disease reporting, national timeliness was approximately **44% in 2020 and 49% in 2021**, well below the ≥80% target (Migisha et al., 2023). Electronic reporting pilots show this bottleneck is tractable: Nigeria's eIDSR evaluation reported higher timeliness (**73% vs 43%**) and that most verified outbreak "rumours" originated from eIDSR sites (**80%**) (Abubakar et al., 2022).

The stakes of delayed detection are severe. **Diarrhoeal diseases** remain the second leading cause of death in children under five globally, killing approximately 525,000 children annually (WHO, 2024). Cholera outbreaks can overwhelm health systems within days if not detected early — the 2022–23 global resurgence affected 30+ countries, with case fatality rates exceeding 3% where response was delayed (WHO Cholera Situation Report, 2023). **Acute respiratory infections**, including pneumonia, are the single largest infectious cause of death in children, responsible for approximately 740,000 deaths annually in under-fives (UNICEF, 2023). Outbreaks of respiratory pathogens (influenza, RSV, COVID-19) require rapid detection to trigger public health measures before community transmission accelerates.

CHW Copilot addresses this gap by transforming messy free-text notes into schema-valid, evidence-anchored structured signals that surveillance systems can aggregate in near real-time.

---

## How MedGemma Powers the Pipeline

I built CHW Copilot as an **agentic pipeline** where MedGemma 1.5 4B-IT serves as the core intelligence for clinical reasoning, with deterministic safety layers enforcing evidence quality.

The pipeline has two complementary implementations:

**Kaggle notebook pipeline** — optimised for batch processing of CHW notes at scale. A single MedGemma call per note extracts structured encounters (symptoms with verbatim evidence quotes, patient demographics, severity, red flags, onset duration, treatments, and referral status). A negation-aware keyword tagger then classifies syndromes. This pipeline processes 60 gold-standard encounters and feeds into surveillance aggregation.

**Streamlit application pipeline** — designed for interactive, single-note processing with a richer agent chain:

1. **Encounter Extractor** (MedGemma): Reads the raw note and produces a structured JSON encounter. MedGemma's medical domain pre-training is essential — it correctly interprets CHW shorthand that general-purpose LLMs miss: "hot body" as fever, "pulling in of chest" as chest indrawing (a WHO danger sign), "rice-water stool" as cholera-like acute watery diarrhoea.

2. **Evidence Grounder** (deterministic): Verifies that every evidence quote genuinely appears verbatim in the source note. Claims with missing or fabricated quotes are downgraded from "yes" to "unknown", preventing hallucinated findings from reaching the surveillance layer.

3. **Hallucination Detector** (MedGemma self-consistency): Re-queries MedGemma to check whether each evidence quote actually supports the corresponding claim. This catches subtle contradictions — for example, if MedGemma extracts "rash: yes" with evidence quote "no rash observed", the evidence text exists in the note (passing Agent 2), but this agent detects that the quote *contradicts* the claim.

4. **Syndrome Tagger** (MedGemma / keyword fallback): Classifies the encounter into syndromic categories aligned with WHO IDSR reporting: respiratory fever, acute watery diarrhoea, other, or unclear. Sub-syndrome hints (pneumonia-like, malaria-like, cholera-like, measles-like, TB-like) guide clinical recommendations. The negation-aware keyword fallback achieves 95% accuracy on a 60-encounter gold-standard test set.

5. **Checklist Generator** (MedGemma / rule-based fallback): Identifies missing clinical information and generates prioritised follow-up questions — e.g., "Has the child been able to drink fluids?" for a febrile infant, or "Are other household members affected?" for AWD cases where cluster detection is critical.

6. **Schema Validator** (deterministic): Validates the final encounter against a JSON Schema and produces a quality report with evidence coverage metrics.

**Why MedGemma over alternatives:** I implemented a full keyword-based fallback pipeline to compare. While the keyword tagger achieves 95% syndrome accuracy, it cannot: (a) extract patient demographics from free text, (b) identify severity or red flags, (c) capture illness onset duration (e.g., "fever 3 days ago" → onset_days: 3), or (d) generate contextual follow-up questions. General-purpose LLMs frequently hallucinate clinical terms and miss CHW-specific shorthand — MedGemma's medical pre-training makes it immediately effective without fine-tuning.

---

## From Individual Encounters to Outbreak Detection

The real impact of CHW Copilot is not in processing individual notes — it is in what happens when structured encounters are **aggregated over time and geography** to reveal population-level signals.

### Symptom Cluster Detection

Once encounters are structured, the surveillance layer tracks symptom combinations across a district. A single child with fever and cough is routine. But **15 children in the same ward presenting with fever, cough, and fast breathing in a single week** is a signal that demands investigation. The system aggregates syndrome-tagged encounters by location and epidemiological week, building time series that make these patterns visible on the district dashboard.

Structured extraction also captures **onset duration** — when a note says "illness started 3 days ago", MedGemma extracts this as `onset_days: 3`. This enables temporal epidemiological analysis: if multiple patients in an area report onset within a narrow window, it suggests a point-source exposure (e.g., contaminated water) rather than ongoing transmission.

The potential extends well beyond the four core syndromes currently tracked. Structured extraction captures "other symptoms" — headache, joint pain, night sweats, rash patterns, weight loss — which can reveal:
- **Measles outbreaks**: clusters of rash + fever + red eyes in children
- **Dengue signals**: fever + joint pain + headache in adults in endemic areas
- **TB clusters**: persistent cough + night sweats + weight loss over weeks
- **Novel or unusual presentations**: outlier symptoms that do not fit established syndromic categories, which may represent emerging threats or environmental exposures

In a production system, MedGemma could be used to **interpret symptom cluster patterns** directly — given a summary of the week's encounters across a district, it could reason about whether the constellation of symptoms across patients suggests a common cause, offer differential diagnoses at the population level, and recommend specific public health actions.

### Anomaly Detection

Detecting a surveillance spike requires a baseline and a threshold. There are several established approaches used in public health surveillance:

- **Z-score method**: Compares the current week's count against the mean and standard deviation of a rolling baseline window. Alerts fire when the count exceeds a defined threshold (e.g., mean + 2σ or mean + 3σ). Simple, interpretable, and widely used in DHIS2 and WHO EWAR systems.
- **Cumulative sum (CUSUM)**: Accumulates deviations from a reference value over time, making it more sensitive to sustained shifts that z-scoring might miss.
- **Exponentially weighted moving average (EWMA)**: Gives greater weight to recent observations, adapting to seasonal trends more smoothly.
- **Poisson regression / negative binomial models**: Model count data directly, accounting for overdispersion common in disease surveillance.
- **Farrington algorithm**: Used by ECDC and Public Health England. Fits a quasi-Poisson model with trend adjustment and outlier reweighting for higher specificity.

For this demonstration, I used **simple z-scoring** (current count vs. 4-week rolling baseline, alert threshold of 3.0) because it is transparent, requires no training data, and is the method most familiar to district health officers working with DHIS2. In a production deployment, the Farrington algorithm or CUSUM would be more appropriate for sustained, high-specificity surveillance.

---

## Technical Feasibility and Real-World Deployment

### What I Built for This Demonstration

The demonstration deployment is a **Streamlit web application hosted on Hugging Face Spaces** with a T4 GPU. MedGemma 1.5 4B-IT runs in 4-bit NF4 quantisation (~5GB VRAM), processing each note in approximately **one minute**. The Kaggle notebook processes a batch of 60 gold-standard CHW notes, demonstrating extraction, syndrome tagging, surveillance aggregation, and anomaly detection at scale.

The app provides two views:

- **CHW Field View**: Enter or select a CHW note → see structured extraction, syndrome tag, evidence grounding, red-flag alerts, and ICCM-guided clinical recommendations
- **District Dashboard**: Aggregated syndrome trends across locations and epi-weeks, with automated anomaly alerts and generated SITREP narratives

### What a Production Deployment Would Require

The HF Spaces demo validates the core pipeline, but a real-world deployment would differ significantly.

**User-facing application**: An Android app (the dominant platform for CHWs in LMIC settings) with offline-capable data collection and integration with existing Community Health Toolkit (CHT) or DHIS2 workflows. The CHW would receive immediate clinical feedback — red-flag alerts, ICCM treatment recommendations — at the point of care.

**Input modalities**: In the current demonstration, notes are typed into a text box. However, many CHWs still record observations on paper registers or prefer to speak rather than type. A production system should support multiple input channels:
- **Typed text** (current) — direct entry into the app
- **Scanned handwritten notes** — using on-device OCR (e.g., Google ML Kit) to digitise paper registers, then feeding the extracted text into MedGemma for structured extraction
- **Voice recordings** — using speech-to-text (e.g., Whisper or Google Speech API, with support for local languages and accents) to transcribe spoken observations, then processing the transcript through the same pipeline

MedGemma's ability to handle noisy, informal clinical text makes it well-suited for all three modalities, since OCR and speech-to-text outputs tend to contain errors and abbreviations similar to those found in manually typed CHW notes.

**On-device inference vs aggregation**: Ideally, MedGemma (or a smaller distilled variant) would run on-device so CHWs can process notes without connectivity. However, **surveillance fundamentally requires aggregation** — individual encounters must flow to a central system to detect population-level signals. The architecture would follow a "process locally, aggregate centrally" pattern:

1. **On-device**: Structured encounters extracted locally and stored in an on-device database
2. **Sync-when-connected**: When the device regains connectivity, structured encounters upload to the district server via a lightweight REST API
3. **Server-side**: Aggregation, anomaly detection, and SITREP generation run on the accumulated data

This mirrors how DHIS2 mobile apps already work in low-connectivity settings. The key advantage is that CHWs get immediate clinical feedback even without connectivity, while surveillance aggregation happens opportunistically.

**Key deployment challenges:**

| Challenge | Approach |
|-----------|----------|
| Device heterogeneity | Progressive model sizing: smaller distilled models for low-end devices, MedGemma 4B for mid-range, cloud fallback for devices without GPU |
| Data privacy | Stateless processing with no patient data persisted beyond the encounter; differential privacy for aggregated counts |
| Integration with MOH systems | FHIR-compatible encounter output, DHIS2-compatible weekly aggregate format |
| Language diversity | MedGemma handles multilingual input; prompt localisation for Swahili, Hausa, French CHW notes |
| Sustaining accuracy | Continuous evaluation against gold-standard encounters, prompt versioning, A/B testing of extraction prompts |

---

## Impact: From Weeks to Minutes

The core value proposition is speed. A CHW note that currently takes days or weeks to reach a surveillance system — through manual tallying, paper submission, and data entry — can be processed by MedGemma in approximately **one minute** into a structured, syndrome-tagged, evidence-grounded encounter ready for aggregation.

| Metric | Current State | With CHW Copilot |
|--------|---------------|-----------------|
| Note to structured data | Days–weeks (manual tally, paper submission, data entry) | ~1 minute (MedGemma extraction at point of care) |
| Data completeness | Partial — only pre-defined tally categories counted, details lost | Complete — all mentioned symptoms, severities, red flags, onset duration, treatments extracted |
| Outbreak detection | After weekly or monthly aggregate review by district officer | Same-day anomaly alert when syndrome counts spike above baseline |
| Response initiation | Delayed by reporting and aggregation lag | Automated SITREP triggers immediate investigation and resource deployment |

For cholera, the difference between detecting a cluster on day 2 versus day 14 can mean the difference between 50 and 5,000 cases (WHO Cholera Guidelines, 2017). For pneumonia in children, early identification of chest indrawing and inability to drink — WHO danger signs — triggers immediate referral, directly contributing to reducing the estimated 740,000 annual under-five pneumonia deaths (UNICEF, 2023).

With 3.8 million CHWs globally, even partial adoption of automated structured extraction could transform the timeliness and completeness of community-level surveillance in the countries where outbreak detection gaps are most critical.

---

## References

1. WHO. (2018). *Early detection, assessment and response to acute public health events.* WHO/HSE/GCR/LYO/2014.4 Rev.1
2. Migisha, R., et al. (2023). Timeliness of Weekly Disease Surveillance Reporting, Uganda, 2020–2021. *Emerging Infectious Diseases*, 29(Suppl 1).
3. Abubakar, A., et al. (2022). Evaluation of Nigeria's electronic Integrated Disease Surveillance and Response system. *BMC Public Health*, 22.
4. Global Fund. (2023). *Community Health Workers: Evidence and Guidance.*
5. WHO. (2024). *Diarrhoeal disease fact sheet.* https://www.who.int/news-room/fact-sheets/detail/diarrhoeal-disease
6. UNICEF. (2023). *Pneumonia in children fact sheet.*
7. WHO. (2023). *Multi-country outbreak of cholera — External situation report.* Nos. 1–12.
8. WHO. (2017). *Ending cholera: A global roadmap to 2030.*

---

*CHW Copilot — Turning the first mile of care into the first mile of surveillance.*
