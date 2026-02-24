# CHW Copilot — Turning the First Mile of Care into the First Mile of Surveillance

## The Problem: Outbreak Signals Buried in Paper Notes

Community Health Workers (CHWs) sit at the "first mile" of case finding and primary care at enormous scale. The Global Fund estimates **3.8 million+ CHWs across 98+ countries**, and their household visit notes can contain the earliest hints of outbreaks — diarrhoeal clusters, febrile rash syndromes, respiratory symptom waves. Yet these notes are unstructured, noisy, and written in shorthand: *"child 3yo fever cough rash not eating"*. They cannot be ingested by digital surveillance systems without manual transcription — a process that introduces days of delay, transcription errors, and data loss.

WHO surveillance guidance stresses early detection through both indicator-based and event-based surveillance (WHO, 2018), but real-world timeliness is often poor. In Uganda's DHIS2 weekly epidemic-prone disease reporting, national timeliness was approximately **44% in 2020 and 49% in 2021**, well below the ≥80% target (Migisha et al., 2023). Electronic reporting pilots show this bottleneck is tractable: Nigeria's eIDSR evaluation reported higher timeliness (**73% vs 43%**) and that most verified outbreak "rumours" originated from eIDSR sites (**80%**) (Abubakar et al., 2022).

The stakes of delayed detection are severe. **Diarrhoeal diseases** remain the second leading cause of death in children under five globally, killing approximately 525,000 children annually (WHO, 2024). Cholera outbreaks can overwhelm health systems within days if not detected early — the 2022–23 global resurgence affected 30+ countries, with case fatality rates exceeding 3% where response was delayed (WHO Cholera Situation Report, 2023). **Acute respiratory infections**, including pneumonia, are the single largest infectious cause of death in children, responsible for approximately 740,000 deaths annually in under-fives (UNICEF, 2023). Outbreaks of respiratory pathogens (influenza, RSV, COVID-19) require rapid detection to trigger public health measures before community transmission accelerates.

CHW Copilot addresses the missing link: transforming messy free-text notes into schema-valid, evidence-anchored structured signals that surveillance systems can aggregate within seconds rather than weeks.

---

## How MedGemma Powers the Pipeline

CHW Copilot is built as a **six-agent agentic pipeline**, where MedGemma 1.5 4B-IT serves as the core intelligence for clinical reasoning while deterministic agents enforce safety constraints.

**Agent 1 — Encounter Extractor** (MedGemma): Reads a raw CHW note and produces a structured JSON encounter with symptoms, patient demographics, severity, red flags, treatments, and referral status. Each positive symptom must be accompanied by a verbatim evidence quote copied from the original note. MedGemma's medical domain pre-training is essential here — it correctly interprets CHW shorthand that general-purpose LLMs miss: "hot body" as fever, "pulling in of chest" as chest indrawing (a WHO danger sign), "rice-water stool" as cholera-like acute watery diarrhoea.

**Agent 2 — Evidence Grounder** (deterministic): Verifies that every evidence quote genuinely appears in the source note. Claims with missing or fabricated quotes are downgraded from "yes" to "unknown", enforcing a hard constraint that prevents hallucinated findings from reaching the surveillance layer.

**Agent 3 — Hallucination Detector** (MedGemma self-consistency): Re-queries MedGemma to check whether each evidence quote actually supports the corresponding claim. This catches subtle contradictions — for example, if MedGemma extracts "rash: yes" with evidence quote "no rash observed", the evidence text exists in the note (passing Agent 2), but Agent 3 detects that the quote *contradicts* the claim.

**Agent 4 — Syndrome Tagger** (MedGemma / keyword fallback): Classifies the encounter into syndromic categories aligned with WHO IDSR reporting: respiratory fever, acute watery diarrhoea, other, or unclear. Sub-syndrome hints (pneumonia-like, malaria-like, cholera-like, measles-like, TB-like) guide clinical recommendations. A negation-aware keyword fallback achieves 95% accuracy on a 60-encounter gold standard test set.

**Agent 5 — Checklist Generator** (MedGemma / rule-based fallback): Identifies missing clinical information and generates prioritised follow-up questions — e.g., "Has the child been able to drink fluids?" for a febrile infant, or "Are other household members affected?" for AWD cases where cluster detection is critical.

**Agent 6 — Schema Validator** (deterministic): Validates the final encounter against a JSON Schema and produces a quality report with evidence coverage metrics.

**Why MedGemma over alternatives:** We implemented a full keyword-based fallback pipeline to compare. While the keyword tagger achieves 95% syndrome accuracy, it cannot: (a) extract patient demographics from free text, (b) identify severity or red flags, (c) handle negation in novel phrasing, or (d) generate contextual follow-up questions. MedGemma handles all four tasks and provides the clinical reasoning that a rule-based system cannot replicate. General-purpose LLMs frequently hallucinate clinical terms and miss CHW-specific shorthand — MedGemma's medical pre-training makes it immediately effective without fine-tuning.

---

## From Individual Encounters to Outbreak Detection

The real impact of CHW Copilot is not in processing individual notes — it is in what happens when structured encounters are **aggregated over time and geography** to reveal population-level signals.

### Symptom Cluster Detection

Once encounters are structured, the surveillance layer can track symptom combinations across a district. A single child with fever and cough is routine. But **15 children in the same ward presenting with fever, cough, and fast breathing in a single week** is a signal that demands investigation. The system aggregates syndrome-tagged encounters by location and epidemiological week, building a time series that makes these patterns visible.

The potential extends beyond the four core syndromes. Structured extraction captures "other symptoms" — headache, joint pain, night sweats, rash patterns, weight loss — which can reveal:
- **Measles outbreaks**: clusters of rash + fever + red eyes in children
- **Dengue signals**: fever + joint pain + headache in adults in endemic areas
- **TB clusters**: persistent cough + night sweats + weight loss over weeks
- **Novel or unusual presentations**: outlier symptoms that don't fit established syndromic categories, which may represent emerging threats

In a production system, MedGemma could be used to **interpret symptom cluster patterns** directly — given a summary of the week's encounters, it could reason about whether the constellation of symptoms across patients suggests a common cause, potential differential diagnoses at the population level, and recommended public health actions.

### Anomaly Detection

Detecting a spike requires a baseline and a threshold. There are several established approaches:

- **Z-score method**: Compares the current week's count against the mean and standard deviation of a rolling baseline window. Alerts fire when the count exceeds a threshold (e.g., mean + 2σ or mean + 3σ). Simple, interpretable, and widely used in DHIS2 and WHO EWAR systems.
- **Cumulative sum (CUSUM)**: Accumulates deviations from a reference value, detecting sustained shifts that z-scoring might miss. More sensitive to gradual increases.
- **Exponentially weighted moving average (EWMA)**: Gives greater weight to recent observations, adapting to seasonal trends.
- **Poisson regression / negative binomial models**: Model count data directly, accounting for overdispersion common in disease surveillance.
- **Farrington algorithm**: Used by ECDC and Public Health England. Fits a quasi-Poisson model with trend adjustment and outlier reweighting.

For this demonstration, we use **simple z-scoring** (current count vs. 4-week rolling baseline, alert threshold of 3.0) because it is transparent, requires no training data, and is the method most familiar to district health officers working with DHIS2. In a production deployment, the Farrington algorithm or CUSUM would be more appropriate for sustained surveillance.

---

## Technical Feasibility and Real-World Deployment

### What We Built for This Demo

The demonstration deployment is a **Streamlit web application hosted on Hugging Face Spaces** with a T4 GPU. MedGemma 1.5 4B-IT runs in 4-bit NF4 quantisation (~5GB VRAM), processing notes in approximately 3–5 seconds each. The app provides two views:

- **CHW Field View**: Enter or select a CHW note → see structured extraction, syndrome tag, evidence grounding, red-flag alerts, and ICCM-guided recommendations
- **District Dashboard**: Aggregated syndrome trends across locations and epi-weeks, anomaly detection with automated alerts, and generated SITREP narratives

### What a Production Deployment Would Look Like

The HF Spaces demo validates the core pipeline, but a real-world deployment would differ significantly:

**User-facing application**: An Android app (the dominant platform for CHWs in LMIC settings) with voice-to-text input, offline-capable data collection, and integration with existing Community Health Toolkit (CHT) or DHIS2 workflows. The CHW receives immediate clinical feedback — red-flag alerts, ICCM treatment recommendations — at the point of care.

**On-device vs cloud inference**: Ideally, MedGemma (or a smaller variant like Gemma 2B distilled for this task) would run on-device so CHWs can process notes without connectivity. However, **surveillance fundamentally requires aggregation** — individual encounters must flow to a central system to detect population-level signals. The architecture would follow a "process locally, aggregate centrally" pattern:

1. **On-device**: MedGemma extracts structured encounters → stored locally
2. **Sync-when-connected**: When the device regains connectivity (even briefly), structured encounters upload to the district server
3. **Server-side**: Aggregation, anomaly detection, and SITREP generation run on the accumulated data

This mirrors how DHIS2 mobile apps already work in low-connectivity settings. The key advantage is that CHWs get immediate clinical feedback even without connectivity, while surveillance aggregation happens opportunistically.

**Deployment challenges:**

| Challenge | Approach |
|-----------|----------|
| Device heterogeneity | Progressive model sizing: Gemma 2B for low-end devices, MedGemma 4B for mid-range, cloud fallback |
| Data privacy | Stateless processing, no patient data persisted beyond the encounter, differential privacy for aggregated counts |
| Integration with MOH systems | FHIR-compatible encounter output, DHIS2-compatible weekly aggregate format |
| Language diversity | MedGemma handles multilingual input; prompt localisation for Swahili, Hausa, French CHW notes |
| Sustaining accuracy | Continuous evaluation against gold-standard encounters, prompt versioning, A/B testing of extraction prompts |

---

## Impact: From Days to Hours

**If deployed across a district of 50 CHWs (each seeing ~10 patients/day):**

| Metric | Current State | With CHW Copilot |
|--------|---------------|-----------------|
| Time from encounter to structured data | Days–weeks (manual tally + submission) | Seconds (at point of care) |
| Data completeness | Partial (only tallied symptoms counted) | Full (all mentioned symptoms, red flags, treatments extracted) |
| Outbreak detection | After weekly/monthly aggregate review | Same-day anomaly alert when counts spike |
| Response initiation | Delayed by reporting lag | Automated SITREP triggers immediate investigation |

The magnitude of this gap matters. For cholera, the difference between detecting a cluster on day 2 versus day 14 can mean the difference between 50 and 5,000 cases (WHO Cholera Guidelines, 2017). For pneumonia in children, early identification of chest indrawing and inability to drink — WHO danger signs — triggers immediate referral, directly reducing the estimated 740,000 annual under-five pneumonia deaths (UNICEF, 2023).

With 3.8 million CHWs globally, even partial adoption of structured extraction could transform the timeliness and completeness of community-level surveillance in the countries where outbreak detection gaps are most critical.

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
