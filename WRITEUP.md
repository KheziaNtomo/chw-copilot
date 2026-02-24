### Project name

Community Health Worker Copilot — Turning the First Mile of Care into the First Mile of Surveillance

### Your team

Khezia Asamoah, Research Assistant in Computational Epidemiology, complete pipeline (with AI support)

### Problem statement

Community Health Workers (CHWs) sit at the "first mile" of case finding and primary care at enormous scale. The Global Fund estimates **3.8 million+ CHWs across 98+ countries**, and their household visit notes can contain the earliest hints of outbreaks — diarrhoeal clusters, febrile rash syndromes, respiratory symptom waves. Yet these notes are unstructured, noisy, and written in shorthand: *"child 3yo fever cough rash not eating"*. They cannot be ingested by digital surveillance systems without manual transcription — a process that introduces days of delay, transcription errors, and data loss.

WHO surveillance guidance stresses early detection through indicator-based and event-based surveillance (WHO, 2018), but real-world timeliness is often poor. In Uganda's DHIS2 weekly epidemic-prone disease reporting, national timeliness was approximately **44% in 2020 and 49% in 2021**, well below the ≥80% target (Migisha et al., 2023). Electronic reporting pilots show this bottleneck is tractable: Nigeria's eIDSR evaluation reported higher timeliness (**73% vs 43%**) and that most verified outbreak "rumours" originated from eIDSR sites (**80%**) (Abubakar et al., 2022).

The stakes of delayed detection are severe. **Diarrhoeal diseases** remain the second leading cause of death in children under five globally, killing approximately 525,000 children annually (WHO, 2024). Cholera outbreaks can overwhelm health systems within days — the 2022–23 global resurgence affected 30+ countries, with case fatality rates exceeding 3% where response was delayed (WHO, 2023). **Acute respiratory infections**, including pneumonia, are the single largest infectious cause of death in children, responsible for approximately 740,000 deaths annually in under-fives (UNICEF, 2023). For cholera, the difference between detecting a cluster on day 2 versus day 14 can mean the difference between 50 and 5,000 cases (WHO, 2017).

CHW Copilot addresses the missing link: transforming messy free-text notes into schema-valid, evidence-anchored structured signals that surveillance systems can aggregate in near real-time. The core value proposition is speed — a CHW note that currently takes days or weeks to reach a surveillance system (through manual tallying, paper submission, and data entry) can be processed by MedGemma in approximately **one minute** into a structured, syndrome-tagged, evidence-grounded encounter ready for aggregation.

| Metric | Current State | With CHW Copilot |
|--------|---------------|-----------------|
| Note to structured data | Days–weeks (manual tally, paper submission, data entry) | ~1 minute (MedGemma extraction at point of care) |
| Data completeness | Partial — only pre-defined tally categories counted | Complete — all symptoms, severities, red flags, onset duration, treatments extracted |
| Outbreak detection | After weekly or monthly aggregate review | Same-day anomaly alert when syndrome counts spike above baseline |
| Response initiation | Delayed by reporting and aggregation lag | Automated SITREP triggers immediate investigation |

The potential extends beyond the four core syndromes currently tracked. Structured extraction captures other symptoms — headache, joint pain, night sweats, rash patterns, weight loss — that can reveal measles outbreaks (rash + fever + red eyes clusters), dengue signals (fever + joint pain in endemic areas), TB clusters (persistent cough + night sweats + weight loss), and novel presentations that may represent emerging threats. When notes record illness onset (e.g., "fever started 3 days ago"), the extracted onset duration enables temporal epidemiological analysis — clustered onset windows suggest point-source exposures rather than ongoing transmission.

Recent evidence strengthens the case for this approach. Rutunda et al. (2026), published in *Nature Health*, evaluated five LLMs against local clinicians using 5,609 real clinical questions from 101 CHWs across four Rwandan districts. All LLMs **significantly outperformed local clinicians** (P < 0.001) across all 11 evaluation metrics, while costing **over 500 times less** per response ($0.0035 vs $5.43 for a GP). The authors note that Rwanda alone has ~60,000 CHWs — even one question per day would cost $13 million per year for human quality assurance. The study concludes that "LLM-based clinical decision support tools" show "great promise" for frontline healthcare in low-resource settings — exactly the gap CHW Copilot fills.

With 3.8 million CHWs globally, even partial adoption of automated structured extraction could transform the timeliness of community-level surveillance in the countries where outbreak detection gaps are most critical.

### Overall solution

I built CHW Copilot as an agentic pipeline where **MedGemma 4B-IT** serves as the core intelligence for clinical reasoning, with deterministic safety layers enforcing evidence quality.

**Encounter Extraction** (MedGemma): Reads raw CHW notes and produces structured JSON encounters with symptoms, patient demographics, severity, red flags, onset duration, treatments, and referral status. Each positive symptom must be accompanied by a verbatim evidence quote copied from the original note. MedGemma's medical domain pre-training is essential — it correctly interprets CHW shorthand that general-purpose LLMs miss: "hot body" as fever, "pulling in of chest" as chest indrawing (a WHO danger sign), "rice-water stool" as cholera-like acute watery diarrhoea. The extraction prompt uses a one-shot approach (research-backed best practice for MedGemma) with a complete worked example.

**Evidence Grounding** (deterministic): Verifies that every evidence quote genuinely appears verbatim in the source note. Claims with missing or fabricated quotes are downgraded, preventing hallucinated findings from reaching the surveillance layer.

**Hallucination Detection** (MedGemma self-consistency): Re-queries MedGemma to check whether each evidence quote actually supports the corresponding claim. This catches subtle contradictions — if MedGemma extracts "rash: yes" with evidence quote "no rash observed", the evidence text exists in the note (passing the grounder), but the self-consistency check detects that the quote *contradicts* the claim.

**Syndrome Tagging** (MedGemma / keyword fallback): Classifies encounters into syndromic categories aligned with WHO IDSR reporting: respiratory fever, acute watery diarrhoea, other, or unclear. Sub-syndrome hints (pneumonia-like, malaria-like, cholera-like, measles-like, TB-like) guide clinical recommendations. A negation-aware keyword fallback achieves 95% accuracy on a 60-encounter gold-standard test set.

**Checklist Generation** (MedGemma / rule-based): Identifies missing clinical information and generates prioritised follow-up questions — e.g., "Has the child been able to drink?" for a febrile infant, or "Are other household members affected?" for AWD cases where cluster detection is critical.

**Schema Validation** (deterministic): Validates the final encounter against a JSON Schema and produces a quality report with evidence coverage metrics.

**Surveillance Aggregation**: Syndrome-tagged encounters aggregate by location and epidemiological week, feeding an anomaly detection layer and automated SITREP generation. In a production system, MedGemma could interpret symptom cluster patterns across patients — reasoning about whether a week's constellation of symptoms suggests a common cause and recommending specific public health actions.

**Why MedGemma over alternatives:** I implemented a full keyword-based fallback to compare. While the keyword tagger achieves 95% syndrome accuracy, it cannot extract patient demographics from free text, identify severity or red flags, capture illness onset duration, or generate contextual follow-up questions. General-purpose LLMs frequently hallucinate clinical terms and miss CHW-specific shorthand — MedGemma's medical pre-training makes it immediately effective without fine-tuning.

### Technical details

**Architecture**: The pipeline has two complementary implementations. The **Kaggle notebook** processes batches of CHW notes through a single MedGemma call per note (structured extraction) plus a negation-aware keyword tagger (syndrome classification), feeding into surveillance aggregation with z-score anomaly detection. The **Streamlit application** runs a richer interactive chain with evidence grounding, hallucination detection, checklist generation, and schema validation for single-note processing.

**Model performance:**

| Metric | Value |
|--------|-------|
| Syndrome tag accuracy (keyword fallback) | 95.0% (57/60 gold-standard encounters) |
| Processing time per note | ~1 minute (T4 GPU, bfloat16 precision) |
| Model | MedGemma 4B-IT, bfloat16 (~8GB VRAM) |

**Anomaly detection**: Several approaches exist in public health surveillance — z-scoring, CUSUM (cumulative sum for sustained shifts), EWMA (exponentially weighted average for seasonal adaptation), Poisson/negative binomial regression, and the Farrington algorithm (used by ECDC with quasi-Poisson trend adjustment). For this demonstration, I used simple z-scoring (count vs. 4-week rolling baseline, threshold 3.0) because it is transparent and familiar to district health officers working with DHIS2. A production system would use Farrington or CUSUM for higher specificity.

**Demonstration deployment**: Streamlit web application on Hugging Face Spaces with a T4-small GPU. MedGemma loads in bfloat16 precision. The app auto-detects whether a GPU and model are available — when deployed on HF Spaces, it runs MedGemma live; without GPU, it falls back to deterministic rules with pre-computed demo cases.

**Real-world deployment considerations**: The HF Spaces demo validates the core pipeline. A production deployment would use an Android app with integration into CHT or DHIS2 workflows. Input modalities would expand beyond typed text to include **scanned handwritten notes** (on-device OCR via Google ML Kit) and **voice recordings** (speech-to-text via Whisper or Google Speech API with local language support). MedGemma's tolerance for noisy, informal text makes it well-suited for OCR and speech-to-text outputs.

On-device MedGemma inference would give CHWs immediate clinical feedback (red flags, ICCM recommendations) without connectivity. However, surveillance fundamentally requires aggregation — individual encounters must flow to a central system. The proposed architecture follows a "process locally, aggregate centrally" pattern: structured encounters are extracted on-device, stored locally, and sync to the district server when connectivity is available. This mirrors how DHIS2 mobile apps already work in low-connectivity settings.

| Deployment Challenge | Approach |
|---------------------|----------|
| Device heterogeneity | Progressive model sizing: distilled models for low-end devices, MedGemma 4B for mid-range, cloud fallback |
| Data privacy | Stateless processing, no patient data persisted; differential privacy for aggregated counts |
| MOH system integration | FHIR-compatible encounter output, DHIS2-compatible weekly aggregate format |
| Language diversity | MedGemma handles multilingual input; prompt localisation for Swahili, Hausa, French CHW notes |
| Sustaining accuracy | Continuous evaluation against gold-standard encounters, prompt versioning |

**References:**
WHO (2018) Early detection, assessment and response to acute public health events; Migisha et al. (2023) Timeliness of Weekly Disease Surveillance Reporting, Uganda, Emerg Infect Dis 29(Suppl 1); Abubakar et al. (2022) Evaluation of Nigeria's eIDSR, BMC Public Health 22; Global Fund (2023) Community Health Workers: Evidence and Guidance; WHO (2024) Diarrhoeal disease fact sheet; UNICEF (2023) Pneumonia in children; WHO (2023) Multi-country cholera situation reports; WHO (2017) Ending cholera: A global roadmap to 2030; **Rutunda et al. (2026) Large language models for frontline healthcare support in low-resource settings, Nature Health 1, 191–197.**
