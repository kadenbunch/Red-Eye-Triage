# Model card: red-eye triage classifier and grounded guidance layer

## Model details

| | |
|---|---|
| Developers | [Author names, institutions] |
| Version | 1.0.0 (manuscript submission, September 2026) |
| Type | Image classifier (EfficientNet-B0 primary; MobileNet-V2 and ResNet-18 comparators) plus an LLM guidance layer (Gemini `gemini-3.5-flash`) constrained to a curated knowledge base |
| Input | One RGB photograph of the external eye, resized to 224 x 224 |
| Output | Probabilities for Inflammatory, Eyelid, Normal and Hemorrhage; optional structured, cited guidance text |
| License | MIT (code) |
| Paper | [Citation once published] |

## Intended use

- **Intended:** research on AI-assisted triage of external eye conditions; benchmarking; methodological development of patient-level validation and LLM grounding evaluation.
- **Out of scope:** diagnosis or triage of real patients; use by patients; use as a substitute for slit-lamp examination, visual acuity, pupil or intraocular pressure assessment; any use without clinical oversight.

## Training and evaluation data

1,304 curated external eye images (Inflammatory 277, Eyelid 323, Normal 642, Hemorrhage 62) derived from a public dataset whose images were collected from online sources. No patient identifiers, demographics, capture device or clinical examination data are available. See `data/README.md`.

## Performance

Pooled out-of-fold predictions from stratified 5-fold cross-validation within the 85% training pool (n = 1,108), EfficientNet-B0:

| Class | Support | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| Inflammatory | 235 | 0.974 | 0.945 | 0.959 | 0.998 | 0.992 |
| Eyelid | 274 | 0.975 | 0.982 | 0.978 | 0.999 | 0.997 |
| Normal | 546 | 0.996 | 0.996 | 0.996 | >0.999 | >0.999 |
| Hemorrhage | 53 | 0.828 | 0.906 | 0.865 | 0.995 | 0.945 |

Held-out test set (n = 196, evaluated once after refitting): weighted F1 0.98, macro AUROC 1.00.

## Known limitations and risks

1. **Possible optimistic bias.** Splits are at image level; different photographs of the same eye could not be reliably separated because patient identifiers are unavailable.
2. **Merged urgency levels.** The Inflammatory class combines conjunctivitis (usually routine) and anterior uveitis (vision-threatening). The model cannot distinguish them.
3. **Under-triage errors.** In cross-validation, 9 of 235 Inflammatory images were predicted as Hemorrhage, a benign category.
4. **Small minority class.** Hemorrhage has 62 images; its estimates are imprecise.
5. **Dataset shift.** Web-sourced images may not reflect the quality, lighting, framing, devices or demographic diversity of photographs taken in community settings.
6. **LLM layer.** Checked only by an automated audit on demonstration cases; no expert rating of accuracy, completeness or urgency appropriateness has been performed. LLM outputs can change with model updates.
7. **Knowledge base.** Author-written paraphrases; citations not yet independently verified.

## Ethical considerations

The images are publicly available and de-identified. Any deployment study would require prospective, patient-level, demographically diverse validation, ethics approval and regulatory review.
