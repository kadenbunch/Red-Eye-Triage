# Changes from the original Colab notebook

The analysis was first run in a single Colab notebook (`Red_Eye_Disease_Triage_Classifier.ipynb`). This repository reorganizes that code into modules and scripts. The training, evaluation, metric, threshold and calibration code is unchanged in behavior: the same seeds, split procedure, transforms, architectures, hyperparameters, and order of random-number use. `metrics.threshold_analysis`, `perclass_metrics`, `bootstrap_ci` and `expected_calibration_error` were checked against the original functions on synthetic data and give identical outputs.

## Bugs fixed

| Area | Original behavior | Fix |
|---|---|---|
| Knowledge base | Defined as `CKNOWLEDGE_BASE` but referenced as `KNOWLEDGE_BASE`, which raises `NameError` in a fresh session | Knowledge base moved to `knowledge_base/knowledge_base.json` and loaded by `load_knowledge_base()` |
| Prompt builder | `format_kb_for_prompt` read `id`, `tags`, `statement`, `citation` fields, but the condition-level knowledge base has none of them (`KeyError`) | Condition entries are flattened into citable items with stable IDs (`UVE-01`, `UVE-RED-01`, ...) |
| Grounding audit | Citation regex required IDs like `UVE-01`; condition IDs such as `anterior_uveitis` could never match, so every clinical line would be flagged as uncited | IDs now follow a single validated pattern shared by the prompt, the knowledge base loader and the audit |
| Demo prompt | `build_grounded_prompt("Uveitis", ...)`: "Uveitis" is not a model class | Unknown classes raise an error; demos use the four model classes |
| Duplicate / leakage check | Ran before the split file existed (`FileNotFoundError`); rebuilt the path list including `.DS_Store` files, offsetting every index by one relative to `load_dataset()` | Runs after training and maps files to partitions through `split_manifest.csv` |
| Figure 2 colorbar | Tick labels rendered as `0, 0, 0, 0, 1` | Ticks formatted to one decimal place |
| Nearest-neighbor labels | Displayed as "L2", but FAISS `IndexFlatL2` returns squared L2 distances | Labeled "squared L2" |
| API key | Read from a Colab secret named `GEMENI_API_KEY` | Reads `GEMINI_API_KEY` |

## Additions for reproducibility

- Out-of-fold and held-out predictions are saved (`*_oof_predictions.npz`, `*_heldout_predictions.npz`), so every reported metric can be recomputed without retraining.
- `split_manifest.csv` records the partition of every image by file name.
- Every LLM generation is saved with its prompt, output, audit result, model name and UTC timestamp.
- Command-line scripts, unit tests, and documentation (README, data README, model card).
- Figures are drawn with a colour-vision-deficiency-safe palette.

## Not verified here

The refactored training pipeline could not be run end to end during reorganization because the image data and a GPU were not available. Re-run `scripts/01_train_evaluate.py` with the saved `split_indices.npz` and confirm that Tables 1-3 are reproduced before tagging a release.
