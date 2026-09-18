# red-eye-triage

Code accompanying the manuscript

> **Integrating Deep Learning and Large Language Models for Community-Based Triage of Red Eye Diseases.**
> [Author list]. *Translational Vision Science & Technology* (submitted, 2026).

The pipeline (1) fine-tunes three ImageNet-pretrained convolutional networks (EfficientNet-B0, MobileNet-V2, ResNet-18) to classify external eye photographs into four categories (Inflammatory, Eyelid, Normal, Hemorrhage); (2) evaluates them with stratified 5-fold cross-validation, bootstrap 95% confidence intervals, clinically motivated operating points and calibration, plus a single held-out test evaluation; (3) explains predictions with Grad-CAM++, Integrated Gradients and nearest-neighbour retrieval; and (4) passes the predicted class to Gemini, which is constrained to an author-curated, cited knowledge base and audited automatically for unsupported statements.

> [!WARNING]
> **Research software, not for clinical use.** This is an exploratory proof of concept. It is not a medical device, has not been prospectively or externally validated, and its LLM output has not undergone expert clinical review. See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Repository structure

```
red-eye-triage/
├── src/redeye_triage/          # all analysis code (single source of truth)
│   ├── config.py               # paths, classes, hyperparameters, seeding
│   ├── data.py                 # loading, transforms/augmentation, fixed train/test split
│   ├── models.py               # model factory, class weights, training, inference
│   ├── metrics.py              # metrics, bootstrap CIs, operating points, calibration
│   ├── plots.py                # ROC/PR, confusion, calibration, comparison figures
│   ├── pipeline.py             # cross-validation, held-out refit, table assembly
│   ├── duplicates.py           # exact/near-duplicate audit and leakage check
│   ├── explainability.py       # Grad-CAM++, Integrated Gradients, FAISS retrieval
│   └── llm_grounding.py        # knowledge-base retrieval, constrained prompt, Gemini call, audit
├── scripts/                    # command-line entry points (steps 01-05)
├── notebooks/
│   └── red_eye_triage_pipeline.ipynb   # Google Colab driver for the same steps
├── knowledge_base/
│   ├── knowledge_base.json     # LLM knowledge base (Supplementary Table S3)
│   └── README.md
├── data/README.md              # dataset source, curation, expected folder layout
├── results/README.md           # released result files (tables, predictions, split)
├── docs/
│   ├── MODEL_CARD.md           # intended use, limitations, performance summary
│   └── CHANGES_FROM_ORIGINAL_NOTEBOOK.md  # bug fixes and differences from the Colab notebook
├── tests/                      # unit and smoke tests
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-github-username>/red-eye-triage.git
cd red-eye-triage
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended for training. The reported analyses were run on Google Colab (Python 3.12, GPU runtime). To run in Colab, open `notebooks/red_eye_triage_pipeline.ipynb`, set `REPO_URL`, and run the cells in order.

## Data

The images are **not** redistributed in this repository by default. Follow [`data/README.md`](data/README.md) to obtain the source dataset and reproduce the curated four-class set. The code expects:

```
data/images/
├── Inflammatory/   (277 images)
├── Eyelid/         (323 images)
├── Normal/         (642 images)
└── Hemorrhage/     (62 images)
```

## Reproducing the analyses

Every script accepts `--data-dir` and `--output-dir`; outputs are written to `outputs/` by default.

| Step | Command | Produces | Manuscript |
|---|---|---|---|
| 1 | `python scripts/01_train_evaluate.py` | split, per-model results, OOF and held-out predictions, checkpoints, CSV tables, ROC/PR/confusion/calibration figures | Tables 1-3, Supplementary Tables S1-S2, Supplementary Figures S1-S2 |
| 2 | `python scripts/02_duplicate_audit.py` | exact and near-duplicate lists, train/test leakage report | Methods, Limitations |
| 3 | `python scripts/03_explainability.py --model efficientnet` | Grad-CAM++ grid and multi-method case panels | Figure 2 |
| 4 | `python scripts/04_comparison_figures.py` | bootstrap comparison figures | - |
| 5 | `export GEMINI_API_KEY=...` then `python scripts/05_generate_guidance.py --demo-classes --test-case 0` | grounded guidance + audit JSON records | Figure 3 |

Quick end-to-end check (tiny settings, not for results): `python scripts/01_train_evaluate.py --fast --output-dir outputs_fast`.

Run the tests with `pytest tests/` (the model smoke test is skipped automatically if PyTorch is not installed).

### Output files

| File | Contents |
|---|---|
| `outputs/data_manifest.csv` | every loaded image: index, relative path, class, label |
| `outputs/checkpoints/split_indices.npz`, `split_manifest.csv` | the fixed 85/15 train-pool / held-out split |
| `outputs/per_model_results/<model>_result.json` | hyperparameters, CV metrics with CIs, per-class metrics, operating points, fold metrics, held-out metrics, bootstrap replicates |
| `outputs/per_model_results/<model>_oof_predictions.npz` | pooled out-of-fold probabilities (all CV metrics can be recomputed from these) |
| `outputs/per_model_results/<model>_heldout_predictions.npz` | held-out test probabilities |
| `outputs/table1_overall_metrics_cv.csv` | Table 1 |
| `outputs/table2_tableS1_perclass_metrics_cv.csv` | Table 2 and Supplementary Table S1 |
| `outputs/table3_tableS2_threshold_analysis_cv.csv` | Table 3 and Supplementary Table S2 |
| `outputs/llm_outputs/*.json` | prompt, LLM output, grounding audit, model name, UTC timestamp |

## Methods summary

| Item | Setting |
|---|---|
| Classes | Inflammatory (conjunctivitis + anterior uveitis), Eyelid, Normal, Hemorrhage |
| Input | 224 x 224 RGB, ImageNet normalization |
| Augmentation (training only) | horizontal flip (p = 0.5), rotation +/-15 deg, colour jitter (brightness 0.15, contrast 0.15, saturation 0.10) |
| Class imbalance | inverse-frequency class-weighted cross-entropy (no SMOTE) |
| Architectures | EfficientNet-B0, MobileNet-V2, ResNet-18; torchvision `IMAGENET1K_V1` weights; all layers fine-tuned |
| Optimization | Adam, learning rate 1e-4, batch size 32, 15 epochs |
| Partitioning | stratified 85/15 train-pool / held-out split; stratified 5-fold CV inside the train pool; seed 42 |
| Primary estimates | pooled out-of-fold predictions; percentile bootstrap 95% CIs (2,000 resamples) |
| Operating points (per class, one-vs-rest) | max-F1; highest threshold with sensitivity >= 0.95; lowest threshold with specificity >= 0.95 |
| Calibration | expected calibration error (10 bins), multiclass Brier score |
| Held-out test | model refit on the full train pool, evaluated once |
| Explainability | Grad-CAM++ (last conv block); Integrated Gradients (all-zero baseline in normalized space, i.e. the ImageNet mean image; 50 steps, convergence delta reported); FAISS exact search over training-partition embeddings (squared L2) |
| LLM | `gemini-3.5-flash`, temperature 0.2, class-based deterministic retrieval from `knowledge_base.json`, mandatory item-ID citations, automated audit |

### Determinism

Python, NumPy, PyTorch and CUDA seeds are fixed per fold and cuDNN is set to deterministic mode. Bit-identical results across different GPU models, driver versions or library versions are not guaranteed; small numeric differences are expected.

## Results (cross-validated, pooled out-of-fold, 95% bootstrap CI)

| Model | Accuracy | Weighted F1 | Macro AUROC | Macro AUPRC | ECE | Brier |
|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.977 (0.969-0.986) | 0.978 (0.969-0.986) | 0.998 (0.996-0.999) | 0.984 (0.971-0.994) | 0.008 | 0.036 |
| MobileNet-V2 | 0.977 (0.968-0.985) | 0.976 (0.967-0.985) | 0.998 (0.998-0.999) | 0.986 (0.975-0.993) | 0.008 | 0.039 |
| ResNet-18 | 0.970 (0.959-0.980) | 0.971 (0.961-0.980) | 0.994 (0.987-0.998) | 0.969 (0.948-0.986) | 0.011 | 0.046 |

These figures are image-level estimates from a web-sourced dataset and may be optimistic; see the Limitations in the manuscript and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Model weights

Trained checkpoints (`<model>_heldout_refit.pth`) are attached to the [GitHub Release](https://github.com/<your-github-username>/red-eye-triage/releases) for this version rather than committed to the repository. Place them in `outputs/checkpoints/` to run steps 3 and 5 without retraining.

## Use of generative AI

Gemini (`gemini-3.5-flash`, Google) is a component of the method (step 5). Every generation is logged with the model name and timestamp.

## Citation

If you use this code, please cite the article (see [`CITATION.cff`](CITATION.cff)). A link to the published article will be added here on publication.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The source images are not covered by this license; see [`data/README.md`](data/README.md).

## Contact

[Kaden Bunch, kaden_bunch@brown.edu]
