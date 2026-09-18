"""Step 5: citation-constrained guidance generation with Gemini + automated grounding audit.

Requires GEMINI_API_KEY in the environment. Every generation is saved as JSON
(prompt, output, audit, model name, timestamp) in outputs/llm_outputs/.

Usage:
    python scripts/05_generate_guidance.py --demo-classes            # one output per class
    python scripts/05_generate_guidance.py --test-case 0             # image -> prediction -> guidance
    python scripts/05_generate_guidance.py --print-prompt Inflammatory  # show prompt, no API call
"""
import os

from _common import apply_env, base_parser

p = base_parser(__doc__)
p.add_argument("--demo-classes", action="store_true")
p.add_argument("--test-case", type=int, help="Position in the held-out test set")
p.add_argument("--model", default="efficientnet", help="Classifier checkpoint used with --test-case")
p.add_argument("--print-prompt", metavar="CLASS")
args = p.parse_args()
apply_env(args)

from redeye_triage import config as C  # noqa: E402
from redeye_triage import llm_grounding as L  # noqa: E402

kb = L.load_knowledge_base()

if args.print_prompt:
    system, user, _ = L.build_grounded_prompt(args.print_prompt, kb, 0.90)
    print(system, "\n---\n", user)

if args.demo_classes or args.test_case is not None:
    client = L.get_gemini_client()

if args.demo_classes:
    for cls in C.CLASSES:
        rec = L.generate_grounded_instruction(client, cls, kb, class_probability=0.90)
        path = L.save_generation_record(rec, name=f"demo_{cls}")
        print("=" * 70, f"\n{cls}\n", "=" * 70, "\n", rec["output_text"])
        print(f"[audit] pass={rec['grounding_audit']['grounding_pass']} "
              f"fabricated={rec['grounding_audit']['n_fabricated_refs']} "
              f"uncited={rec['grounding_audit']['n_uncited_clinical_lines']} -> {path}")

if args.test_case is not None:
    import torch
    import torch.nn.functional as F
    from redeye_triage.data import EyeDiseaseDataset, get_transforms, load_dataset, make_or_load_split
    from redeye_triage.models import load_checkpoint

    data, labels, manifest = load_dataset()
    _, idx_test = make_or_load_split(labels, manifest)
    _, eval_tf = get_transforms()
    ds = EyeDiseaseDataset(data[idx_test], labels[idx_test], eval_tf)
    model = load_checkpoint(args.model, os.path.join(C.CKPT_DIR, f"{args.model}_heldout_refit.pth"))
    img_t, true_label = ds[args.test_case]
    with torch.no_grad():
        probs = F.softmax(model(img_t.unsqueeze(0).to(C.get_device())), dim=1).cpu().numpy().ravel()
    pred = int(probs.argmax())
    rec = L.generate_grounded_instruction(client, C.CLASSES[pred], kb, class_probability=float(probs[pred]))
    rec.update({"test_case": args.test_case, "image_path": manifest.iloc[int(idx_test[args.test_case])]["path"],
                "true_class": C.CLASSES[true_label], "class_probabilities": dict(zip(C.CLASSES, probs.tolist()))})
    path = L.save_generation_record(rec, name=f"test_case_{args.test_case}")
    print(rec["output_text"])
    print(f"[audit] pass={rec['grounding_audit']['grounding_pass']} -> {path}")
