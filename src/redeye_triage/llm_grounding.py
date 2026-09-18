"""Citation-constrained LLM guidance generation and automated grounding audit.

The LLM is restricted to an author-curated knowledge base (knowledge_base/knowledge_base.json,
manuscript Supplementary Table S3):
  * items for the predicted class (plus any universal items with class "ALL") are
    retrieved deterministically - no embedding similarity, so nothing relevant is missed;
  * the model must use ONLY those items and cite an item ID after every clinical claim;
  * gaps must be stated explicitly ("Not covered by provided guidance");
  * urgency must be escalated when a red flag applies;
  * a post-hoc audit flags fabricated item IDs and clinical lines without a citation.

The audit is a first-pass automated check and does NOT replace expert clinical review.
The output of this module is research software and is not intended for clinical use.
"""

import datetime as _dt
import json
import os
import random
import re
import time

from . import config as C

ID_PATTERN = r"[A-Z]+(?:-[A-Z]+)*-\d+"  # e.g. UVE-01, UVE-RED-01, GEN-RED-01

# ------------------------------------------------------------------ #
# 1. Knowledge base
# ------------------------------------------------------------------ #
_FIELD_ORDER = [
    ("condition_summary", "01", "summary"),
    ("typical_clinical_features", "FEAT-01", "typical_features"),
    ("red_flags", "RED-01", "red_flag"),
    ("initial_management", "MGMT-01", "initial_management"),
    ("consultation_urgency", "URG-01", "consultation_urgency"),
]


def _format_citation(citations):
    parts = []
    for c in citations or []:
        s = c.get("source", "")
        if c.get("section"):
            s += f", {c['section']}"
        if c.get("year"):
            s += f" ({c['year']})"
        parts.append(s)
    return "; ".join(parts) if parts else "No citation provided"


def flatten_condition_entries(entries):
    """Convert condition-level entries (Supplementary Table S3 layout) into citable items.

    Each condition yields up to five items with stable IDs derived from its id_prefix:
    <PREFIX>-01 (summary), -FEAT-01, -RED-01, -MGMT-01, -URG-01.
    """
    items = []
    for e in entries:
        prefix = e["id_prefix"].upper()
        citation = _format_citation(e.get("citations"))
        for field, suffix, tag in _FIELD_ORDER:
            value = e.get(field)
            if not value:
                continue
            if isinstance(value, list):
                label = {"typical_clinical_features": "Typical clinical features",
                         "red_flags": "Red flags"}[field]
                statement = f"{label} of {e['name']}: " + "; ".join(value) + "."
            elif field == "consultation_urgency":
                statement = f"Consultation urgency for {e['name']}: {value}"
            elif field == "initial_management":
                statement = f"Initial management of {e['name']}: {value}"
            else:
                statement = f"{e['name']}: {value}"
            items.append({"id": f"{prefix}-{suffix}", "class": e["class"],
                          "condition": e["name"], "tags": [tag],
                          "statement": statement, "citation": citation})
    return items


def load_knowledge_base(path=None):
    """Load the KB and return a flat list of citable items.

    Accepts either the condition-level JSON shipped with this repository
    ({"entries": [...]}) or a list of statement-level items that already have
    "id", "class", "tags", "statement" and "citation" keys.
    """
    path = path or C.KNOWLEDGE_BASE_PATH
    with open(path, encoding="utf-8") as f:
        kb = json.load(f)
    if isinstance(kb, dict) and "entries" in kb:
        items = flatten_condition_entries(kb["entries"])
    elif isinstance(kb, list) and kb and "statement" in kb[0]:
        items = kb
    else:
        raise ValueError(f"Unrecognized knowledge-base format: {path}")
    ids = [it["id"] for it in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate knowledge-base item IDs: {sorted(dupes)}")
    bad = [i for i in ids if not re.fullmatch(ID_PATTERN, i)]
    if bad:
        raise ValueError(f"Item IDs not matching {ID_PATTERN}: {bad}")
    return items


# ------------------------------------------------------------------ #
# 2. Retrieval
# ------------------------------------------------------------------ #
def retrieve_kb_items(predicted_class, kb_items, include_general=True):
    """All items tagged to the predicted class, plus universal ("ALL") items."""
    items = [it for it in kb_items if it["class"] == predicted_class]
    if include_general:
        items += [it for it in kb_items if it["class"] == "ALL"]
    return items


def format_kb_for_prompt(items):
    return "\n".join(f"[{it['id']}] ({', '.join(it['tags'])}) {it['statement']} "
                     f"(Source: {it['citation']})" for it in items)


# ------------------------------------------------------------------ #
# 3. Prompt
# ------------------------------------------------------------------ #
SYSTEM_INSTRUCTION = (
    "You are a clinical-decision SUPPORT assistant for non-ophthalmologist "
    "clinicians. You do NOT provide definitive diagnoses. You summarize "
    "considerations and triage urgency to help decide when to involve "
    "ophthalmology.\n\n"
    "STRICT GROUNDING RULES:\n"
    "1. Use ONLY the provided knowledge items. Do not introduce facts that are "
    "not supported by a provided item.\n"
    "2. After every clinical claim, cite the supporting item id in brackets, "
    "e.g. [UVE-01]. A claim with no citation is not permitted.\n"
    "3. If the knowledge items do not cover something needed, explicitly write "
    "'Not covered by provided guidance' rather than inventing content.\n"
    "4. When any red-flag item applies, escalate the urgency accordingly and "
    "err toward MORE urgent when uncertain (safety-first triage).\n"
)

OUTPUT_CONTRACT = (
    "Respond in markdown with EXACTLY these headers:\n"
    "## Condition summary\n"
    "## Typical clinical features\n"
    "## Red flags\n"
    "## Initial management\n"
    "## Consultation urgency\n"
    "Under 'Consultation urgency' output exactly one of: "
    "'Emergent (within 1 hour)', 'Urgent (within 24 hours)', "
    "'Routine (outpatient follow-up)'. Base this on the red-flag items; if any "
    "emergent red flag applies, you must select Emergent.\n"
)


def build_grounded_prompt(predicted_class, kb_items, class_probability=None):
    if predicted_class not in C.CLASSES:
        raise ValueError(f"Unknown class '{predicted_class}'. Expected one of {C.CLASSES}.")
    items = retrieve_kb_items(predicted_class, kb_items)
    conf = f" (confidence {class_probability:.2f})." if class_probability is not None else "."
    user_prompt = (f"Model predicted class: '{predicted_class}'{conf}\n\n"
                   f"KNOWLEDGE ITEMS (the ONLY permitted source of facts):\n"
                   f"{format_kb_for_prompt(items)}\n\n{OUTPUT_CONTRACT}")
    return SYSTEM_INSTRUCTION, user_prompt, items


# ------------------------------------------------------------------ #
# 4. Post-hoc grounding audit
# ------------------------------------------------------------------ #
_CLINICAL_KW = re.compile(
    r"\b(pain|vision|redness|discharge|urgent|emergent|photophobia|floaters|swelling|treat|"
    r"refer|steroid|antibiotic|IOP|proptosis|diplopia)\b", re.I)


def audit_grounding(generated_text, allowed_item_ids):
    """Flag (a) cited IDs not in the retrieved set and (b) clinical lines with no citation."""
    cited = re.findall(r"\[(" + ID_PATTERN + r")\]", generated_text)
    allowed = set(allowed_item_ids)
    fabricated = sorted({c for c in cited if c not in allowed})
    uncited = []
    for line in generated_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if _CLINICAL_KW.search(s) and not re.search(r"\[" + ID_PATTERN + r"\]", s):
            uncited.append(s)
    return {"n_citations": len(cited), "n_fabricated_refs": len(fabricated),
            "fabricated_refs": fabricated, "n_uncited_clinical_lines": len(uncited),
            "uncited_clinical_lines": uncited,
            "grounding_pass": not fabricated and not uncited}


# ------------------------------------------------------------------ #
# 5. Gemini call
# ------------------------------------------------------------------ #
def get_gemini_client(api_key=None):
    """Create a google-genai client. Reads GEMINI_API_KEY (or GOOGLE_API_KEY) if no key is given."""
    from google import genai

    api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable (never commit keys).")
    return genai.Client(api_key=api_key)


def _generate_with_retry(client, model, contents, config, max_retries=5, base_delay=2.0):
    from google.genai import errors as genai_errors

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except (genai_errors.ServerError, genai_errors.ClientError) as e:
            retryable = e.code in (429, 500, 503)
            if retryable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  HTTP {e.code} (attempt {attempt + 1}/{max_retries}); retrying in {delay:.1f}s")
                time.sleep(delay)
                continue
            raise


def generate_grounded_instruction(client, predicted_class, kb_items, class_probability=None,
                                  model=None, temperature=None):
    """Generate guidance with Gemini and return prompt, output, audit and provenance metadata."""
    from google.genai import types

    model = model or C.GEMINI_MODEL
    temperature = C.LLM_TEMPERATURE if temperature is None else temperature
    system, user, items = build_grounded_prompt(predicted_class, kb_items, class_probability)
    resp = _generate_with_retry(
        client, model, user,
        types.GenerateContentConfig(system_instruction=system, temperature=temperature))
    text = resp.text
    allowed = [it["id"] for it in items]
    return {
        "predicted_class": predicted_class,
        "class_probability": class_probability,
        "retrieved_item_ids": allowed,
        "system_instruction": system,
        "user_prompt": user,
        "output_text": text,
        "grounding_audit": audit_grounding(text, allowed),
        "llm_model": model,
        "llm_model_version_reported": getattr(resp, "model_version", None),
        "temperature": temperature,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def save_generation_record(record, output_dir=None, name=None):
    """Persist every generation (ARVO AI policy: model, version and date of generated content)."""
    output_dir = output_dir or C.LLM_DIR
    os.makedirs(output_dir, exist_ok=True)
    name = name or f"{record['predicted_class']}_{record['generated_at_utc'].replace(':', '')}"
    path = os.path.join(output_dir, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path
