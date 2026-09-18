# Knowledge base

`knowledge_base.json` is the author-curated knowledge base that constrains the LLM (manuscript Supplementary Table S3). It contains 15 condition entries mapped to the four classifier categories.

## Schema

| Field | Type | Description |
|---|---|---|
| `condition_id` | string | stable identifier, e.g. `anterior_uveitis` |
| `id_prefix` | string (A-Z) | prefix for citable item IDs, e.g. `UVE` |
| `name` | string | display name |
| `class` | string | classifier class it is retrieved for: `Inflammatory`, `Eyelid`, `Normal`, `Hemorrhage`, or `ALL` (retrieved for every class) |
| `condition_summary` | string | short paraphrased summary |
| `typical_clinical_features` | list of strings | |
| `red_flags` | list of strings | findings that should escalate urgency |
| `initial_management` | string | |
| `consultation_urgency` | string | |
| `citations` | list of `{source, section, year}` | guidance the entry was paraphrased from |

## Citable item IDs

At load time (`redeye_triage.llm_grounding.load_knowledge_base`) each entry is split into up to five citable items, and the LLM must cite these IDs:

| ID | Content |
|---|---|
| `<PREFIX>-01` | condition summary |
| `<PREFIX>-FEAT-01` | typical clinical features |
| `<PREFIX>-RED-01` | red flags |
| `<PREFIX>-MGMT-01` | initial management |
| `<PREFIX>-URG-01` | consultation urgency |

Print the exact prompt for a class without calling the API:

```bash
python scripts/05_generate_guidance.py --print-prompt Inflammatory
```

## Status

The content was written and reviewed by the authors. Citations have not yet been independently verified against the primary guidance documents (manuscript, Future Directions). This knowledge base is not a clinical guideline.
