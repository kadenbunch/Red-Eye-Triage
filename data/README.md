# Data

## Source

Images were derived from the publicly available Mendeley Data dataset:

> *Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation*, version 2 (2024). https://doi.org/10.17632/n9zp473wfw.2

According to its description, the source images were collected from online sources using disease-specific search terms, and the published version includes SMOTE-based class balancing. Only the original (non-synthetic) images were eligible for this study. 

## Curation (see manuscript Methods)

1. Cataract images were excluded (lens pathology is outside red-eye triage).
2. Two reviewers with ophthalmic clinical experience independently reviewed every remaining image and excluded images inconsistent with the four target categories, conditions with fewer than 30 example images (for example pterygium and chemosis), duplicate images, and multiple images identifiable as coming from the same patient.
3. Conjunctivitis and anterior uveitis were merged into a single **Inflammatory** class.
4. Final dataset used can be found within data/Archive.zip

| Class | Images |
|---|---|
| Inflammatory | 277 |
| Eyelid | 323 |
| Normal | 642 |
| Hemorrhage | 62 |
| **Total** | **1,304** |

## Expected layout

```
data/images/
├── Inflammatory/
├── Eyelid/
├── Normal/
└── Hemorrhage/
```

Class folder names must match exactly. Files are read in sorted filename order, so **do not rename, add or remove files** if you want to reuse the released split (`results/split_manifest.csv`); the pipeline stops with an error if the image count no longer matches the saved split. Hidden files such as `.DS_Store` are ignored.
