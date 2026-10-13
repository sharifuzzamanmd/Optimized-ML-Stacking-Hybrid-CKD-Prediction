# Dataset

**Chronic Kidney Disease (CKD) dataset**: 400 patient records collected at a hospital in Karaikudi, Tamil Nadu, India (L. Jerlin Rubini, Dr. P. Soundarapandian and Dr. P. Eswaran, 2015).

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- Kaggle mirror used for the CSV: https://www.kaggle.com/datasets/mansoordaku/ckdisease

File: `original/kidney_disease.csv` (raw, unmodified; 400 rows × 26 columns including `id`).

| Column | Attribute | Type / Units |
|---|---|---|
| `age` | Age | years |
| `bp` | Blood pressure | mm/Hg |
| `sg` | Specific gravity | 1.005 / 1.010 / 1.015 / 1.020 / 1.025 |
| `al` | Albumin | 0–5 |
| `su` | Sugar | 0–5 |
| `rbc` | Red blood cells | normal / abnormal |
| `pc` | Pus cell | normal / abnormal |
| `pcc` | Pus cell clumps | present / notpresent |
| `ba` | Bacteria | present / notpresent |
| `bgr` | Blood glucose random | mgs/dl |
| `bu` | Blood urea | mgs/dl |
| `sc` | Serum creatinine | mgs/dl |
| `sod` | Sodium | mEq/L |
| `pot` | Potassium | mEq/L |
| `hemo` | Haemoglobin | gms |
| `pcv` | Packed cell volume | % |
| `wc` | White blood cell count | cells/cumm |
| `rc` | Red blood cell count | millions/cmm |
| `htn` | Hypertension | yes / no |
| `dm` | Diabetes mellitus | yes / no |
| `cad` | Coronary artery disease | yes / no |
| `appet` | Appetite | good / poor |
| `pe` | Pedal edema | yes / no |
| `ane` | Anemia | yes / no |
| `classification` | Target | ckd (250) / notckd (150) |

The raw file contains missing values and whitespace artefacts (e.g. `"\tno"`, `"ckd\t"`, `"\t?"`). These are cleaned in `src/ckd_pipeline.py`.
