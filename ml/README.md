# Kenya Watch ML workspace

This folder contains dataset checks and experimental category-model training. The live service continues to use `backend/utils/classifier.js` for category and urgency triage. A separately generated runtime artifact can provide reviewers an experimental category suggestion only after they submit their own independent category assessment. The trained model never sets report category or urgency and is not approved for real-world casework.

The reviewer queue now has a separate, unsupervised TF-IDF/cosine-similarity feature in `backend/utils/relatedCaseMatcher.js`. It ranks possible textually similar reports within the same county after a reviewer submits an independent category assessment. It does not use a trained model or category labels, and its score is not a probability or a duplicate finding. Reviewers must decide whether to link cases. This bounded lexical baseline is suitable only as a review aid; language and phrasing limits and the 500-candidate scan are documented in the root README.

Kenyan-language work (code-switching, Sheng, vocabulary growth, and what data may and may not be used) is in [LINGO.md](LINGO.md).

See [DATA_SOURCES.md](DATA_SOURCES.md) for dataset provenance and collection guidance, and [LABELING_GUIDE.md](LABELING_GUIDE.md) for the pilot annotation rules.

## Dataset provenance and status

There is currently **no approved training dataset in this repository**. No external dataset was downloaded or used. The earlier database review counted a small number of local MongoDB reports in aggregate; those were not exported, their narratives were not used for training, and the reporter-selected category is not a ground-truth label.

The intended source is reports with a final `categoryReview.adjudicatedCategory` produced by the blinded staff review workflow. Agreement between two reviewers or an Admin-documented adjudication is required. The application does not yet provide a dataset-export endpoint, so do not assume that the MongoDB files under `backend/data/` are a training corpus.

## Private dataset layout

Keep all report data out of source control:

- `data/raw/`: source material under an approved data-handling plan; do not put raw exports here unless strictly necessary.
- `data/approved/`: reviewed, minimized CSV prepared for an approved experiment.
- `data/processed/`: temporary cleaned or split files, if needed.
- `artifacts/`: local model and evaluation outputs.

These paths are ignored by Git. Before creating a training CSV, obtain approval for the intended use, minimize the text, remove direct identifiers and unnecessary sensitive details, and use a pseudonymous `group_id` to keep reports about the same underlying incident in one validation fold. Do not commit narratives, identifiers, evidence, credentials, model artifacts, or evaluation outputs.

The input CSV must have UTF-8 headers `text,label,group_id`. Optional columns `language` and `county` are used only for aggregate evaluation, not as model features. `label` must be one of the canonical categories in `backend/utils/categories.js` and must be independently reviewed; do not use reporter-selected categories as labels. `group_id` is a stable pseudonym for the incident, not the reporter or an individual.

## Building the local candidate dataset

The exporter reads only report IDs listed in an explicit local allowlist and only accepts active reports with a finalized reviewed category. Two matching reviewer categories or a category chosen in documented Admin adjudication are eligible. Insufficient-information decisions, disagreements, unreviewed reports, and reporter-selected labels are rejected. Linked reports are assigned the same keyed pseudonymous incident group.

Before using the exporter, have the supervisor/institutional data-protection contact approve the intended research use and confirm each allowlisted report is eligible. The exporter cannot determine legal authority or reporter consent; the manifest is an operational allowlist, not a substitute for approvals.

Record those approvals in [`research_approvals.json`](research_approvals.json) by filling in the `reference` of each entry (an ethics approval number, a DPO assessment reference). An entry with an empty `reference` counts as **not granted**, and the startup exporter copies this record into every export's `.metadata.json` under `approvals`. The readiness gate reads the resulting `approval_reference` on the evaluation, so a model trained on real reports cannot be described as accurate unless both references are present. The supervisor also holds the sealed test set named in that file; only reports in it may be used to report real-world performance, so the figure cannot be inflated by tuning on the same data.

1. Create `ml/data/approved/report_ids.txt` locally, one Mongo report ObjectId per line. Use only reports cleared under the agreed research/data-handling process. Lines beginning with `#` are ignored. This file and its parent folder are Git-ignored.
2. Set `ML_GROUP_HASH_KEY` in the local ignored `backend/.env` to a securely generated random secret of at least 32 characters. Do not reuse a password or commit/share the secret; keep it stable for repeatable group IDs.
3. Run a summary-only dry run; this checks that all selected reports are active and have finalized reviewed category labels without writing narrative text:

   ```powershell
   node ml\scripts\export-reviewed-dataset.js --approval-file ml\data\approved\report_ids.txt
   ```

4. Only after checking the aggregate summary and approval, explicitly write the sensitive staging export:

   ```powershell
   node ml\scripts\export-reviewed-dataset.js --approval-file ml\data\approved\report_ids.txt --write-raw --confirm-sensitive-export
   ```

The exporter writes title plus narrative into `ml/data/raw/` only. Treat this output as sensitive personal data: keep it local and access-restricted, review and manually de-identify each narrative, remove direct identifiers and unnecessary identifying details, and prepare a minimized CSV in `ml/data/approved/`. The exporter does not automatically anonymize text, and pseudonymous group IDs do not make narratives anonymous. Do not point the training command at the raw staging file.

Missing text or labels are not imputed. Empty/invalid labels, exact duplicate text, missing incident groups, or insufficient class/group counts block training. Review likely near-duplicates manually during dataset preparation; the audit does not automatically delete unusual or rare examples.

## Setup and commands

Use the repository's `.venv` (Python 3.10 or newer):

```powershell
.\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s ml\tests -v
.\.venv\Scripts\python.exe -m ml.src.generate_synthetic_dataset
.\.venv\Scripts\python.exe -m ml.src.audit_dataset ml\data\synthetic\kenya_watch_synthetic_seed.csv
.\.venv\Scripts\python.exe -m ml.src.train ml\data\synthetic\kenya_watch_synthetic_seed.csv
.\.venv\Scripts\python.exe -m ml.src.audit_dataset ml\data\approved\reports.csv
.\.venv\Scripts\python.exe -m ml.src.train ml\data\approved\reports.csv
node --test backend\tests\mlDatasetExport.test.js
node --test backend\tests\experimentalCategoryModel.test.js
```

## Synthetic pipeline demonstration

`data/synthetic/kenya_watch_synthetic_seed.csv` is a deterministic, hand-authored seed corpus of 540 fictional examples: 15 scenario groups for each of the nine categories. `kenya_watch_synthetic_colloquial.csv` adds 216 examples, and `kenya_watch_synthetic_realistic.csv` adds 2,160 report-form-like examples across 540 incident groups. The combined benchmark therefore has 2,916 rows and 729 incident groups. These examples are not observations of Kenyan reports, and Kiswahili, mixed-language and Sheng passages have not been independently reviewed as real citizen language.

The combined benchmark uses five-fold stratified group outer validation and three-fold grouped inner model selection; variants of each fictional incident remain together. The notebook run produced 0.9571 accuracy and macro-F1. A separate runtime run added 10,000 register rows only to the training side at weight 0.25 and produced 0.9705 nested out-of-fold accuracy and macro-F1, with a 0.9706 outer-fold mean and 0.0080 standard deviation. These scores demonstrate pipeline and corpus behaviour only. Templatic phrasing, simplified scenarios and balanced classes can inflate them; neither number estimates accuracy on real Kenyan reports or establishes fitness for use. The live reviewer queue shows the model only as an explicit experimental suggestion after the reviewer has committed an independent assessment. It does not show a probability or score, overwrite any label, change urgency or trigger workflow actions. Runtime and metadata artifacts are written locally under ignored `ml/artifacts/`; the app reports an error if an artifact is missing or invalid.

### How much of that score is the corpus?

`data/synthetic/kenya_watch_synthetic_colloquial.csv` is a hand-authored corpus (216 rows, 54 incidents) written in registers the seed lacks: short first-person accounts, SMS-style spelling, Sheng and code-switching, second-hand reports and question forms. `data/synthetic/kenya_watch_synthetic_realistic.csv` adds 2,160 report-form-like rows: 60 incidents per category in English, Kiswahili, mixed and Sheng variants.

Generate all corpora and combine them with:

```powershell
python -m ml.src.generate_synthetic_dataset
python -m ml.src.generate_colloquial_dataset
python -m ml.src.generate_realistic_synthetic_dataset
python -m ml.src.build_combined_corpus
python -m ml.src.train ml\data\synthetic\kenya_watch_synthetic_combined.csv
```

The seed-only result was 0.8784 macro-F1, and the older seed-plus-colloquial result was 0.8346. The expanded combined result is 0.9571 in the notebook run and 0.9705 in the register-supplemented runtime run. These are different experiments and must not be combined or presented as a real-world accuracy progression.

### The register corpus: training only, never scored

`data/synthetic/kenya_watch_synthetic_registers.csv` (`python -m ml.src.generate_register_dataset`) holds 10,000 fictional reports: 2,500 scenarios from everyday Kenyan settings (roadblocks, county askari, the lands registry, Huduma Centres, county hospitals, bursary and CDF money, tenders, county jobs), each written in plain English, plain Kiswahili, mixed English–Kiswahili and Sheng. The four versions of a scenario describe the same incident and share a `group_id`.

Because it comes from templates, its own cross-validation score would say little, so `ml.src.train` adds it to the training side of every fit (model selection, nested evaluation, calibration folds and the final model) at weight 0.25 and never to a test fold. Every score in the combined experiment is still measured on the 2,916-row main corpus. Set `KW_SUPPLEMENT=0` to train without it.

`python -m ml.src.experiments_registers` measures the effect on text the corpus was not generated from. Nested macro F1 on the 756 rows rose from 0.877 to 0.898 (Sheng rows 0.773 to 0.869). Trained on the formal seed and tested on the colloquial corpus, the score rose from 0.680 to 0.806: 34 reports corrected and 8 broken, McNemar p < 0.001. The colloquial corpus and the register corpus were both written for this project, so part of that gain may be shared phrasing; only adjudicated real reports can confirm it.

The most informative number in that experiment is the one that did not move. Run the original model over the register corpus and it scores only **0.694**, with Sheng lowest at 0.664. That is a model which looks good on text written the way it was trained and falls by roughly twenty points the moment it meets a different register. The supplement closes part of that gap; it does not close all of it, and neither corpus was written by a citizen describing a real incident.

### How close is the live database to real-data training?

Before any export, `node ml/scripts/label-progress.js` reports how many reports the exporter would actually accept today, and why the rest do not count. It reads only the `categoryReview` block and the archive flag, never the narrative, and it reuses the exporter's own eligibility rule, so it cannot disagree with the export or with the gate.

```powershell
node ml\scripts\label-progress.js          # readable summary with progress bars
node ml\scripts\label-progress.js --json   # machine-readable, for recording in the log
```

It separates reports that are eligible from those waiting on a second reviewer, waiting on an Admin adjudication, judged insufficient, not yet reviewed, and eligible but archived (which the exporter ignores). Point it at whichever database you mean with `MONGO_URI`.

A wider candidate concept vocabulary was also tried and rejected (`KW_CONCEPTS_FILE`), because it did not improve the held-out score. The honest summary is that corpus variety helped and vocabulary inflation did not.

### Native-speaker Sheng review

Everything in the two corpora was written by the project author, so the Sheng is unverified. `ml/lingo/build_sheng_review_sheet.py` writes `ml/data/lingo/sheng_native_review.csv`: a deterministic, category-balanced sample of Sheng sentences with empty columns for a Sheng speaker to judge. The reviewer sets `verdict` to `keep`, `rewrite` or `drop`, rewrites the sentence in `corrected_text` where needed, and answers `is_it_natural_sheng` with yes or no.

```powershell
.\.venv\Scripts\python.exe -m ml.lingo.build_sheng_review_sheet
# ...give the CSV to the Sheng reviewer, then...
.\.venv\Scripts\python.exe -m ml.lingo.apply_sheng_review --reviewer "Name" --reviewed-on 2026-11-22
```

`apply_sheng_review.py` refuses an incomplete sheet rather than guessing: every row needs a valid verdict, a rewrite needs replacement text, and the naturalness answer is mandatory. It then writes `kenya_watch_synthetic_sheng_verified.csv` (leaving the originals untouched) and an attestation file naming the reviewer, the date and the keep/rewrite/drop counts. The other Sheng rows are carried through marked `unreviewed`, so no claim is silently made about text nobody checked.

The step verifies *language*, not *labels*: a reviewer saying a sentence is natural Sheng does not confirm that it exemplifies the alleged category. Treat the output as a better-written copy of the same synthetic data, not as ground truth.

One honest caveat about the parallel-text Sheng source. It was learned with IBM Model 1 from a parallel corpus whose English side is a Bible translation, so the 1,288 stored word pairs are dominated by narrative verbs (`went`, `fell`, `came`, `called`, `replied`) and include scripture vocabulary (`dhambi` sin, `kanisa` church, `imani` faith). Only about eleven entries fall in the corruption domain (`kanjo` council, `jela` jail, `pesa`/`ganji`/`munde` money, `thao` thousand, `machief` chief). The list is a reasonable source of ordinary Sheng words and a poor source of corruption vocabulary, so it must not be presented as a corruption glossary; the corruption glossary in `backend/utils/kenyaLingo.js` remains the place where corruption terms are mapped. Two of the learned pairs are also plainly wrong (`finje` glossed as "groups" rather than fifty, `kuwaseti` as "apostles"), which is why nothing from this list reaches the classifier without review.

The synthetic-trained suggestion is connected to the local live application for demonstration only. It is not ready for deployment or real reports: synthetic results cannot establish operational usefulness, fairness, or safety. Before using a trained classifier in real casework, replace this demonstration with independently reviewed, lawfully approved and de-identified reports, and evaluate against a separately frozen incident-separated test set. Reviewers should validate every label, including ambiguous category boundaries; Kiswahili and other language coverage needs competent human review. Also require pre-agreed per-category acceptance criteria, subgroup error review, model/version provenance, a rollback path, and ongoing monitoring. Never treat an advisory output as a finding or automated decision.

Each training run writes `category_model.runtime.json` alongside the Python model, evaluation, and metadata. The Express service reads that local file (or the path set with `ML_CATEGORY_MODEL_PATH`) when a reviewer requests a suggestion. The artifact is ignored by Git and must be generated and transferred through the deployment process; do not commit it. Restart the backend after replacing the file so its cached model is refreshed.

The audit prints aggregate diagnostics only; it never prints report text. Training currently has a **minimum mechanical gate** of 50 examples and ten distinct incident groups per category. This gate is intended only to permit nested grouped validation to run; it is not a claim that this amount is statistically adequate. The pipeline compares two pre-declared sparse-text baselines—word n-gram TF-IDF with balanced logistic regression, and combined word/character n-gram TF-IDF with balanced linear SVM. The choice is made by macro F1 inside grouped inner folds; outer grouped folds estimate the complete selection process. Results include a fold-local majority-class baseline, per-class precision/recall/F1, balanced accuracy, a confusion matrix, and descriptive outer-fold macro-F1 spread (not a confidence interval). Language/county subgroup results are shown only when the sample is sufficiently large; these exploratory metrics are not proof of fairness. The final artifact is refit on the approved data after model selection and written under ignored `ml/artifacts/`.

The model comparisons are a small, pre-declared benchmark, not a broad hyperparameter search. Do not use the final full-data selection score as a performance claim. Before any model is considered successful, freeze a later-time, incident-separated holdout, define acceptable class-specific error rates with reviewers and affected stakeholders, gather enough independent incidents to estimate them with useful uncertainty, and assess reviewer usefulness. Do not connect an artifact to the live app or make consequential decisions from it. Review data quality, subgroup sample sizes, leakage, errors, and limitations with the project team before changing the workflow.

## Is the model over- or underfitting, and may we call it accurate?

Two commands answer those questions with numbers instead of opinions:

```powershell
.\.venv\Scripts\python.exe -m ml.src.diagnose_fit ml\data\synthetic\kenya_watch_synthetic_combined.csv
.\.venv\Scripts\python.exe -m ml.src.readiness
```

`diagnose_fit` compares training and validation scores with incident groups kept apart, draws a learning curve, runs a shuffled-label control (must fall to chance) and sweeps the regularisation strength. On the expanded combined synthetic corpus the diagnosis gave training macro-F1 1.00 and validation 0.965, a gap of 0.035. It found no over/under-fitting or leakage signal at the configured thresholds. The learning curve still rose at full size, so more independent incidents would remain useful. This diagnosis does not make the score real-world accurate.

`readiness` is the gate for the word "accurate". It passes only for data marked `adjudicated_real_reports` (the metadata JSON must also carry an `approval_reference`) with at least 450 reports, 30 per category, a held-out score whose lower bound is at least 0.70, no category with recall below 0.50, English and Kiswahili within 0.10 of each other, and a clean fit diagnosis. On synthetic data it fails by design, so the model stays labelled an experimental reviewer aid. To pass it: collect adjudicated reports through the blinded review workflow, export them with `ml/scripts/export-reviewed-dataset.js`, then train, diagnose and run the gate again.
