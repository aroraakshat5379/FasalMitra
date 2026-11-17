# FasalMitra: Implementation Plan (revised)

**Federated Deep Learning for Multi-Crop Disease Detection and Farmer Advisory System**
Team: Khushi Singh, Pritha Gulati, Akshat Arora | Guide: Dr Mahesh Kumar | IT, BVCOE

---

## 1. Overview

**Goal:** train one shared multi-crop disease classifier across 5 simulated non-IID clients using federated learning (FedAvg), without moving raw images, and wrap it in a Farmer Advisory Module and a demo app.

**Locked choices:** PlantVillage (38 classes) | MobileNetV2 (ImageNet-pretrained) | FedAvg (Flower in Semester 2) | TensorFlow/Keras | 5 clients split by crop | rule-based advisory.

**Team tracks** (rotate every phase or two so all three can explain every part in the viva):
- **Track A: Data & Evaluation** (splits, metrics, plots, analysis)
- **Track B: Model & Training** (MobileNetV2, baseline, tuning)
- **Track C: Federated Setup, Advisory & Docs** (FL loop, advisory content, demo app, report)

**Working rules:** one GitHub repo, pinned `requirements.txt`, fixed seed (42), results saved to files after every run, and a short log with screenshots after every phase (this feeds the progress form).

---

## 2. Phase Implementation Plan

Each phase builds on the output of the one before it. Semester 1 delivers the basic working pipeline; Semester 2 delivers the full product.

| Phase | Goal | Key work | Output | Semester |
|---|---|---|---|---|
| **0** | Ideation and planning | Domain, literature survey, synopsis, mini prototype notebook | Synopsis and a working prototype | 1 (done) |
| **1** | Script-based pipeline | Convert the notebook into scripts: data, model, centralized training, FedAvg, evaluation; cache the data subset | Repo that runs end to end from the command line | 1 |
| **2** | Larger run, inference, advisory seed | Larger-subset runs, `predict.py`, confidence threshold, advisory entries for the most common classes | Image-to-disease-and-advice flow and a results table | 1 |
| **Checkpoint** | Freeze and document | Tag the repo, results table, report draft, progress form | Basic pipeline with preliminary results | 1 (end) |
| **3** | Full dataset and Flower | Full PlantVillage, Flower client and server, per-round logging | Flower FedAvg on the full dataset | 2 |
| **4** | Full baseline and fine-tuning | Fine-tune top layers, tune hyperparameters | Frozen centralized benchmark | 2 |
| **5** | Federated experiments and evaluation | Rounds and local epochs, local-only baseline, non-IID severity, error analysis, communication cost | Complete evaluation chapter | 2 |
| **6** | Full advisory module | All 38 classes, sourced from ICAR or agricultural universities, optional Hindi | Complete `advisory.json` and lookup | 2 |
| **7** | Improvements and demo app | Weak-class fixes, Streamlit or Gradio app, field-photo test, optional TFLite | Working demo from photo to advice | 2 |
| **8** | Report, PPT and documentation | Final report, PPT, README, backup demo video | Submission-ready documents | 2 |
| **9** | Final review and buffer | Demo rehearsals, viva practice, final submission | Final submission | 2 |

---

## 3. Phase 0: Ideation and Planning (DONE)

- [x] Domains evaluated, project and title finalized
- [x] Literature survey (10 papers) and research gap identified
- [x] One-page synopsis written
- [x] Mini prototype notebook: small subset, non-IID split, MobileNetV2, centralized vs FedAvg trial

---

# SEMESTER 1 (now to end of December): basic working pipeline

## Today (Tue 17 Nov): run it locally and prepare for tomorrow

- [ ] Set up a local Python environment (Python 3.10-3.12) and install the packages
- [ ] Run the prototype notebook with the default settings first (safety net, about 10-20 min)
- [ ] Copy the `results` folder to `results_run1` so the next run does not overwrite it
- [ ] Launch a larger overnight run (bigger subset, more epochs and rounds) as a notebook execution that does not need the browser open
- [ ] Update the progress slide with real numbers while it runs
- [ ] Morning: collect the overnight results, finalize the slide, rehearse once

**Output:** centralized and FedAvg accuracy, macro-F1, classification reports, split chart, convergence plot.

## Wed 18 Nov: progress presentation

- [ ] Present using the progress slide, the split chart, the convergence plot and the comparison table
- [ ] Note the faculty's feedback and questions in the repo's notes file

## Phase 1: Script-Based Pipeline (19 Nov - 30 Nov)

Convert the notebook into plain Python scripts so runs are repeatable and the team can run them independently.

**Proposed layout**
```
fasalmitra/
  config.py         # seed, image size, clients, rounds, epochs
  data.py           # load PlantVillage, subset, global test split, non-IID split, cache to disk
  model.py          # MobileNetV2 + classifier head
  train_central.py  # centralized baseline
  train_fedavg.py   # FedAvg simulation (5 clients)
  evaluate.py       # metrics, classification report, plots
  predict.py        # one image -> class, confidence, advice
  advisory.json     # seed entries (see Phase 2)
  results/
```

**Tasks**
- [ ] Create the GitHub repo and `requirements.txt`; add a README with run commands
- [ ] Move the notebook code into the modules above; cache the sampled subset to disk so reruns are fast
- [ ] Make every script write its metrics, plots and logs into `results/`
- [ ] Each teammate runs the pipeline once on their own machine to confirm it is reproducible

**Output:** a repo where `data.py`, `train_central.py` and `train_fedavg.py` run end to end.
**Lead:** Track A (data, evaluate) | Track B (model, central) | Track C (fedavg, predict, README)

## Phase 2: Larger Run, Inference and Advisory Seed (1 Dec - 15 Dec)

**Tasks**
- [ ] Run the pipeline on a larger subset (more images per class) and record centralized and FedAvg numbers
- [ ] Write `predict.py`: load the trained model, take one leaf image, print the disease, the confidence and the advice
- [ ] Add a confidence threshold: low confidence returns "consult an expert" instead of guessing
- [ ] Seed `advisory.json` with entries for the 5-6 most common classes, each sourced from ICAR or state agricultural university material (full 38 classes next semester)
- [ ] Optional stretch: a Flower hello-world run on a toy model, to start the migration early

**Output:** end-to-end flow from photo to disease and advice for the seeded classes; larger-run results table.
**Lead:** Track B (larger runs, threshold) | Track C (predict, advisory seed) | Track A (results, plots)

## December-End Checkpoint (16 Dec - 31 Dec)

- [ ] Freeze the pipeline: tag the repo, confirm it runs from a fresh clone
- [ ] Results table: centralized vs FedAvg on the larger subset, with the classification reports
- [ ] Draft report sections: introduction, literature survey, methodology, preliminary results
- [ ] Update the progress form and PPT with the final Semester 1 status
- [ ] Write a short "what remains" note for Semester 2

**Semester 1 deliverable:** a working basic pipeline (data, split, baseline, FedAvg, evaluation, one-image prediction with advice) with real preliminary results.

---

# SEMESTER 2 (next): full product (major chunk)

## Phase 3: Full Dataset and Flower Migration (Weeks 1-2)
- [ ] Run on the full PlantVillage dataset with the saved non-IID partitions
- [ ] Replace the hand-written FedAvg with a Flower client and server, using the same model builder
- [ ] Log per-round global accuracy and loss to CSV

**Lead:** Track C | **Support:** A, B

## Phase 4: Full Baseline and Fine-Tuning (Weeks 3-4)
- [ ] Fine-tune the top backbone layers; tune learning rate, batch size and epochs
- [ ] Freeze the final centralized baseline and its numbers as the benchmark

**Lead:** Track B | **Support:** A

## Phase 5: Federated Experiments and Evaluation (Weeks 5-7)
- [ ] Tune rounds and local epochs
- [ ] Add a local-only baseline (each client trains alone) to show what federation gains
- [ ] Test different non-IID severities
- [ ] Per-client and per-class analysis; confusion-matrix error analysis
- [ ] Communication cost: weights per round vs uploading raw images

**Lead:** Track A | **Support:** B, C

## Phase 6: Full Farmer Advisory Module (Weeks 6-8, in parallel)
- [ ] Extend `advisory.json` to all 38 classes: disease, symptoms, treatment (organic and chemical), prevention
- [ ] Source every entry from ICAR or agricultural university material, with links
- [ ] Stretch: Hindi text for the same entries

**Lead:** Track C | **Support:** A (verification)

## Phase 7: Improvements and Demo App (Weeks 8-10)
- [ ] Address weak classes (class weights or extra augmentation)
- [ ] Build a Streamlit or Gradio app: upload a leaf photo, get disease, confidence and advice
- [ ] Test on a few field-style photos and record how performance differs from PlantVillage
- [ ] Stretch: TFLite conversion to show edge readiness

**Lead:** Track C (app) | Track B (model) | Track A (testing)

## Phase 8: Report, PPT and Documentation (Weeks 10-11)
- [ ] Final report: methodology, implementation, results, limitations, future work
- [ ] Final PPT, demo script and a backup demo video or screenshots
- [ ] README with setup instructions; repo runs from a fresh clone

**Lead:** all three, each writing their own chapters

## Phase 9: Final Review and Buffer (Week 12)
- [ ] Full demo rehearsals with timing and viva practice
- [ ] Final submission and a buffer for anything that slipped

---

## 4. Milestones

| When | Deliverable |
|---|---|
| 18 Nov | Progress presentation with prototype results |
| End of November | Script-based pipeline in a shared repo |
| Mid December | Larger-run numbers and image-to-advice flow for seeded classes |
| 31 Dec | Frozen basic pipeline, results table, report draft, updated progress form |
| Semester 2, mid | Full-dataset Flower FedAvg and complete evaluation |
| Semester 2, end | Demo app, full advisory, final report, PPT, submission |

Update the project progress form at the end of every phase.

---

## 5. Risks and Fallbacks

- **December exams:** Phases 1 and 2 are light by design; if a week is lost, skip the optional stretch items first.
- **Local machine is slow or has no GPU:** keep the default small config for quick checks, and run larger configs overnight.
- **Flower setup trouble:** the hand-written FedAvg stays as a working fallback.
- **FedAvg converges slowly on skewed clients:** a valid finding; report it, tune rounds and local epochs, and use the local-only baseline to show the benefit.
- **Accuracy looks too good on PlantVillage:** mention near-duplicate leakage and the lab-vs-field gap openly.
- **A teammate falls behind:** shared repo, rotated tracks and weekly reviews, so any phase can be picked up by another member.

---

## 6. Handoff to the Major Project (Year 4)

Real multi-device FL deployment, differential privacy, a yield-prediction module, a regional-language chatbot, and Grad-CAM explainability.

---

## 7. Likely Viva Questions

1. Why federated learning for this problem?
2. What does non-IID mean here, and how did you create it?
3. Why MobileNetV2?
4. Why is FedAvg lower than centralized, and how did you narrow the gap?
5. What would change with real field images?
6. How does the advisory module work and where did its content come from?
