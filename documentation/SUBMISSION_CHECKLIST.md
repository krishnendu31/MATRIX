# MATRIX: final submission checklist

Tick each box before submitting. Items marked **(team)** need information or decisions only the
team has; everything else has been done and verified (status as of the last update of this file).

## 1. Repository and README

- [ ] **(team)** Fill in the team member names in the README header table.
- [ ] **(team)** Add the live demo link (e.g. Streamlit Community Cloud) to the README.
- [ ] **(team)** Add the demo video link to the README.
- [ ] **(team)** Add screenshots to `demo/` and reference them in the README "Screenshots / demo"
      section. Suggested shots (all reachable from the running app):
  - Overview page (hero + "Model at a glance").
  - Analyzer with the **Prize scam** example: verdict, highlighted words, "Why".
  - Analyzer counterfactual table for the same example.
  - Analyzer with the **Delivery link** example (amber "borderline" state + uncertainty notes).
  - Dashboard: Performance tab (confusion matrix + calibration).
  - Dashboard: Explanation quality tab (agreement + faithfulness).
  - Dashboard: Robustness tab.
  - The app on a phone-width screen.
- [ ] **(team)** Confirm the "Use of AI tools" statement in the README is accurate for your team.
- [ ] **(team)** Create the Git repository and push it (this folder is not a Git repository yet).
      `.gitignore` already excludes `.venv/` and caches.
- [ ] **(team)** Decide whether to commit `data/SMSSpamCollection` (CC BY 4.0, 478 KB, allowed
      with attribution) and `models/` (about 1.5 MB). Committing both lets judges run the app
      without training; otherwise they must download the data and run `python -m src.train`.
- [x] README covers problem, solution, features, architecture, methodology, installation,
      usage, dataset, models, results, limitations, future scope, responsible AI, attributions.
- [x] Every number in the README Results section comes from `models/report.json`.
- [x] License file present (MIT).

## 2. Runs on a clean machine

- [x] `requirements.txt` pins every runtime dependency; a brand-new virtual environment installs
      it with no conflicts (`pip check` clean).
- [x] `python -m src.train` runs end to end on a laptop CPU (about 3 minutes).
- [x] `streamlit run app.py` starts; all four pages render with no errors.
- [x] The app never trains by itself. A missing dataset, missing or corrupt model, model from
      another scikit-learn version, missing or broken report, and `--fast`-only results each show
      a clear message instead of crashing.
- [ ] **(team)** Do one final run-through on a second team member's laptop, following only the
      README.

## 3. Correctness and honesty

- [x] 31 automated tests pass (`pytest`).
- [x] No duplicate messages; no message in both training and test data (tested).
- [x] The test set is never used for training, tuning or calibration.
- [x] Reported metrics recompute exactly from the saved test predictions (tested).
- [x] Retraining from the code and seed reproduces the saved model exactly (tested).
- [x] SHAP checked against brute-force Shapley values (tested); SHAP units (log-odds, before
      calibration) are stated in the app and README.
- [x] Counterfactual scores are real model outputs for every edited message (tested).
- [x] User text is shown exactly as typed and never executed as HTML or Markdown (tested).
- [x] Known limitations are listed in the README and in the app's About page.
- [ ] Optional: run `notebooks/verify_xai.py` against the official `shap` and `lime` packages
      (`pip install -r requirements-dev.txt`; these packages are large and were not installed
      during development).

## 4. Demo

- [ ] **(team)** Rehearse `documentation/demo_script.md` once with the app running.
- [ ] **(team)** Record the demo video (suggested length 3–5 minutes) following the script.
- [ ] **(team)** If deploying to Streamlit Community Cloud: the app needs `models/` and the
      dataset in the repository (or a training step); the main file is `app.py`, Python 3.13.

## 5. Hackathon rules

- [ ] **(team)** Check the submission form's required fields and deadline.
- [ ] **(team)** Check that the Stage 1 proposal and this submission describe the same project
      (the proposal mentions UCI Spambase and DistilBERT; both are listed under Future scope,
      not implemented).
