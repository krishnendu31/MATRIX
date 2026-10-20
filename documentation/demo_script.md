# MATRIX demo script (3–5 minutes)

Every result below is the real output of the trained model (`models/matrix.joblib`, seed 42),
recorded with the final code. If you retrain, re-check the numbers before presenting.
Start the app with `streamlit run app.py` and open http://localhost:8501.

## 1. Overview (30 s)

- Open on **Overview**. One sentence on the problem: filters give a verdict, never a reason.
- Point at **Model at a glance**: F1 0.961, precision 99.2%, recall 93.1%, and only **1 of 904**
  normal test messages wrongly flagged. Stress that these come from 1,035 held-out messages the
  model never saw during training.
- Click **Analyze a message**.

## 2. A classic scam (60 s)

Click the **Prize scam** example, then **Analyze**.

- Verdict: **Suspicious, >99%**, uncertainty low.
- Highlighting: the phone number, the money amount, "mobile" and "call" are strong signals.
- **Why:** "Flagged mainly for a request to call, text or click (a phone number, “call”), backed
  up by a prize or reward offer (“claim”, “WON”) and money or premium-rate pricing…"
- **What would change the decision:** the table removes 8 words one at a time. The score only
  starts to fall once the evidence thins out, and crosses the line at the end (64% → **48%**).
  Point out: every number is a fresh prediction on the edited message, not an estimate. The
  message is suspicious for many reasons at once, not because of a single word.

## 3. Disguised spam (45 s)

Click **Disguised spam** (`C0ngratulati0ns!! … fr3e iPh0ne … cl@im`), then **Analyze**.

- Still **Suspicious, >99%**: MATRIX reads `fr3e` as "free" and `cl@im` as "claim".
- The highlighting points at the disguised words exactly where they were typed.
- Counterfactual flips after 4 removals: short code → `fr3e` → "Reply" → `cl@im` (**47%**).
- Open the **Robustness check** tab: none of the edits MATRIX tries (more disguises, capitals,
  an added greeting…) changes the verdict.

## 4. Honest uncertainty (60 s)

Click **Delivery link**, then **Analyze**.

- Amber **Leans legitimate, 45%**, uncertainty **high**, with the note that the score is
  borderline. This is a modern parcel scam, and the model (trained on 2000s UK SMS) is not sure.
  MATRIX says so instead of hiding it.
- **Why:** "Not flagged, but close to the line…"; the web link and "address" are weak spam signals.
- Robustness check tab: a single typo changes this verdict. That fragility is exactly what the
  borderline warning predicts.

Optional: type a message in another language, e.g. `Hallo, kannst du mich morgen anrufen? Danke!`.
The result is "Looks legitimate", but uncertainty is **high**: "Only 0 of 7 words were seen in
training".

## 5. A normal message that looks risky (30 s)

Click **Tricky normal** ("Call me when you're free, I won the bet so dinner is on you tomorrow!").

- **Looks legitimate, 2%** even though it contains "call", "free" and "won": the context words
  ("when", "dinner", "tomorrow") outweigh them, and the reason says so.
- Switch the highlight to **LIME** and open **SHAP vs LIME**: both methods agree on all of
  their top words.

## 6. Analyst dashboard (60 s)

Open **Analyst dashboard**.

- **Performance:** confusion matrix (122 caught, 9 missed, 903 passed, 1 wrongly flagged) and
  the calibration chart (calibration error 0.0167 → 0.0039).
- **Error explorer:** open the one wrongly flagged normal message (an "added you as a contact…
  send FREE SMS" notice, 85%) and show why it was flagged.
- **Explanation quality:** removing the top 3 SHAP words lowers the score by 1.49 (log-odds),
  versus 0.16 for 3 random words, so the explanations point at words the model really uses.
  LIME and SHAP share 84% of their top-5 words.
- **Robustness:** with de-obfuscation, the explanation keeps its top words after leetspeak (100%);
  without it, only 77%.

## Closing line (10 s)

"Every threat leaves a trace, and MATRIX shows you where it is, how sure it is, and what
would change its mind."

## If something goes wrong

- *"The model is not ready"*: run `python -m src.train`, then restart the app.
- The first visit to the dashboard takes a few seconds while it prepares the error explorer.
