
# ═══════════════════════════════════════════════════════════════════════════════
# DATA SCIENCE ARCHITECTURE AUDIT REPORT
# SaaS User Retention Pipeline — 13-Block Canvas Review
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd

DIVIDER  = "═" * 80
SECTION  = "─" * 80
SUBSEC   = "·" * 60

print(DIVIDER)
print("  DATA SCIENCE ARCHITECTURE AUDIT REPORT")
print("  SaaS User Retention Pipeline | 13-Block Canvas")
print("  Analyst: Zerve AI Agent  |  Scope: Full Pipeline Review")
print(DIVIDER)

# ───────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PIPELINE ALIGNMENT
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SECTION 1 — PIPELINE ALIGNMENT & SEQUENCE VALIDATION")
print(f"{'━'*80}\n")

print("Declared Pipeline:")
print("  Example Dataset → User Retention Analysis → User Retention Labeling →")
print("  User Feature Table → Feature Scaling → XGBoost Retention Classifier →")
print("  [Retention Probability Scatter | False Positive Analysis | SHAP Analysis]")
print("  └→ Business Recommendation")
print("  (Parallel: Tool Frequency by Retention Group, Behavioral Consistency &")
print("   Habit Engineering → Correlation Heatmap)\n")

print("Sequence Assessment:")
print("  ✅  Cleaning  (User Retention Analysis):  Missing value removal (>70% thresh)")
print("      Applied BEFORE feature engineering — CORRECT.")
print("  ✅  Labeling  (User Retention Labeling):  is_retained computed from")
print("      retention_clean, then label merged back on event rows — CORRECT order.")
print("  ✅  Feature Engineering (User Feature Table): All 9 features built from")
print("      raw events. Label already exists at this point — features computed")
print("      independently of the label — CORRECT.")
print("  ✅  Scaling  (Feature Scaling): StandardScaler applied after features are")
print("      assembled. No data re-ingestion or label touch — CORRECT.")
print("  ✅  Modeling (XGBoost Retention Classifier): Stratified 80/20 split on the")
print("      scaled user-level table. GBM trained only on X_train — CORRECT.")
print()

# ───────────────────────────────────────────────────────────────────────────────
# SECTION 2 — LEAKY LOGIC RISKS
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SECTION 2 — LEAKY LOGIC RISKS (Future Information Contamination)")
print(f"{'━'*80}\n")

print("🔴  RISK 1 — LABEL COMPUTED OVER FULL OBSERVATION WINDOW (HIGH SEVERITY)")
print(SUBSEC)
print("  Location:  User Retention Labeling & User Feature Table")
print("  Definition: is_retained = 1 if distinct_iso_weeks >= 3, computed over")
print("              ALL events in the dataset (up to 13 weeks).")
print()
print("  Critical question: Are ALL features computed using only pre-label events,")
print("  or do some features use data from weeks 4–13 that contribute to the label?")
print()
print("  Finding: The label window IS the full data window. Features like")
print("  'total_events' and 'days_since_first_event' are also computed over the")
print("  full window. This means a user who accumulates events in weeks 4–13")
print("  will have high total_events AND be labeled as retained — creating a")
print("  PERFECT CORRELATION by construction, not by predictive value.")
print()
print("  Specifically leaky features:")
print("  ⚠️  total_events (log1p)  — counts events across the ENTIRE window,")
print("      including weeks that DEFINE the retention label.")
print("  ⚠️  days_since_first_event — spans the full observation period,")
print("      correlated with 'was user observed for a long time' = survives.")
print("  ⚠️  unique_tools_used      — uses all tool events, including post-week-3.")
print("  ⚠️  agent_usage_ratio      — tool ratio over entire window.")
print("  ⚠️  exploration_index      — derived from unique_tools_used / total_events,")
print("      both of which span the full window.")
print()
print("  Non-leaky features (genuinely predictive — captured BEFORE outcome):")
print("  ✅  first_week_events   — strictly first 7 days after signup.")
print("  ✅  consistency_score   — distinct active days in first 7 days.")
print("  ✅  first_24h_events    — first 24 hours only.")
print()
print("  ROOT CAUSE: There is no 'feature cutoff date' distinct from the")
print("  'label observation end date'. In a proper temporal pipeline, you would:")
print("    1. Define a feature window (e.g., days 1–14 post-signup)")
print("    2. Define a label window (e.g., days 15–90 post-signup)")
print("    3. Features must be computed BEFORE the label window opens.")
print()
print("  CONSEQUENCE: The model may appear predictive but is partially measuring")
print("  activity DURING the period it claims to predict. On new users who have")
print("  only 2 weeks of data, 'total_events' will be much lower — the deployed")
print("  model will systematically underestimate retention probability for recent")
print("  sign-ups, causing recall to collapse in production.")
print()

print("🟡  RISK 2 — SCALER FITTED ON FULL DATASET BEFORE SPLIT (MEDIUM SEVERITY)")
print(SUBSEC)
print("  Location:  Feature Scaling block (StandardScaler)")
print("  Issue:     StandardScaler.fit_transform() is called on ALL 5,410 users")
print("             BEFORE the train/test split. The scaler thus 'sees' test-set")
print("             statistics (mean, std) during fitting.")
print("  Impact:    Minor in this case (scaler stats won't differ dramatically)")
print("             but technically violates evaluation integrity. Correct fix:")
print("             scaler.fit(X_train) → scaler.transform(X_train) and X_test.")
print()

print("🟡  RISK 3 — SHAP MODEL RETRAINED IN-BLOCK ON FULL FEATURE MATRIX (MEDIUM)")
print(SUBSEC)
print("  Location:  SHAP Analysis block")
print("  Issue:     SHAP block retrains a FRESH model on ALL 5,410 users, then")
print("             computes SHAP values on the same 5,410 users used for training.")
print("             This is a training-set SHAP explanation, not test-set SHAP.")
print("             Model may memorise retained-user patterns → inflated SHAP")
print("             magnitudes for rare class features.")
print()

print("🟡  RISK 4 — FALSE POSITIVE BLOCK USES DIFFERENT FEATURE SET (MEDIUM)")
print(SUBSEC)
print("  Location:  False Positive Analysis block")
print("  Issue:     This block uses 5 features (log_total_events,")
print("             log_days_since_first_event, agent_usage_ratio,")
print("             unique_tools_used, first_week_events) — a DIFFERENT set from")
print("             the main classifier's 7 features. It also uses a column name")
print("             'log_days_since_first_event' which does not exist in the main")
print("             pipeline (main pipeline drops days_since). This block rebuilds")
print("             the entire pipeline from scratch, meaning its FP analysis")
print("             is not from the same model as the main classifier. Results")
print("             are from a DIFFERENT model trained on DIFFERENT features.")
print()

# ───────────────────────────────────────────────────────────────────────────────
# SECTION 3 — STRUCTURAL GAPS
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SECTION 3 — STRUCTURAL GAPS (What a Senior DS Would Expect)")
print(f"{'━'*80}\n")

gaps = [
    ("❌ MISSING", "Temporal Train/Test Split",
     "Split should respect time: e.g., train on cohorts signed up before month M,\n"
     "      test on cohorts signed up after M. Random split allows future-cohort users\n"
     "      into training (additional label leakage for time-sensitive features)."),
    ("❌ MISSING", "Cross-Validation Strategy",
     "With only 138 positive examples, a single 80/20 split gives only ~28 retained\n"
     "      users in the test set — extremely noisy. Stratified k-fold (k=5) would give\n"
     "      5× more stable recall estimates. Current recall confidence interval is ±15pp."),
    ("❌ MISSING", "Baseline Model",
     "No majority-class baseline, logistic regression, or rule-based threshold is\n"
     "      established. Without a baseline, we cannot claim GBM adds value over simpler\n"
     "      approaches. 'first_week_events > N' alone may achieve comparable recall."),
    ("❌ MISSING", "Outlier Detection / Treatment",
     "first_24h_events max = 2,693. This extreme outlier (likely a bot/stress-test)\n"
     "      will dominate StandardScaler mean/std and distort feature distributions.\n"
     "      No IQR capping, Winsorization, or outlier exclusion is performed."),
    ("❌ MISSING", "Class Imbalance Audit",
     "38:1 imbalance is noted and handled via sample_weight but there is no:\n"
     "      a) SMOTE/ADASYN alternative evaluated,\n"
     "      b) Threshold optimization (default 0.5 threshold likely suboptimal),\n"
     "      c) Precision-Recall AUC (PR-AUC) reported — the correct metric for imbalance,\n"
     "      d) ROC-AUC reported or compared."),
    ("❌ MISSING", "Probability Calibration",
     "GBM probabilities are used directly as 'retention probability' in the scatter\n"
     "      plot. However, the 50%/80% threshold values output as '1 event' — a clear\n"
     "      sign the probabilities are UNCALIBRATED. Platt scaling or isotonic regression\n"
     "      would fix this so probability scores are interpretable and usable operationally."),
    ("❌ MISSING", "Held-Out Temporal Validation Set",
     "No separate holdout cohort. The pipeline has no way to estimate how the model\n"
     "      would perform on users who signed up THIS WEEK, when the retention label\n"
     "      cannot yet be computed."),
    ("❌ MISSING", "Feature Correlation Deduplication",
     "first_week_events and first_24h_events are likely highly correlated (r may\n"
     "      be >0.8). consistency_score and first_week_events are also correlated.\n"
     "      No VIF analysis or feature selection step is run to remove multicollinearity."),
    ("❌ MISSING", "Model Versioning / Reproducibility Checkpoint",
     "gbm_model.pkl is overwritten every run. No versioning, experiment tracking\n"
     "      (MLflow/W&B), or hash of training data is stored."),
    ("⚠️  WEAK",   "Retention Label Definition Robustness",
     "WEEK_THRESH = 3 is set without justification or sensitivity analysis.\n"
     "      No comparison of thresholds (2-week, 4-week, 30-day cohort) is shown.\n"
     "      Label stability test: if WEEK_THRESH changed to 2, how many labels flip?"),
]

for _severity, _gap_name, _detail in gaps:
    print(f"  {_severity}:  {_gap_name}")
    print(f"      {_detail}")
    print()

# ───────────────────────────────────────────────────────────────────────────────
# SECTION 4 — RESULTS INTEGRITY
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SECTION 4 — RESULTS INTEGRITY: DO OUTPUTS ACTUALLY ANSWER 'WHY USERS STAY'?")
print(f"{'━'*80}\n")

outputs = {
    "Tool Frequency by Retention Group (Grouped Bar)": {
        "verdict": "DESCRIPTIVE — partially answers the question",
        "analysis": (
            "Shows that retained vs churned users have different tool usage distributions.\n"
            "      This IS directionally useful — if Coder Agent appears more in retained users,\n"
            "      it suggests a causal hypothesis. HOWEVER:\n"
            "      - No statistical significance test (chi-squared, permutation test)\n"
            "      - Counts are NOT normalized by total user events, so heavier users\n"
            "        (who are also retained) trivially dominate tool event counts\n"
            "      - Answers 'which tools correlate with retention' not 'why users stay'"
        )
    },
    "Correlation Heatmap": {
        "verdict": "DESCRIPTIVE — measures association, not causation",
        "analysis": (
            "Pearson r measures linear association. consistency_score ↔ is_retained r≈0.39\n"
            "      is the strongest signal. PROBLEMS:\n"
            "      - Pearson r on a binary target (is_retained) is point-biserial correlation\n"
            "        and is valid but low r values are EXPECTED with 2.5% base rate\n"
            "      - Does NOT control for confounders (tenure, acquisition channel)\n"
            "      - Does NOT answer 'why' — only 'what correlates with what'\n"
            "      - total_events ↔ is_retained r is artificially inflated by window leakage"
        )
    },
    "Feature Importance (Gain-Based)": {
        "verdict": "PARTIALLY ANSWERS — reveals model's decision logic, not ground truth",
        "analysis": (
            "Gain-based importance shows which features GBM splits on most. This is\n"
            "      model-intrinsic and can be misleading with correlated features.\n"
            "      - If total_events and first_week_events are correlated, importance\n"
            "        is split arbitrarily between them\n"
            "      - Gain does NOT represent causal effect sizes\n"
            "      - Because total_events uses full window, top importance may simply\n"
            "        be recapitulating the label (see Leaky Logic Risk 1)"
        )
    },
    "Confusion Matrix": {
        "verdict": "EVALUATION OUTPUT — necessary but incomplete",
        "analysis": (
            "Correctly shows TP/FP/FN/TN on held-out test set. Recall ~60-78% is\n"
            "      reported (variance between blocks due to different feature sets).\n"
            "      MISSING: PR-AUC, ROC-AUC, calibration curve, threshold sensitivity"
        )
    },
    "SHAP Analysis (Beeswarm)": {
        "verdict": "BEST BLOCK FOR 'WHY' — but methodologically compromised",
        "analysis": (
            "SHAP is the right tool for answering 'why does the model predict retention'.\n"
            "      The beeswarm correctly shows direction (high feature value → positive SHAP).\n"
            "      PROBLEMS:\n"
            "      - SHAP computed on training data (all 5,410 users) not held-out test set\n"
            "      - Custom tree-path decomposition (not the official SHAP library) — may\n"
            "        not match official TreeSHAP values exactly\n"
            "      - Because leaky features (total_events) dominate, SHAP tells us 'users\n"
            "        with more total events are retained' — which partially defines the label\n"
            "      - Still valuable for the non-leaky features (first_week_events,\n"
            "        consistency_score, first_24h_events)"
        )
    },
    "Retention Probability Scatter": {
        "verdict": "BROKEN — threshold values are meaningless",
        "analysis": (
            "CRITICAL: Both threshold_50 and threshold_80 output as '1 event'.\n"
            "      This means the model assigns >50% retention probability to users with\n"
            "      just 1 event — impossible with only 2.5% base rate unless the model\n"
            "      is uncalibrated or the feature (raw_total_events from user_feature_table)\n"
            "      is incorrectly mixed with the log-transformed training features.\n"
            "      The scatter plot is not a reliable guide to 'how many events = retained'."
        )
    },
    "Business Recommendation (Markdown)": {
        "verdict": "CONCEPTUALLY SOUND — but numbers need verification",
        "analysis": (
            "The recall > accuracy framing is correct and well-argued. The EV calculation\n"
            "      is logically structured. RISKS:\n"
            "      - Recall numbers cited (78%) differ from classifier output (varies by run)\n"
            "      - Assumes $1,500 LTV and 40% re-engagement rate — both unvalidated\n"
            "      - Precision of 32% would mean ~338 flagged users but the model is\n"
            "        calibration-broken (threshold_50 = 1 event) so operational list\n"
            "        size estimates are unreliable\n"
            "      - Block status is PENDING (Status 1) — was never run"
        )
    },
    "False Positive Analysis": {
        "verdict": "BROKEN — uses a different model than the main classifier",
        "analysis": (
            "This block rebuilds the entire pipeline with a 5-feature model\n"
            "      (including log_days_since_first_event which the main pipeline drops)\n"
            "      and reports FP analysis from THAT model's results. The 13 FPs\n"
            "      identified are NOT the same FPs as the main GBM model produces.\n"
            "      The block is conceptually useful but practically disconnected."
        )
    },
}

for _block_name, _info in outputs.items():
    print(f"  Block: {_block_name}")
    print(f"  Verdict: {_info['verdict']}")
    print(f"  Analysis: {_info['analysis']}")
    print()

# ───────────────────────────────────────────────────────────────────────────────
# SECTION 5 — KEEP / KILL / COMBINE
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SECTION 5 — KEEP / KILL / COMBINE RECOMMENDATIONS")
print(f"{'━'*80}\n")

recommendations = [
    ("Example Dataset",
     "KEEP",
     "Correctly loads raw event data. Simple, focused. Keep as-is."),

    ("User Retention Analysis",
     "KEEP + REFACTOR",
     "Correct cleaning logic (>70% missing drop). Refactor to: (a) separate the\n"
     "      top-user EDA into its own block, (b) extract retention_clean as the only\n"
     "      output, (c) document the 88-column drop decision more clearly."),

    ("User Retention Labeling",
     "COMBINE → into User Feature Table",
     "This block computes is_retained correctly, but User Feature Table re-computes\n"
     "      the EXACT same logic from scratch. The label computation is duplicated.\n"
     "      Combine by making User Feature Table the single source for labeling and\n"
     "      features. Retain Labeling block only if it serves as an observable QA step."),

    ("User Feature Table",
     "KEEP + REFACTOR (CRITICAL)",
     "Central feature hub — right idea. MUST be refactored to:\n"
     "      (a) Define a strict feature cutoff date (e.g., days 1–14 post-signup).\n"
     "      (b) Define a label window that starts AFTER the feature window closes.\n"
     "      (c) Remove total_events and days_since_first_event from the model feature\n"
     "          set (keep for EDA only) — or recompute them using only the feature window.\n"
     "      (d) Add IQR-capping for outliers (first_24h_events max=2,693)."),

    ("Behavioral Consistency and Habit Engineering",
     "KILL / MERGE",
     "This is a pass-through validation block. All its features are already in\n"
     "      User Feature Table. It adds no computation, only an assertion. Remove it\n"
     "      and fold the assertion into User Feature Table or Feature Scaling."),

    ("Feature Scaling",
     "KEEP + REFACTOR",
     "Correct separation of scaling from feature engineering. MUST fix:\n"
     "      Scaler should be fit ONLY on X_train after the split, not on all 5,410 users.\n"
     "      Export fitted scaler as variable for downstream blocks (already done via pkl)."),

    ("Tool Frequency by Retention Group",
     "KEEP + ENHANCE",
     "Good EDA block with correct grouping. Enhance by:\n"
     "      (a) Normalizing by user-level event count (events per user, not raw counts)\n"
     "      (b) Adding chi-squared test for tool × retention_group independence\n"
     "      (c) Repositioning in the DAG as an EDA branch (not in the modeling path)."),

    ("Correlation Heatmap",
     "KEEP + ENHANCE",
     "Visually clear and technically valid for exploratory purposes. Enhance by:\n"
     "      (a) Noting that Pearson r on binary target is point-biserial\n"
     "      (b) Adding Spearman r for robustness check\n"
     "      (c) Flagging the leaky features visually."),

    ("XGBoost Retention Classifier",
     "KEEP + REFACTOR (CRITICAL)",
     "Core modeling block — good architecture with stratified split and class weighting.\n"
     "      MUST fix:\n"
     "      (a) Remove total_events (leaky) from feature set\n"
     "      (b) Move StandardScaler.fit() INSIDE this block after split (or pass fitted\n"
     "          scaler from Feature Scaling)\n"
     "      (c) Add PR-AUC and ROC-AUC to evaluation metrics\n"
     "      (d) Add threshold optimization (PR curve peak)\n"
     "      (e) Add 5-fold stratified CV for stable recall estimate\n"
     "      (f) Add a majority-class baseline comparison."),

    ("Retention Probability Scatter",
     "KILL / REBUILD",
     "Currently broken — threshold_50=1, threshold_80=1 is nonsensical.\n"
     "      The confusion between log-transformed 'total_events' (from user_feature_table)\n"
     "      and the raw event count corrupts the x-axis. If rebuilt, must:\n"
     "      (a) Use calibrated probabilities\n"
     "      (b) Use a meaningful x-axis feature (e.g., first_week_events — uncorrupted)\n"
     "      (c) Only score X_test users, not the full dataset."),

    ("False Positive Analysis",
     "KILL / REBUILD",
     "Uses a different model (5 features vs 7), different column names, and rebuilds\n"
     "      the pipeline from scratch — results do not reflect the canonical model.\n"
     "      Rebuild as a downstream block from XGBoost Retention Classifier that uses\n"
     "      the SAME X_test, y_test, and predictions already in scope."),

    ("SHAP Analysis",
     "KEEP + REFACTOR",
     "Best block for causality narrative. MUST fix:\n"
     "      (a) Compute SHAP on X_test only (not full 5,410 users)\n"
     "      (b) Use official SHAP library (shap.TreeExplainer) for correctness\n"
     "      (c) Remove or flag leaky features from the SHAP display\n"
     "      (d) Add SHAP waterfall for a single 'typical retained user' as a case study."),

    ("Business Recommendation (Markdown)",
     "KEEP + UPDATE",
     "Well-structured business framing. Update to:\n"
     "      (a) Run the block (status is PENDING)\n"
     "      (b) Revise numbers once model is rebuilt with non-leaky features\n"
     "      (c) Add sensitivity analysis for LTV and re-engagement rate assumptions."),
]

for _rec_block, _action, _justification in recommendations:
    print(f"  Block:       {_rec_block}")
    print(f"  Action:      {_action}")
    print(f"  Rationale:   {_justification}")
    print()

# ───────────────────────────────────────────────────────────────────────────────
# SECTION 6 — IDEAL PIPELINE PROPOSAL
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SECTION 6 — IDEAL PIPELINE: PROPOSED RE-ORDERED & EXPANDED BLOCK SEQUENCE")
print(f"{'━'*80}\n")

print("  PHASE 0 — DATA INGESTION & EDA")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 01: Raw Data Load")
print("  │    Purpose: Load parquet, assert schema, print dtypes + row counts.")
print("  │")
print("  │  Block 02: Exploratory Data Analysis")
print("  │    Purpose: Missing value analysis, event type distribution,")
print("  │    user tenure distribution, tool usage overview.")
print("  │    (Combines: User Retention Analysis EDA + top-user table)")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("  PHASE 1 — TEMPORAL DESIGN & LABELING (NO LEAKAGE)")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 03: Cohort Definition & Temporal Split")
print("  │    Purpose: Define feature window (days 1-14 post-signup) and label")
print("  │    window (days 15-90 post-signup). Create cohort_feature_events and")
print("  │    cohort_label_events DataFrames. Exclude users with < 2 weeks data.")
print("  │")
print("  │  Block 04: Retention Label Engineering")
print("  │    Purpose: Compute is_retained from cohort_label_events only.")
print("  │    Include: sensitivity analysis for WEEK_THRESH in {2, 3, 4}.")
print("  │    Output: retention_label Series (user-level)")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("  PHASE 2 — FEATURE ENGINEERING (FEATURE WINDOW ONLY)")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 05: User Feature Engineering")
print("  │    Purpose: Compute all features from cohort_feature_events ONLY.")
print("  │    Features:")
print("  │    - first_24h_events, first_week_events (non-leaky, keep)")
print("  │    - consistency_score (distinct active days, days 1-14)")
print("  │    - unique_tools_used (days 1-14 only)")
print("  │    - agent_usage_ratio (days 1-14 only)")
print("  │    - event_velocity_trend (linear slope of daily events, days 1-14)")
print("  │    - NEW: days_to_second_session (time between session 1 and 2)")
print("  │    - NEW: tool_diversity_ratio (tools tried / total tool events)")
print("  │    Add: IQR-capping / Winsorization for skewed features.")
print("  │    Drop: total_events (full window), days_since_first_event (full window)")
print("  │")
print("  │  Block 06: Tool Usage EDA (Exploratory, parallel branch)")
print("  │    Purpose: Grouped bar chart of tool frequency by retention group.")
print("  │    Add: Normalised rates + chi-squared test.")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("  PHASE 3 — PREPROCESSING & VALIDATION")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 07: Feature Validation & Correlation Analysis")
print("  │    Purpose: Pearson + Spearman correlation heatmap, VIF analysis,")
print("  │    outlier summary table, feature completeness audit.")
print("  │    (Combines: Correlation Heatmap + Behavioral Consistency validation)")
print("  │")
print("  │  Block 08: Train/Test Split & Scaling")
print("  │    Purpose: Stratified temporal split (train=older cohorts,")
print("  │    test=newer cohorts). Fit StandardScaler on X_train only.")
print("  │    Export: X_train_scaled, X_test_scaled, y_train, y_test, scaler.")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("  PHASE 4 — MODELING")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 09: Baseline Model")
print("  │    Purpose: Majority-class baseline + LogisticRegression +")
print("  │    rule-based threshold (first_week_events > N).")
print("  │    Metrics: recall, precision, PR-AUC on test set.")
print("  │")
print("  │  Block 10: GBM Retention Classifier (Main Model)")
print("  │    Purpose: GradientBoostingClassifier with 5-fold stratified CV.")
print("  │    Add: Threshold optimization via PR-curve, calibration via Platt scaling.")
print("  │    Metrics: PR-AUC, ROC-AUC, recall @ optimal threshold, calibration curve.")
print("  │    (Refactored from: XGBoost Retention Classifier)")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("  PHASE 5 — EVALUATION & EXPLAINABILITY")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 11: Model Evaluation")
print("  │    Purpose: Confusion matrix on X_test, PR-AUC curve, ROC curve,")
print("  │    calibration plot, baseline comparison table.")
print("  │    (Replaces: Retention Probability Scatter + False Positive Analysis)")
print("  │")
print("  │  Block 12: SHAP Explainability")
print("  │    Purpose: Official shap.TreeExplainer on X_test only.")
print("  │    Outputs: Beeswarm (global), waterfall (single user), dependence plots.")
print("  │    (Refactored from: SHAP Analysis)")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()
print("  PHASE 6 — BUSINESS OUTPUT")
print("  ┌─────────────────────────────────────────────────────────────────────┐")
print("  │  Block 13: Business Recommendation")
print("  │    Purpose: Updated EV calculation, recall vs accuracy framing,")
print("  │    actionable segment export (top users by calibrated probability).")
print("  │    (Refactored from: Business Recommendation Markdown)")
print("  └─────────────────────────────────────────────────────────────────────┘")
print()

# ───────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ───────────────────────────────────────────────────────────────────────────────
print(f"\n{'━'*80}")
print("  SUMMARY TABLE — ALL 13 BLOCKS")
print(f"{'━'*80}\n")

summary_data = {
    "Block Name": [
        "Example Dataset",
        "User Retention Analysis",
        "User Retention Labeling",
        "User Feature Table",
        "Behavioral Consistency & Habit Eng.",
        "Feature Scaling",
        "Tool Frequency by Retention Group",
        "Correlation Heatmap",
        "XGBoost Retention Classifier",
        "Retention Probability Scatter",
        "False Positive Analysis",
        "SHAP Analysis",
        "Business Recommendation",
    ],
    "Status": [
        "✅ Pass","✅ Pass","⚠️ Duplicate","🔴 Leaky","🔴 Kill",
        "⚠️ Scaler Leak","✅ Partial","✅ Partial","⚠️ Leaky",
        "🔴 Broken","🔴 Wrong Model","⚠️ Train SHAP","⚠️ Pending"
    ],
    "Action": [
        "KEEP","KEEP + SPLIT","COMBINE → UFT","REFACTOR CRITICAL","KILL",
        "REFACTOR","KEEP + ENHANCE","KEEP + ENHANCE","REFACTOR CRITICAL",
        "KILL / REBUILD","KILL / REBUILD","REFACTOR","RUN + UPDATE"
    ],
    "Key Risk": [
        "None","EDA mixed with cleaning","Duplicates UFT label logic",
        "5/7 features use full obs window","Purely pass-through — no value",
        "Scaler fit before split","Counts not normalized per user",
        "Pearson only, no significance","Leaky total_events in features",
        "Thresholds both = 1 event (uncalibrated)","Different model/features than main",
        "SHAP on training data, custom impl.","Never executed (Status=PENDING)"
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.index = range(1, 14)
summary_df.index.name = "#"

# Print as clean table
col_widths = {
    "Block Name": 38,
    "Status": 22,
    "Action": 22,
    "Key Risk": 42,
}
header = (
    f"{'#':<3} "
    f"{'Block Name':<38}  "
    f"{'Status':<22}  "
    f"{'Action':<22}  "
    f"{'Key Risk':<42}"
)
print(header)
print("─" * len(header))
for _audit_idx, _audit_row in summary_df.iterrows():
    print(
        f"{_audit_idx:<3} "
        f"{_audit_row['Block Name']:<38}  "
        f"{_audit_row['Status']:<22}  "
        f"{_audit_row['Action']:<22}  "
        f"{_audit_row['Key Risk']:<42}"
    )

print(f"\n\n{'━'*80}")
print("  AUDIT COMPLETE")
print(f"{'━'*80}")
print()
print("  HEADLINE FINDINGS:")
print()
print("  1. 🔴 CONFIRMED LEAKY LOGIC: 5 of 7 model features (total_events,")
print("     days_since_first_event, unique_tools_used, agent_usage_ratio,")
print("     exploration_index) are computed over the SAME time window used to")
print("     define the retention label. The model partially describes its own")
print("     label, inflating apparent performance.")
print()
print("  2. 🔴 BROKEN OUTPUTS: Retention Probability Scatter has threshold_50=1")
print("     and threshold_80=1 event — both values are artefacts of uncalibrated")
print("     probabilities and feature-scale confusion. False Positive Analysis")
print("     silently uses a different model than the main classifier.")
print()
print("  3. ⚠️  MISSING FUNDAMENTALS: No cross-validation (only single 80/20 split"),
print("     with 28 positives in test — recall is ±15pp unstable), no baseline"),
print("     model, no PR-AUC, no probability calibration, no outlier treatment.")
print()
print("  4. ✅ WHAT WORKS: The temporal feature trio (first_24h_events,"),
print("     first_week_events, consistency_score) is genuinely non-leaky and"),
print("     causally interpretable. The recall-first framing in Business"),
print("     Recommendation is economically correct. The SHAP beeswarm is the"),
print("     right analytical tool — it just needs to run on test data.")
print()
print("  5. ✅ PROPOSED IDEAL PIPELINE: 13-block sequence with clean temporal"),
print("     feature/label separation, stratified temporal split, CV, baseline,"),
print("     calibration, and official SHAP — retaining 11 of the current 13"),
print("     blocks in restructured form.")
print(DIVIDER)
