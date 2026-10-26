# Builds the git history in logical steps. Run once from the project folder
# after "git init" and after .gitignore is in place:
#   powershell -ExecutionPolicy Bypass -File .\make_commits.ps1
$ErrorActionPreference = "Stop"

function Commit($message, [string[]]$paths) {
    foreach ($p in $paths) { if (Test-Path $p) { git add $p } }
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { git commit -q -m $message; Write-Host "committed: $message" }
}

Commit "Add gitignore" @(".gitignore")
Commit "Add project README and requirements" @("README.md", "requirements.txt")
Commit "Add Kenya facility data and KMPDC scraping and merge scripts" @("data/healthcare_facilities.csv", "data/healthcare_facilities_backup.csv", "data/healthcare_facilities_enriched.csv", "data/kmpdc_public_facilities.csv", "data/kmpdc_private_facilities.csv", "data/kmpdc_all_facilities.csv", "data/scrape_kmpdc.py", "data/merge_kmpdc_with_coords.py", "SCRAPE_STATUS.txt")
Commit "Add symptom datasets and the script that builds the augmented set" @("data/dataset_real_kaggle_raw.csv", "data/dataset.csv", "data/generate_augmented_dataset.py")
Commit "Add disease to KEPH level mapping" @("data/disease_keph_mapping.csv", "data/build_keph_mapping_real.py")
Commit "Add script to load facilities into SQLite" @("backend/init_db.py", "backend/database.sqlite")
Commit "Add symptom vectorizer and model training script" @("ML/train_model.py")
Commit "Add straight-line distance and facility ranking" @("backend/geo.py")
Commit "Add road-distance routing with straight-line fallback" @("backend/road_distance.py")
Commit "Add emergency red-flag rules" @("backend/red_flags.py")
Commit "Add age, pregnancy and severity adjustments to care level" @("backend/patient_context.py")
Commit "Add prediction log without personal data" @("backend/prediction_log.py")
Commit "Add security headers, size limit and safe error handling" @("backend/security.py")
Commit "Add FastAPI app with predict, nearest-facilities and triage endpoints" @("backend/main.py")
Commit "Add saved model files" @("ML/disease_classifier.joblib", "ML/symptom_features.joblib", "ML/disease_keph_mapping.joblib")
Commit "Add model evaluation script and results" @("ML/evaluate_model.py", "ML/evaluation/metrics.json", "ML/evaluation/classification_report.txt", "ML/evaluation/confusion_matrix.png", "ML/evaluation/cv_scores.png", "ML/evaluation/feature_importance.png", "ML/evaluation/train_vs_test_accuracy.png")
Commit "Compare Logistic Regression, Random Forest and Gradient Boosting" @("ML/evaluate_model_comparison.py", "ML/evaluation/model_comparison.json")
Commit "Add duplicate and robustness checks on the raw Kaggle data" @("ML/evaluate_deduplicated.py", "ML/evaluate_robustness.py", "ML/evaluation/deduplicated_metrics.json", "ML/evaluation/deduplicated_classification_report.txt", "ML/evaluation/deduplicated_confusion_matrix.png", "ML/evaluation/robustness_metrics.json", "ML/evaluation/robustness_curve.png")
Commit "Add extended model search (C, Naive Bayes, SVM, MLP, ensembles)" @("ML/extended_search")
Commit "Add web interface" @("frontend/index.html", "frontend/images", "frontend/fonts")
Commit "Add unit tests for distance and symptom encoding" @("backend/tests/conftest.py", "backend/tests/test_geo.py", "backend/tests/test_vectorizer.py")
Commit "Add tests for red flags, care level rules and routing fallback" @("backend/tests/test_red_flags.py", "backend/tests/test_patient_context.py", "backend/tests/test_road_distance.py", "backend/tests/test_rules_vocabulary.py")
Commit "Add API and security tests" @("backend/tests/test_api.py")
Commit "Add model development notebook" @("notebook/Healthcare_Model_Development.ipynb", "notebook/README.txt", "notebook/dataset.csv", "notebook/dataset_real_kaggle_raw.csv", "notebook/train_model.py")
Commit "Add data exploration and access-gap analysis notebook with charts" @("notebook/Data_Exploration_and_Analysis.ipynb", "ML/evaluation/story")
Commit "Add test plan" @("docs")
Commit "Add commit helper script" @("make_commits.ps1")

git log --oneline
