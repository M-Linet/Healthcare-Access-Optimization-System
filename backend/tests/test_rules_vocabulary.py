"""UT-07: every symptom name used in the safety rules must exist in the model's symptom list."""
import os
import joblib

from red_flags import RED_FLAG_RULES, SINGLE_SYMPTOM_FLAGS
from patient_context import PREGNANCY_RELATED_SYMPTOMS

FEATURES = os.path.join(os.path.dirname(__file__), "..", "..", "ML", "symptom_features.joblib")


def test_all_rule_symptoms_are_in_the_model_vocabulary():
    vocab = set(joblib.load(FEATURES))
    used = set(SINGLE_SYMPTOM_FLAGS) | set(PREGNANCY_RELATED_SYMPTOMS)
    for rule in RED_FLAG_RULES:
        used |= rule["symptoms"]
    missing = sorted(used - vocab)
    assert not missing, f"Rules use symptoms the model does not know: {missing}"
