"""
red_flags.py
-------------
Emergency red-flag detection. Runs BEFORE the ML prediction is trusted at
face value. If a dangerous symptom combination is present, the system should
short-circuit the normal "here's a ranked list of facilities" flow and
instead surface an urgent warning, because a probabilistic disease
classifier is the wrong tool to be quietly routing a possible heart attack
or stroke through the standard triage path.

This is intentionally a simple rule-based layer, not ML-based: emergency
detection needs to be predictable, auditable, and high-recall, not a
black box. Each rule is a set of symptoms; if ALL symptoms in a rule are
present in the user's reported list, the rule fires.
"""

from typing import List, Dict

EMERGENCY_NUMBER_KE = "999 / 112"

# Each rule: (label, required_symptoms, advice)
# Symptom names must match tokens used in data/dataset.csv / symptom_features.joblib.
# tests/test_rules_vocabulary.py checks this automatically.
# These rules are a simple safety layer and have not been clinically approved.
RED_FLAG_RULES = [
    {
        "label": "Possible heart attack",
        "symptoms": {"chest_pain", "breathlessness", "sweating"},
        "advice": "Chest pain with breathlessness and sweating can indicate a heart attack. "
                  "Call emergency services or get to the nearest hospital immediately. Do not drive yourself.",
    },
    {
        "label": "Possible heart attack",
        "symptoms": {"chest_pain", "fast_heart_rate"},
        "advice": "Chest pain with a fast heart rate can indicate a heart problem. "
                  "Call emergency services or get to the nearest hospital immediately.",
    },
    {
        "label": "Possible stroke",
        "symptoms": {"slurred_speech", "weakness_in_limbs"},
        "advice": "Slurred speech with limb weakness can indicate a stroke. Time-critical — "
                  "get to the nearest hospital with emergency capacity immediately.",
    },
    {
        "label": "Possible stroke",
        "symptoms": {"slurred_speech", "weakness_of_one_body_side"},
        "advice": "Slurred speech with weakness on one side of the body can indicate a stroke. "
                  "Time-critical: get to the nearest hospital with emergency capacity immediately.",
    },
    {
        "label": "Severe breathing difficulty",
        "symptoms": {"breathlessness", "fast_heart_rate"},
        "advice": "Breathlessness with a fast heart rate needs urgent medical evaluation. "
                  "Get to the nearest facility with emergency care now.",
    },
    {
        "label": "Severe dehydration",
        "symptoms": {"sunken_eyes", "dehydration", "vomiting"},
        "advice": "Signs of severe dehydration, especially in children or the elderly, need "
                  "urgent care. Get to the nearest facility promptly.",
    },
    {
        "label": "Possible severe allergic reaction",
        "symptoms": {"breathlessness", "skin_rash", "puffy_face_and_eyes"},
        "advice": "Breathing difficulty with a rash and a swollen face can indicate a severe allergic "
                  "reaction (anaphylaxis). Call emergency services immediately.",
    },
]

# Individual single symptoms that are urgent on their own, regardless of combination.
SINGLE_SYMPTOM_FLAGS = {
    "loss_of_balance": "Sudden loss of balance can be a stroke or neurological emergency sign — "
                        "seek urgent evaluation, especially if sudden in onset.",
}


def check_red_flags(symptoms: List[str]) -> List[Dict]:
    """
    Returns a list of triggered red-flag rules (empty list if none).
    Each item: {label, advice, matched_symptoms}
    """
    reported = set(s.strip() for s in symptoms)
    triggered = []

    for rule in RED_FLAG_RULES:
        if rule["symptoms"].issubset(reported):
            triggered.append({
                "label": rule["label"],
                "advice": rule["advice"],
                "matched_symptoms": sorted(rule["symptoms"]),
            })

    for symptom, advice in SINGLE_SYMPTOM_FLAGS.items():
        if symptom in reported:
            triggered.append({
                "label": "Urgent single symptom",
                "advice": advice,
                "matched_symptoms": [symptom],
            })

    return triggered
