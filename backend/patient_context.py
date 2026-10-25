"""
patient_context.py
--------------------
Adjusts the raw ML recommendation using patient context the classifier
itself doesn't see: age, sex, pregnancy status, and per-symptom severity.

The disease classifier only knows symptom presence/absence. It has no idea
if the patient is a pregnant woman, an infant, or a report of "mild" vs
"severe" pain. This module applies simple, explainable adjustment rules on
top of the model's output rather than trying to bake all of this into the
ML model itself (which would need a much larger, richer training set than
this project currently has).
"""

from typing import Optional, Dict, List

PREGNANCY_RELATED_SYMPTOMS = {
    "abdominal_pain", "vomiting", "puffy_face_and_eyes", "headache", "high_fever", "back_pain",
}

INFANT_MAX_AGE = 1
CHILD_MAX_AGE = 12
ELDERLY_MIN_AGE = 65


def adjust_keph_level(
    base_level: int,
    symptoms: List[str],
    age: Optional[int] = None,
    sex: Optional[str] = None,
    pregnant: Optional[bool] = None,
    severity: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Returns {adjusted_level, reasons: [...]} — base_level bumped upward
    (never downward — context should never make a case seem LESS urgent)
    based on patient context.
    """
    level = base_level
    reasons = []

    if pregnant:
        overlap = PREGNANCY_RELATED_SYMPTOMS.intersection(symptoms)
        if overlap:
            level = max(level, 3)  # at minimum a facility with maternity capability
            reasons.append(
                f"Pregnancy with {', '.join(sorted(overlap))} — routed to at least a "
                "Health Centre (KEPH L3) or above for maternity-capable care."
            )

    if age is not None:
        if age <= INFANT_MAX_AGE:
            level = max(level, 3)
            reasons.append("Infant (under 1 year) — routed to at least a Health Centre or above.")
        elif age >= ELDERLY_MIN_AGE:
            level = min(level + 1, 6)
            reasons.append("Elderly patient (65+) — recommendation raised one KEPH level as a precaution.")

    if severity:
        severe_count = sum(1 for s, sev in severity.items() if sev == "severe")
        if severe_count >= 2:
            level = min(level + 1, 6)
            reasons.append(
                f"{severe_count} symptoms reported as severe — recommendation raised one KEPH level."
            )

    return {"adjusted_level": level, "reasons": reasons}
