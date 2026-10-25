"""UT-06: red-flag rules."""
from red_flags import check_red_flags


def test_heart_attack_rule_fires():
    result = check_red_flags(["chest_pain", "breathlessness", "sweating"])
    assert any(r["label"] == "Possible heart attack" for r in result)


def test_order_and_spaces_do_not_matter():
    result = check_red_flags([" sweating", "chest_pain ", "breathlessness"])
    assert result


def test_no_flag_for_common_symptoms():
    assert check_red_flags(["headache", "mild_fever"]) == []


def test_one_symptom_of_a_rule_is_not_enough():
    assert check_red_flags(["chest_pain"]) == []


def test_stroke_rule_with_one_sided_weakness():
    result = check_red_flags(["slurred_speech", "weakness_of_one_body_side"])
    assert any(r["label"] == "Possible stroke" for r in result)


def test_allergic_reaction_rule_uses_symptoms_a_user_can_pick():
    result = check_red_flags(["breathlessness", "skin_rash", "puffy_face_and_eyes"])
    assert any("allergic" in r["label"].lower() for r in result)


def test_single_symptom_flag():
    result = check_red_flags(["loss_of_balance"])
    assert result and result[0]["label"] == "Urgent single symptom"
