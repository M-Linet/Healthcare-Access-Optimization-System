"""UT-08: care level adjustments."""
from patient_context import adjust_keph_level


def test_pregnancy_with_abdominal_pain_is_at_least_level_3():
    r = adjust_keph_level(2, ["abdominal_pain"], pregnant=True)
    assert r["adjusted_level"] >= 3 and r["reasons"]


def test_pregnancy_without_related_symptoms_is_unchanged():
    assert adjust_keph_level(2, ["skin_rash"], pregnant=True)["adjusted_level"] == 2


def test_infant_is_at_least_level_3():
    assert adjust_keph_level(2, ["cough"], age=0)["adjusted_level"] == 3


def test_elderly_goes_up_one_level_but_never_above_6():
    assert adjust_keph_level(3, ["cough"], age=70)["adjusted_level"] == 4
    assert adjust_keph_level(6, ["cough"], age=70)["adjusted_level"] == 6


def test_two_severe_symptoms_raise_the_level():
    r = adjust_keph_level(3, ["a", "b"], severity={"a": "severe", "b": "severe"})
    assert r["adjusted_level"] == 4


def test_level_never_goes_down():
    for base in range(2, 7):
        for age in (None, 0, 30, 80):
            for preg in (None, True):
                r = adjust_keph_level(base, ["abdominal_pain"], age=age, pregnant=preg)
                assert r["adjusted_level"] >= base
