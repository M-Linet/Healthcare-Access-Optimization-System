"""
build_keph_mapping_real.py
-----------------------------
Builds data/disease_keph_mapping.csv for the 41 disease classes in the real
Kaggle "Disease Symptom Prediction" dataset (itachi9604). The source dataset
has no KEPH level column, so this mapping is a clinical judgment call based
on typical severity and care requirements for each condition -- documented
here explicitly so it's auditable, not a black box.

KEPH levels: 2=Dispensary, 3=Health Centre, 4=Primary/Sub-County Hospital,
             5=County Referral Hospital, 6=National Referral Hospital
"""

import csv

DISEASE_KEPH = {
    "(vertigo) Paroymsal  Positional Vertigo": 3,   # outpatient neuro workup
    "AIDS": 4,                                       # chronic ART management
    "Acne": 2,                                       # minor dermatological
    "Alcoholic hepatitis": 5,                        # serious liver inflammation
    "Allergy": 2,                                    # minor, outpatient
    "Arthritis": 3,                                  # chronic musculoskeletal mgmt
    "Bronchial Asthma": 3,                           # outpatient mgmt (escalates if severe)
    "Cervical spondylosis": 3,                       # chronic musculoskeletal
    "Chicken pox": 2,                                # self-limited infectious
    "Chronic cholestasis": 5,                        # liver specialist care
    "Common Cold": 2,                                # minor
    "Dengue": 4,                                     # can progress to severe/hemorrhagic
    "Diabetes": 4,                                   # chronic endocrine management
    "Dimorphic hemmorhoids(piles)": 3,               # minor procedural
    "Drug Reaction": 4,                              # can be severe/anaphylactic
    "Fungal infection": 2,                           # minor dermatological
    "GERD": 3,                                       # chronic outpatient
    "Gastroenteritis": 3,                            # usually self-limited, dehydration risk
    "Heart attack": 6,                               # emergency, national referral
    "Hepatitis B": 5,                                # chronic liver specialist
    "Hepatitis C": 5,                                # chronic liver specialist
    "Hepatitis D": 5,                                # chronic liver specialist
    "Hepatitis E": 4,                                # usually self-limited, risk in pregnancy
    "Hypertension": 3,                               # chronic outpatient management
    "Hyperthyroidism": 4,                            # endocrine specialist
    "Hypoglycemia": 4,                               # can be an acute emergency
    "Hypothyroidism": 3,                             # chronic outpatient management
    "Impetigo": 2,                                   # minor skin infection
    "Jaundice": 4,                                   # can indicate serious liver pathology
    "Malaria": 3,                                    # uncomplicated malaria, outpatient
    "Migraine": 3,                                   # outpatient neuro management
    "Osteoarthristis": 3,                            # chronic musculoskeletal
    "Paralysis (brain hemorrhage)": 6,               # stroke emergency
    "Peptic ulcer diseae": 4,                        # perforation/bleeding risk
    "Pneumonia": 4,                                  # can be severe, needs hospital care
    "Psoriasis": 3,                                  # chronic dermatological
    "Tuberculosis": 4,                               # specialized long-term treatment
    "Typhoid": 3,                                    # can be severe but often outpatient-managed
    "Urinary tract infection": 2,                    # minor, common outpatient
    "Varicose veins": 3,                             # elective/minor procedural
    "hepatitis A": 3,                                # usually self-limited
}


def write_mapping(path="disease_keph_mapping.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Disease", "Recommended_KEPH_Level"])
        for disease, level in DISEASE_KEPH.items():
            writer.writerow([disease, level])
    print(f"Wrote {len(DISEASE_KEPH)} disease -> KEPH mappings to {path}")


if __name__ == "__main__":
    write_mapping()
