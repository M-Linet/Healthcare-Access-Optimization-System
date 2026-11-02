# Healthcare Access Optimization System

**Know where to go, before you need to.**

You tell it what you are feeling. It suggests how serious it might be, which level of care you need, and the nearest facilities in Kenya that offer that level. It works in English and Swahili.

## Why I built it

Kenya has about 9,700 health facilities in its public registers, but most of them are dispensaries and health centres. Only a few are referral hospitals, and those are mostly in a handful of towns. When someone gets sick, they often go to whatever is closest, even when that place cannot treat them. Then they get sent on somewhere else, which costs time and money.

I wanted to try a simple answer to a simple question: given these symptoms, what is the lowest level of care that is enough, and where is the nearest facility at that level?

## How it works

1. **You pick your symptoms.** There are 131 to choose from. You can add your age, sex, whether you are pregnant, and how bad each symptom is.
2. **A model reads them.** A Logistic Regression model suggests the most likely condition out of 41, and says how confident it is. If it is under 50% sure, the app says so and tells you to see a health worker.
3. **A care level is chosen.** Each condition is mapped to a KEPH level (the Kenya Essential Package for Health): 2 is a dispensary, 3 a health centre, 4 a sub-county hospital, 5 a county referral hospital, and 6 a national referral hospital. Your age, pregnancy and symptom severity can raise the level.
4. **Emergency rules run on their own.** Combinations like chest pain with breathlessness and sweating are checked by plain rules, not by the model. If one matches, the app shows 999 / 112 and only lists level 5 and 6 hospitals, whatever the model said.
5. **It finds the nearest facilities** at that level or higher, using road distance when the routing service is reachable and straight-line distance when it is not. If the nearest suitable facility is more than 50 km away, it tells you.

The About page of the app has five real examples you can click through, each starting from a different town.

## Run it on your computer

You need Python 3 and an internet connection (the map and road routing use online services).

```
git clone https://github.com/M-Linet/Healthcare-Access-Optimization-System.git
cd Healthcare-Access-Optimization-System
pip install -r requirements.txt
cd backend
python -m uvicorn main:app --port 8000
```

Then open http://127.0.0.1:8000 in your browser. The whole app is served from that one link.

The trained model and the facility database are already in the repo, so you do not need to train anything. If you want to rebuild them:

```
python ML/train_model.py        # trains the model that the app uses
python backend/init_db.py       # rebuilds the facility database from the CSV
python backend/apply_corrections.py   # re-applies my fixes to the facility data
```

If you get a scikit-learn version warning, running `python ML/train_model.py` fixes it. It trains the same model on your version.

## How accurate is it?

On cases the model had not seen, it picks the right condition about **92%** of the time:

| | Result |
|---|---|
| Held-out test accuracy | 92.67% |
| 5-fold cross-validation | 92.13% ± 0.89% |
| Logistic Regression vs Random Forest vs Gradient Boosting | 92.13% / 89.09% / 88.49% |

I would not take that 92% at face value, and the next section says why.

The mistakes are not random. They are mostly between conditions that really are similar, such as hepatitis B, C, D and E, and some of the asthma variants.

The model's confidence is also worth something. With only 2 or 3 symptoms entered, it was right about 72% of the time when it was 40 to 50% confident, about 87% at 50 to 60%, and about 95% at 50% or more. That is where the 50% cut-off for the low-confidence warning comes from (`ML/evaluation/confidence_calibration.py`).

## About the data (and why 92% is probably too high)

**Symptoms and conditions.** I used the public Kaggle dataset "Disease Symptom Prediction" (author: itachi9604). It has 4,920 rows, but only 304 of them are different. The same cases are copied again and again, so a normal train/test split puts copies of one case on both sides, and the model just remembers them. I checked this: with the copies removed, the model still scored 100%, and it stayed above 99% even when I randomly dropped and added symptoms. The reason is that the dataset gives each condition a few fixed symptom lists, so it behaves like a lookup table, not like real patients.

To get something closer to real life, I built a larger training set of 6,150 cases from the real symptom lists in that dataset (`data/generate_augmented_dataset.py`). It keeps the real condition-to-symptom links. It adds symptoms that are common to many illnesses, sometimes borrows a symptom from another condition, and sometimes leaves a symptom out. Nothing medical is invented, but it is still built from the same 304 cases. That is why I say the accuracy is probably optimistic, because near-copies of one case can still end up on both sides of a split. Training, evaluation and model comparison all use the same case-grouped split, and the grouped scores came out almost the same as the earlier random-split scores (92.44% and 92.37%).

**Facilities.** The facility list is the World Bank / energydata.info "Kenya - Healthcare Facilities" file (from the Kenya Open Data portal). It has 10,013 rows, and 9,736 are loaded into the app after dropping bad coordinates and things that are not places to send a patient (stand-alone labs, VCT centres, a funeral home). KEPH levels are not in that file, so I worked them out from the facility type. The result is 7,722 at level 2, 1,431 at level 3, 552 at level 4, 28 at level 5 and 3 at level 6.

**Data story.** The notebook `notebook/Data_Exploration_and_Analysis.ipynb` walks through all of this with charts, including what the symptom data really contains, how facilities are spread across counties, how far people are from referral hospitals, and where the model gets it wrong. The charts are also in the app's "Data story" tab and saved in `ML/evaluation/story/`.

## What is in the repo

```
backend/        FastAPI server and the logic behind the answers
  main.py         the API, and serves the web page
  red_flags.py    emergency symptom combinations
  patient_context.py   age, sex, pregnancy and severity adjustments
  geo.py          distance, ranking, and hiding eye/dental/blood-bank listings
  road_distance.py     road routing with a straight-line fallback
  prediction_log.py    usage log that stores no personal details
  init_db.py, apply_corrections.py   build and fix the facility database
  tests/          54 automated tests
frontend/       the whole web app in one page (English and Swahili)
ML/             training, evaluation scripts, saved model, charts and metrics
data/           facility list, symptom data, care-level mapping, my corrections
notebook/       the data story and the model development notebooks
docs/           the test plan
```

## Limitations

This is a prototype. It has not been tested with patients, and no clinician has reviewed it.

- **Accuracy is likely too high.** See above. No significance test was run, and only the regularisation strength was checked, not a full grid search.
- **The care levels are my own judgement.** The model predicts a condition. Which KEPH level each condition needs comes from a mapping I decided myself. It is not an official guideline.
- **The emergency rules are mine too.** I wrote them for this project and a clinician has not looked at them.
- **Facility levels are inferred.** They come from the facility type, not the official register, so some are wrong. I corrected three private hospitals that were tagged as national referral hospitals (`data/facility_corrections.csv`). Known remaining problems are listed by `ML/evaluation/data_quality_check.py`: two Turkana dispensaries with coordinates outside Kenya, 13 name and county pairs that appear twice, and one row with placeholder coordinates. I left them in so the facility counts stay the same across the report and the notebook.
- **Eye, dental, blood-bank, pharmacy and physiotherapy listings are hidden** from results, but other odd listings may remain.
- **Routing is approximate.** Drive times ignore traffic, and the app cannot know whether a facility is open or has beds.
- **Not done yet:** user testing with real people, and a review by a health professional.

## Tests

```
cd backend
python -m pytest tests -v
```

54 tests cover distance and ranking, symptom encoding, the red-flag rules, the care level rules, the specialist-clinic filter, the routing fallback, the far-facility warning, the confidence cut-off and the API.

## What I would do next

- Test it with real users and fix what they find.
- Ask a clinician to check the emergency rules and the condition-to-level mapping.
- Check the facility levels against the official KMHFL registry.
- Tag the augmented cases by their source case, so the train/test split can be properly clean.
- Add a significance test when comparing models.

## Credits and sources

- Facilities: World Bank / energydata.info, "Kenya - Healthcare Facilities" (Kenya Open Data).
- Symptoms and conditions: "Disease Symptom Prediction", Kaggle (author: itachi9604).
- Road routing: Project OSRM. Map tiles: OpenStreetMap.
- Built with FastAPI, scikit-learn, pandas, SQLite and Leaflet.
