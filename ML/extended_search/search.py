import sys, time, json, warnings
warnings.filterwarnings("ignore")
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from train_model import load_raw_rows, build_symptom_vocabulary, vectorize, DATA_PATH, RANDOM_SEED, manual_train_test_split

_, rows = load_raw_rows(DATA_PATH)
vocab = build_symptom_vocabulary(rows)
X, y = vectorize(rows, vocab)
Xtr, Xte, ytr, yte = manual_train_test_split(X, y)
le = LabelEncoder().fit(y)
ytr_i, yte_i = le.transform(ytr), le.transform(yte)
print("train", Xtr.shape, "test", Xte.shape, flush=True)
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)

C = {}
for c in [0.1, 0.3, 1, 3, 10, 30, 100]:
    C[f"LogReg C={c}"] = lambda c=c: LogisticRegression(C=c, max_iter=2000, random_state=RANDOM_SEED)
for a in [0.1, 0.5, 1.0]:
    C[f"BernoulliNB a={a}"] = lambda a=a: BernoulliNB(alpha=a)
    C[f"MultinomialNB a={a}"] = lambda a=a: MultinomialNB(alpha=a)
for c in [1, 10]:
    C[f"SVC linear C={c}"] = lambda c=c: SVC(kernel="linear", C=c, probability=True, random_state=RANDOM_SEED)
    C[f"SVC rbf C={c}"] = lambda c=c: SVC(kernel="rbf", C=c, probability=True, random_state=RANDOM_SEED)
for k in [5, 15]:
    C[f"kNN k={k}"] = lambda k=k: KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
for a in [1e-3, 1e-2, 1e-1]:
    C[f"MLP a={a}"] = lambda a=a: MLPClassifier(hidden_layer_sizes=(128,), alpha=a, max_iter=400, early_stopping=True, random_state=RANDOM_SEED)
C["RandomForest 300"] = lambda: RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=RANDOM_SEED)
C["ExtraTrees 300"] = lambda: ExtraTreesClassifier(n_estimators=300, n_jobs=2, random_state=RANDOM_SEED)
C["HistGB lr0.1"] = lambda: HistGradientBoostingClassifier(max_iter=150, random_state=RANDOM_SEED)
C["XGBoost d4 lr0.1 n150"] = lambda: XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, tree_method="hist", n_jobs=2, random_state=RANDOM_SEED, verbosity=0)

results = []
for name, f in C.items():
    t0 = time.time(); accs = []
    for tr, va in skf.split(Xtr, ytr_i):
        m = f(); m.fit(Xtr[tr], ytr_i[tr]); accs.append(accuracy_score(ytr_i[va], m.predict(Xtr[va])))
    r = dict(model=name, cv_mean=float(np.mean(accs)), cv_std=float(np.std(accs)), secs=round(time.time()-t0,1))
    results.append(r); print(f"{name:<28} CV {r['cv_mean']:.4f} +/- {r['cv_std']:.4f}  ({r['secs']}s)", flush=True)
    json.dump(results, open(os.path.join(HERE, "search_results.json"), "w"), indent=1)
print("DONE_SEARCH", flush=True)
