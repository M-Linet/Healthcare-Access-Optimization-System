import sys, warnings, json; warnings.filterwarnings("ignore")
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from train_model import load_raw_rows, build_symptom_vocabulary, vectorize, DATA_PATH, RANDOM_SEED, manual_train_test_split
_, rows = load_raw_rows(DATA_PATH); vocab = build_symptom_vocabulary(rows); X, y = vectorize(rows, vocab)
Xtr, Xte, ytr, yte = manual_train_test_split(X, y); le = LabelEncoder().fit(y); ytr_i = le.transform(ytr)
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
def mk(parts):
    d = {"lr": lambda: LogisticRegression(C=1, max_iter=2000, random_state=RANDOM_SEED),
         "nb": lambda: MultinomialNB(alpha=0.5),
         "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=400, early_stopping=True, random_state=RANDOM_SEED)}
    return lambda: VotingClassifier([(p, d[p]()) for p in parts], voting="soft")
for parts in (["lr","nb"], ["lr","mlp"], ["nb","mlp"], ["lr","nb","mlp"]):
    accs=[]
    for tr, va in skf.split(Xtr, ytr_i):
        m = mk(parts)(); m.fit(Xtr[tr], ytr_i[tr]); accs.append(accuracy_score(ytr_i[va], m.predict(Xtr[va])))
    print("+".join(parts), f"CV {np.mean(accs):.4f} +/- {np.std(accs):.4f}", flush=True)
