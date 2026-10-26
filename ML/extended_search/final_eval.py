import sys, warnings, json; warnings.filterwarnings("ignore")
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from train_model import load_raw_rows, build_symptom_vocabulary, vectorize, DATA_PATH, RANDOM_SEED, manual_train_test_split
_, rows = load_raw_rows(DATA_PATH); vocab = build_symptom_vocabulary(rows); X, y = vectorize(rows, vocab)
Xtr, Xte, ytr, yte = manual_train_test_split(X, y)
lr = LogisticRegression(C=1, max_iter=2000, random_state=RANDOM_SEED)
nb = MultinomialNB(alpha=0.5)
ens = VotingClassifier([("lr", LogisticRegression(C=1, max_iter=2000, random_state=RANDOM_SEED)), ("nb", MultinomialNB(alpha=0.5))], voting="soft")
for n, m in (("LogReg C=1", lr), ("MultinomialNB a=0.5", nb), ("LogReg+NB soft vote", ens)):
    m.fit(Xtr, ytr); p = m.predict(Xte)
    print(f"{n:<22} test acc {accuracy_score(yte,p):.4f}  macro F1 {f1_score(yte,p,average='macro'):.4f}  train acc {accuracy_score(ytr,m.predict(Xtr)):.4f}")
