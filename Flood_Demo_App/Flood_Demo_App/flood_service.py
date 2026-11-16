"""Prediction helpers for the demo app (no Streamlit code here, so it can be tested alone)."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import urban_flood as uf


class FloodSystem:
    def __init__(self, seed=42):
        self.city = uf.make_city(seed)
        self.sets = uf.make_samples(self.city)
        tr, va = self.sets["train"], self.sets["val"]
        Xtr, Xva = (uf.tabular(s, self.city["static"]) for s in (tr, va))
        self.clf = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=3,
                                          n_jobs=-1, random_state=seed).fit(Xtr, tr["flood"].astype(int))
        self.reg = RandomForestRegressor(n_estimators=120, max_depth=14, min_samples_leaf=3,
                                         max_features=0.5, n_jobs=-1, random_state=seed).fit(Xtr, tr["depth"])
        p_va = self._prob(Xva)
        self.thresholds = [uf.best_threshold(va["flood"][:, i], p_va[:, i]) for i in range(len(uf.HORIZONS))]
        te = self.sets["test"]
        self.n_test_hours = len(te["cell"]) // uf.N_CELLS
        self.test_seq = te["seq"].reshape(self.n_test_hours, uf.N_CELLS, uf.WINDOW, 4)
        self.test_flood = te["flood"].reshape(self.n_test_hours, uf.N_CELLS, len(uf.HORIZONS))
        self.test_depth = te["depth"].reshape(self.n_test_hours, uf.N_CELLS, len(uf.HORIZONS))
        self.test_now = (te["now"].reshape(self.n_test_hours, uf.N_CELLS) > uf.FLOOD_DEPTH)

    def _prob(self, X):
        return np.stack([c[:, 1] for c in self.clf.predict_proba(X)], axis=1)

    def warning_hour(self, h_idx=1):
        """Best early-warning demo hour: many cells correctly flagged that are NOT flooded yet,
        with few misses and false alarms (picked on the test period, for demonstration only)."""
        te = self.sets["test"]
        X = uf.tabular(te, self.city["static"])
        flag = (self._prob(X)[:, h_idx] >= self.thresholds[h_idx]).reshape(self.n_test_hours, uf.N_CELLS)
        truth = self.test_flood[:, :, h_idx] > 0
        new_hits = (flag & truth & ~self.test_now).sum(1)
        wrong = (~flag & truth).sum(1) + (flag & ~truth).sum(1)
        return int(np.argmax(new_hits - wrong))

    def window(self, hour):
        return self.test_seq[hour].copy()                       # (cells, WINDOW, 4)

    @staticmethod
    def scenario(seq, rain_mult=1.0, river_add=0.0):
        s = seq.copy()
        s[:, :, 0] *= rain_mult                                  # local rainfall
        s[:, :, 1] *= rain_mult                                  # mean rainfall
        s[:, :, 2] += river_add                                  # river level
        return s

    def predict(self, seq, h_idx):
        split = {"seq": seq, "cell": np.arange(uf.N_CELLS)}
        X = uf.tabular(split, self.city["static"])
        prob = self._prob(X)[:, h_idx]
        depth = self.reg.predict(X)[:, h_idx]
        thr = self.thresholds[h_idx]
        return prob, depth, prob >= thr, thr

    def actual(self, hour, h_idx):
        return self.test_flood[hour, :, h_idx], self.test_depth[hour, :, h_idx]
