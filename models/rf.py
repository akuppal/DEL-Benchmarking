import joblib
from sklearn.ensemble import RandomForestClassifier

SEED = 42
N_ESTIMATORS = 1000
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1
MAX_DEPTH = 57
FIT_KWARGS = {}
THRESHOLD = 0.95  # predict_proba cutoff for calling a hit


def build_model():
    return RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                  min_samples_split=MIN_SAMPLES_SPLIT,
                                  min_samples_leaf=MIN_SAMPLES_LEAF,
                                  max_depth=MAX_DEPTH,
                                  n_jobs=-1, random_state=SEED)


if __name__ == "__main__":
    import h5py, numpy as np
    from sklearn.model_selection import train_test_split

    with h5py.File("train_feature.h5") as hf:
        X = hf["feature"][:]
    y = np.load("labels.npy")  # same order as the features

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = build_model()
    model.fit(Xtr, ytr)
    joblib.dump(model, "my_rf.joblib")
