from xgboost import XGBClassifier

SEED = 42
MAX_DEPTH = 9
GAMMA = 5  # min loss reduction to split
COLSAMPLE_BYTREE = 1
ALPHA = 0.8  # L1 regularization term
FIT_KWARGS = {}


def build_model():
    return XGBClassifier(max_depth=MAX_DEPTH, gamma=GAMMA,
                         colsample_bytree=COLSAMPLE_BYTREE, reg_alpha=ALPHA,
                         random_state=SEED)


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
    model.save_model("my_xgb.json")
