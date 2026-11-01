import joblib
from sklearn.svm import SVC

SEED = 42
KERNEL = "rbf"
GAMMA = 0.001
C = 1
FIT_KWARGS = {}


def build_model():
    return SVC(kernel=KERNEL, gamma=GAMMA, C=C, random_state=SEED)


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
    joblib.dump(model, "my_svm.joblib")
