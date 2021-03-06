#ref: https://github.com/RRdmlearning/Machine-Learning-From-Scratch/tree/master/random_forest

from sklearn import datasets
from utils import train_test_split, accuracy_score, Plot
from RandomForest.random_forest_model import RandomForest

def main():
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, testdata_ratio=0.4, seed=2)
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", y_train.shape)

    clf = RandomForest(n_estimators=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    Plot().plot_in_2d(X_test, y_pred, title="Random Forest", accuracy=accuracy, legend_labels=data.target_names)


if __name__ == "__main__":
    main()