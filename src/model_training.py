from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(n_estimators=500, num_class=5, objective='multi:softmax'),
        "CatBoost": CatBoostClassifier(iterations=800, learning_rate=0.03, depth=8, verbose=0)
    }

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)

    return models
