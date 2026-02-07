import pandas as pd

def feature_importance(model, X):
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.coef_[0]
    }).sort_values(by="Importance", ascending=False)

    print(importance.head(10))
