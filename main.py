from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import encode_features
from src.train_model import train_model
from src.evaluate_model import evaluate
from src.explain_model import feature_importance

df = load_and_clean_data("data/raw/telco_churn.csv")
df = encode_features(df)

model, X_test, y_test = train_model(df)
evaluate(model, X_test, y_test)
feature_importance(model, X_test)
