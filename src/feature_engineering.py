from sklearn.preprocessing import LabelEncoder

def encode_features(df):
    cat_cols = df.select_dtypes(include="object").columns
    encoder = LabelEncoder()

    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    return df
