from sklearn.linear_model import LogisticRegression

def build_model(params):
    return LogisticRegression(**params)
