import xgboost as xgb


def build_xgb():
    return xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,             # Back to original depth
        learning_rate=0.05,      # Back to original LR
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,      # Less aggressive
        reg_alpha=0.05,          # Light regularization
        reg_lambda=0.5,          # Light regularization
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        n_jobs=-1,
        random_state=42,
    )