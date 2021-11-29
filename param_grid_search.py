import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def grid_matrix():

    param_grid_reg = {
                'C': [0.01,0.1,1,10],
    }
    model_reg = LogisticRegression(solver='liblinear', penalty="l2")

    param_grid_tree = {
                'max_depth':range(10, 30),
    }
    model_tree = DecisionTreeClassifier(criterion="gini", random_state=42)

    param_grid_forest = {
                'n_estimators': [50, 100, 150],
                'max_depth':range(3, 8),
                'max_features': ['auto', 'sqrt'],
    }
    model_forest = RandomForestClassifier(oob_score=True, bootstrap=True)

    param_grid_xgb = {
                "n_estimators": [150, 200, 300],
                "max_depth": [5, 10, 25],
                "learning_rate": [0.1, 0.05],
                "subsample": [0.5, 0.8],
                "colsample_bytree": [0.5, 0.8],
    }
    model_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    param_grid = [param_grid_reg, param_grid_tree, param_grid_forest, param_grid_xgb]
    model = [model_reg, model_tree, model_forest, model_xgb]
    names = ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"]

    matrix = list(zip(param_grid,model, names))

    return matrix


