
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

#CUDA boosted XGBoost Classifier with 5-Fold CV, change the parameter distribution to a wider range if the hardware allows
def train_XGBoost(embeds, y):
    print("XGBoost training start")
    X_train, X_test, y_train, y_test = train_test_split(
        embeds, y,
        test_size=0.1,
        stratify=y,
        random_state=42
    )

    xgb = XGBClassifier(
        tree_method='hist',
        device='cuda',
        verbosity=1,
        objective='multi:softmax' if len(set(y)) > 2 else 'binary:logistic',
        eval_metric='mlogloss' if len(set(y)) > 2 else 'logloss',
        random_state=42
    )

    pipeline = Pipeline([
        ('xgb', xgb)
    ])

    param_dist = {
        'xgb__n_estimators': stats.randint(50, 150),          
        'xgb__max_depth': stats.randint(3, 7),                 
        'xgb__learning_rate': stats.uniform(0.05, 0.2),        
        'xgb__subsample': stats.uniform(0.7, 0.2),
        'xgb__colsample_bytree': stats.uniform(0.7, 0.2),
        'xgb__gamma': stats.uniform(0, 2),
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("In Randomized Search")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=inner_cv,
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train,
                      y_train,
                      xgb__verbose=False)
    best_clf = random_search.best_estimator_

    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_accuracy = accuracy_score(y_test, y_pred)
    print("In Cross Val")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_scores = cross_val_score(
        random_search,
        embeds,
        y,
        cv=outer_cv,
        scoring='f1_weighted'
    )

    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Test F1: {test_f1:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Nested CV F1: {nested_scores.mean():.2f} (±{nested_scores.std():.2f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Alzheimer'], zero_division=0))
    return best_clf,y_test, y_pred, y_prob

#Support Vector Machine of scikit-learn, no gpu boost
def train_SVM(imgs, y):

    print("SVM training start\nsplitting")
    X_train, X_test, y_train, y_test = train_test_split(
        imgs, y,
        test_size=0.1,
        stratify=y,
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(class_weight='balanced', kernel='rbf', probability=True))
    ])

    param_dist = {
        'svm__C': stats.loguniform(1e-3, 1e3),  
        'svm__gamma': ['scale', 'auto'] + list(np.logspace(-3, 1, 5))  
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("hypparam tuning")
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,  
        cv=inner_cv,
        scoring='f1_weighted',
        n_jobs=4,
        random_state=42
    )
    print("fitting on random search")
    random_search.fit(X_train, y_train)
    print("getting the best model")
    best_clf = random_search.best_estimator_
    print("eval the best model")
    y_pred = best_clf.predict(X_test)
    y_prob = best_clf.predict_proba(X_test)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_accuracy = accuracy_score(y_test, y_pred)
    print("cross validation")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    nested_scores = cross_val_score(
        random_search,
        imgs,
        y,
        cv=outer_cv,
        scoring='f1_weighted'
    )
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Test F1: {test_f1:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Nested CV F1: {nested_scores.mean():.2f} (±{nested_scores.std():.2f})")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Alzheimer'], zero_division=0))
    return best_clf,y_test, y_pred, y_prob
