import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

# Load 50000 rows
df = pd.read_csv("train.csv", nrows=50000)
df.drop(columns=['id'], inplace=True, errors='ignore')
df['flood_label'] = (df['FloodProbability'] >= 0.5).astype(int)

X = df[FEATURES].fillna(df[FEATURES].median())
y = df['flood_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_tr = scaler.fit_transform(X_train)
X_te = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_tr, y_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=50, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                          eval_metric='logloss', verbosity=0),
}

print("\n📊 Results on 50,000 rows:")
print(f"{'Model':<25} {'Accuracy':>10} {'AUC-ROC':>10} {'F1-Score':>10}")
print("-" * 60)
for name, model in models.items():
    model.fit(X_sm, y_sm)
    yp    = model.predict(X_te)
    yprob = model.predict_proba(X_te)[:,1]
    acc   = accuracy_score(y_test, yp) * 100
    auc   = roc_auc_score(y_test, yprob)
    f1    = f1_score(y_test, yp)
    print(f"{name:<25} {acc:>9.2f}% {auc:>10.4f} {f1:>10.4f}")