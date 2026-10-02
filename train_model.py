# ============================================================
# Target: FloodProbability | Source: Kaggle train.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================
print("=" * 60)
print(" LOAD, EDA, TRAIN, EVALUATE")
print("=" * 60)

df = pd.read_csv("train.csv")
print(f"\n✅ Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")

df.drop(columns=['id'], inplace=True)

TARGET = 'FloodProbability'

FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

print(f"\n📋 Features : {len(FEATURES)}")
print(f"🎯 Target   : {TARGET}")
print(f"\n📊 Target stats:\n{df[TARGET].describe()}")

# ============================================================
# STEP 2: BINARIZE TARGET
# FloodProbability is 0.0–1.0 → convert to 0/1
# ============================================================
df['flood_label'] = (df[TARGET] >= 0.5).astype(int)
print(f"\n✅ Binarized at threshold 0.5")
print(f"   Flood (1)    : {df['flood_label'].sum()}")
print(f"   No Flood (0) : {(df['flood_label']==0).sum()}")
print(f"   Flood ratio  : {df['flood_label'].mean():.2%}")

# ============================================================
# STEP 3: EDA PLOTS
# ============================================================
print("\n📊 Generating EDA plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Flood Prediction Dataset - EDA Overview', fontsize=16, fontweight='bold')

# Class distribution
axes[0,0].bar(['No Flood','Flood'],
              df['flood_label'].value_counts().sort_index().values,
              color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[0,0].set_title('Class Distribution')
axes[0,0].set_ylabel('Count')
for i,v in enumerate(df['flood_label'].value_counts().sort_index().values):
    axes[0,0].text(i, v+50, str(v), ha='center', fontweight='bold')

# FloodProbability histogram
axes[0,1].hist(df[TARGET], bins=40, color='#3498db', edgecolor='black', alpha=0.8)
axes[0,1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold=0.5')
axes[0,1].set_title('FloodProbability Distribution')
axes[0,1].set_xlabel('Flood Probability')
axes[0,1].legend()

# Correlation heatmap
corr = df[FEATURES].corr()
sns.heatmap(corr, ax=axes[0,2], cmap='RdYlGn', center=0,
            annot=False, square=True, linewidths=0.3)
axes[0,2].set_title('Feature Correlation Heatmap')
axes[0,2].tick_params(axis='x', rotation=90, labelsize=7)
axes[0,2].tick_params(axis='y', rotation=0,  labelsize=7)

# Top correlated features
top_corr = (df[FEATURES+[TARGET]].corr()[TARGET]
            .drop(TARGET).abs()
            .sort_values(ascending=False).head(8))
axes[1,0].barh(top_corr.index, top_corr.values,
               color=plt.cm.RdYlGn(np.linspace(0.8,0.2,len(top_corr))),
               edgecolor='black')
axes[1,0].set_title('Top 8 Features vs FloodProbability')
axes[1,0].set_xlabel('Absolute Correlation')

# MonsoonIntensity scatter
sc = axes[1,1].scatter(df['MonsoonIntensity'], df[TARGET],
                       c=df['flood_label'], cmap='RdYlGn_r', alpha=0.3, s=5)
axes[1,1].set_xlabel('Monsoon Intensity')
axes[1,1].set_ylabel('Flood Probability')
axes[1,1].set_title('Monsoon Intensity vs Flood Probability')

# Feature means by class
flood_means = df.groupby('flood_label')[FEATURES[:6]].mean().T
flood_means.plot(kind='bar', ax=axes[1,2],
                 color=['#2ecc71','#e74c3c'], edgecolor='black')
axes[1,2].set_title('Feature Means: Flood vs No Flood')
axes[1,2].set_xticklabels(FEATURES[:6], rotation=20, ha='right', fontsize=8)
axes[1,2].legend(['No Flood','Flood'])

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: eda_overview.png")

# ============================================================
# STEP 4: PREPROCESSING
# ============================================================
print("\n⚙️  Preprocessing...")
X = df[FEATURES].copy()
y = df['flood_label'].copy()
X.fillna(X.median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"⚖️  Before SMOTE: {dict(y_train.value_counts())}")
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train_sc, y_train)
print(f"✅ After  SMOTE: {dict(pd.Series(y_sm).value_counts())}")

# ============================================================
# STEP 5: TRAIN ALL 4 MODELS
# ============================================================
print("\n🤖 Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost":             XGBClassifier(n_estimators=100, random_state=42,
                                         eval_metric='logloss', verbosity=0),
}

results = {}
trained = {}

for name, model in models.items():
    model.fit(X_sm, y_sm)
    yp    = model.predict(X_test_sc)
    yprob = model.predict_proba(X_test_sc)[:,1]
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test, yp)*100, 2),
        'Precision': round(precision_score(y_test, yp, zero_division=0), 4),
        'Recall':    round(recall_score(y_test, yp, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test, yp, zero_division=0), 4),
        'AUC-ROC':   round(roc_auc_score(y_test, yprob), 4),
    }
    trained[name] = (model, yp, yprob)
    print(f"   ✅ {name}: Acc={results[name]['Accuracy']}%  AUC={results[name]['AUC-ROC']}")

results_df = pd.DataFrame(results).T
best_name  = results_df['AUC-ROC'].idxmax()
print(f"\n🏆 Best Model: {best_name}")
print(results_df.to_string())

# ============================================================
# STEP 6: RESULT PLOTS
# ============================================================
colors = ['#3498db','#2ecc71','#e67e22','#e74c3c']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Model Comparison', fontsize=14, fontweight='bold')
for i, metric in enumerate(['Accuracy','AUC-ROC','F1-Score']):
    vals = results_df[metric].values
    if metric == 'Accuracy': vals = vals / 100
    bars = axes[i].bar(results_df.index, vals, color=colors, edgecolor='black')
    axes[i].set_title(metric)
    axes[i].set_ylim(0, 1.15)
    axes[i].set_xticklabels(results_df.index, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01,
                     f'{val:.3f}', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: model_comparison.png")

plt.figure(figsize=(8,6))
for (name,(mdl,yp,yprob)), color in zip(trained.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, yprob)
    plt.plot(fpr, tpr, label=f'{name} (AUC={results[name]["AUC-ROC"]})',
             color=color, linewidth=2)
plt.plot([0,1],[0,1],'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: roc_curves.png")

best_model, best_pred, _ = trained[best_name]
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Flood','Flood'],
            yticklabels=['No Flood','Flood'],
            annot_kws={'size':14,'weight':'bold'})
plt.title(f'Confusion Matrix - {best_name}')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: confusion_matrix.png")

# ============================================================
# STEP 7: SHAP
# ============================================================
print("\n🔍 Computing SHAP values...")
try:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_sc)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    plt.figure(figsize=(10,7))
    shap.summary_plot(sv, X_test_sc, feature_names=FEATURES, show=False)
    plt.title(f'SHAP Summary - {best_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('shap_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Saved: shap_importance.png")
except Exception as e:
    print(f"⚠️  SHAP: {e}")

results_df.to_csv('model_results.csv')
print("✅ Saved: model_results.csv")

print("\n" + "="*60)
print("="*60)
print(f"\n🏆 Best Model : {best_name}")
print(f"   Accuracy   : {results[best_name]['Accuracy']}%")
print(f"   AUC-ROC    : {results[best_name]['AUC-ROC']}")
print(f"   F1-Score   : {results[best_name]['F1-Score']}")