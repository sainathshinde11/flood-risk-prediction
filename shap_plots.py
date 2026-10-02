# ============================================================
# FLOOD RISK PREDICTION - DAY 3: SHAP (YOUR DATASET)
# Run: python flood_prediction_day3_shap.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score,
                              confusion_matrix, roc_curve)
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("  DAY 3: SHAP ANALYSIS & PAPER-READY PLOTS")
print("="*60)

FEATURES = [
    'MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
    'Siltation', 'AgriculturalPractices', 'Encroachments',
    'IneffectiveDisasterPreparedness', 'DrainageSystems',
    'CoastalVulnerability', 'Landslides', 'Watersheds',
    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
    'InadequatePlanning', 'PoliticalFactors'
]

# ============================================================
# STEP 1: LOAD
# ============================================================
print("\n📦 Loading train.csv...")
df = pd.read_csv("train.csv", nrows=50000)
df.drop(columns=['id'], inplace=True, errors='ignore')
df['flood_label'] = (df['FloodProbability'] >= 0.5).astype(int)
print(f"✅ Loaded: {df.shape[0]} rows")

# ============================================================
# STEP 2: PREPROCESS
# ============================================================
print("✅ Sampling 50,000 rows for SHAP analysis...")
df = df.sample(n=50000, random_state=42).reset_index(drop=True)
X = df[FEATURES].fillna(df[FEATURES].median())
y = df['flood_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train_sc, y_train)
print("✅ Preprocessing done")

# ============================================================
# STEP 3: TRAIN ALL 4 MODELS
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
for name, mdl in models.items():
    mdl.fit(X_sm, y_sm)
    yp    = mdl.predict(X_test_sc)
    yprob = mdl.predict_proba(X_test_sc)[:,1]
    results[name] = {
        'Accuracy':  round(accuracy_score(y_test,yp)*100, 2),
        'Precision': round(precision_score(y_test,yp, zero_division=0), 4),
        'Recall':    round(recall_score(y_test,yp, zero_division=0), 4),
        'F1-Score':  round(f1_score(y_test,yp, zero_division=0), 4),
        'AUC-ROC':   round(roc_auc_score(y_test,yprob), 4),
    }
    trained[name] = (mdl, yp, yprob)
    print(f"   ✅ {name}: Acc={results[name]['Accuracy']}%  AUC={results[name]['AUC-ROC']}")

results_df = pd.DataFrame(results).T
best_name  = results_df['AUC-ROC'].idxmax()
best_model = trained[best_name][0]

# Force SHAP to use XGBoost (TreeExplainer doesn't support Logistic Regression)
shap_model_name = "XGBoost"
shap_model = trained[shap_model_name][0]

# ============================================================
# STEP 4: SHAP VALUES
# ============================================================
print("\n🔍 Computing SHAP values (may take ~30 sec)...")
explainer   = shap.TreeExplainer(shap_model)
print(f"   (Using {shap_model_name} for SHAP — TreeExplainer requires tree-based models)")
shap_values = explainer.shap_values(X_test_sc)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
print("✅ SHAP done")

# ── Plot 1: Beeswarm ─────────────────────────────────────────
print("\n📊 SHAP Beeswarm plot...")
plt.figure(figsize=(11,8))
shap.summary_plot(sv, X_test_sc, feature_names=FEATURES, show=False)
plt.title(f'SHAP Summary (Beeswarm) — {best_name}', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_beeswarm.png")

# ── Plot 2: Bar importance ───────────────────────────────────
print("📊 SHAP Bar importance...")
mean_abs = np.abs(sv).mean(axis=0)
shap_df  = pd.DataFrame({'Feature':FEATURES,'SHAP':mean_abs}).sort_values('SHAP', ascending=True)

colors = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(shap_df)))
fig, ax = plt.subplots(figsize=(10,7))
bars = ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars, shap_df['SHAP']):
    ax.text(val+0.0002, bar.get_y()+bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
ax.set_title(f'Mean |SHAP| Feature Importance — {best_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('shap_bar_importance.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_bar_importance.png")

# ── Plot 3: Dependence (top 2 features) ─────────────────────
print("📊 SHAP Dependence plots...")
top2 = shap_df.sort_values('SHAP', ascending=False)['Feature'].values[:2]
fig, axes = plt.subplots(1,2, figsize=(14,5))
fig.suptitle('SHAP Dependence Plots — Top 2 Features', fontsize=13, fontweight='bold')
for i, feat in enumerate(top2):
    fidx = FEATURES.index(feat)
    sc = axes[i].scatter(X_test_sc[:,fidx], sv[:,fidx],
                         c=sv[:,fidx], cmap='RdYlGn', alpha=0.5, s=15)
    axes[i].axhline(0, color='black', linewidth=0.8, linestyle='--')
    axes[i].set_xlabel(feat, fontsize=11)
    axes[i].set_ylabel('SHAP Value', fontsize=11)
    axes[i].set_title(f'Effect of {feat}', fontsize=11)
    plt.colorbar(sc, ax=axes[i], label='SHAP Value')
plt.tight_layout()
plt.savefig('shap_dependence.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_dependence.png")

# ── Plot 4: Waterfall (single prediction) ───────────────────
print("📊 SHAP Waterfall plot...")
high_risk_idx = np.where(y_test.values == 1)[0]
sample_idx    = high_risk_idx[0] if len(high_risk_idx) > 0 else 0
base_val = (explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value)
explanation = shap.Explanation(
    values      = sv[sample_idx],
    base_values = base_val,
    data        = X_test_sc[sample_idx],
    feature_names = FEATURES
)
plt.figure(figsize=(10,7))
shap.waterfall_plot(explanation, show=False)
plt.title(f'SHAP Waterfall — Single High-Risk Sample ({best_name})',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_waterfall.png")

# ============================================================
# STEP 5: ROC CURVES (all models)
# ============================================================
print("📊 ROC Curves...")
colors_roc = ['#3498db','#2ecc71','#e67e22','#e74c3c']
fig, ax = plt.subplots(figsize=(8,6))
for (name,(mdl,yp,yprob)), color in zip(trained.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, yprob)
    ax.plot(fpr, tpr, label=f'{name} (AUC={results[name]["AUC-ROC"]})',
            color=color, linewidth=2.2)
ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig('roc_all_models.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: roc_all_models.png")

# ============================================================
# STEP 6: ALL CONFUSION MATRICES
# ============================================================
print("📊 Confusion Matrices...")
fig, axes = plt.subplots(2,2, figsize=(12,10))
fig.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
axes = axes.flatten()
for i,(name,(mdl,yp,yprob)) in enumerate(trained.items()):
    cm = confusion_matrix(y_test, yp)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['No Flood','Flood'],
                yticklabels=['No Flood','Flood'],
                annot_kws={'size':14,'weight':'bold'})
    axes[i].set_title(f'{name}\nAcc={results[name]["Accuracy"]}%  F1={results[name]["F1-Score"]}',
                      fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Actual'); axes[i].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrices_all.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: confusion_matrices_all.png")

# ============================================================
# STEP 7: PAPER TABLE IMAGE
# ============================================================
print("📊 Paper Table 1...")
fig, ax = plt.subplots(figsize=(13,3))
ax.axis('off')
table_data = [[name]+list(v.values()) for name,v in results.items()]
col_labels = ['Model','Accuracy (%)','Precision','Recall','F1-Score','AUC-ROC']
table = ax.table(cellText=table_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.2)
for j in range(len(col_labels)):
    table[0,j].set_facecolor('#2c3e50')
    table[0,j].set_text_props(color='white', fontweight='bold')
best_row = list(results.keys()).index(best_name)+1
for j in range(len(col_labels)):
    table[best_row,j].set_facecolor('#d5f5e3')
    table[best_row,j].set_text_props(fontweight='bold')
for i in range(1, len(table_data)+1):
    if i != best_row:
        for j in range(len(col_labels)):
            table[i,j].set_facecolor('#f9f9f9' if i%2==0 else 'white')
ax.set_title('Table 1: Performance Comparison (Green = Best)',
             fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('paper_table1.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: paper_table1.png")

# ============================================================
# STEP 8: COMBINED FIGURE FOR PAPER
# ============================================================
print("📊 Combined paper figure...")
fig = plt.figure(figsize=(16,12))
gs  = gridspec.GridSpec(2,2, figure=fig, hspace=0.4, wspace=0.35)

# (a) SHAP bar
ax1 = fig.add_subplot(gs[0,0])
c   = plt.cm.RdYlGn(np.linspace(0.2,0.9,len(shap_df)))
ax1.barh(shap_df['Feature'], shap_df['SHAP'], color=c, edgecolor='black', linewidth=0.4)
ax1.set_title('(a) SHAP Feature Importance', fontweight='bold', fontsize=12)
ax1.set_xlabel('Mean |SHAP Value|')
ax1.tick_params(axis='y', labelsize=8)
ax1.spines[['top','right']].set_visible(False)

# (b) ROC
ax2 = fig.add_subplot(gs[0,1])
for (name,(mdl,yp,yprob)),color in zip(trained.items(), colors_roc):
    fpr,tpr,_ = roc_curve(y_test,yprob)
    ax2.plot(fpr,tpr,label=f'{name} ({results[name]["AUC-ROC"]})',linewidth=2)
ax2.plot([0,1],[0,1],'k--',linewidth=1)
ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
ax2.set_title('(b) ROC Curves', fontweight='bold', fontsize=12)
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.spines[['top','right']].set_visible(False)

# (c) Best confusion matrix
ax3 = fig.add_subplot(gs[1,0])
cm_best = confusion_matrix(y_test, trained[best_name][1])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['No Flood','Flood'],
            yticklabels=['No Flood','Flood'],
            annot_kws={'size':13,'weight':'bold'})
ax3.set_title(f'(c) Confusion Matrix — {best_name}', fontweight='bold', fontsize=12)
ax3.set_ylabel('Actual'); ax3.set_xlabel('Predicted')

# (d) Accuracy bar
ax4 = fig.add_subplot(gs[1,1])
mnames = list(results.keys())
accs   = [results[m]['Accuracy'] for m in mnames]
bcols  = ['#e74c3c' if m==best_name else '#3498db' for m in mnames]
bars   = ax4.bar(mnames, accs, color=bcols, edgecolor='black')
ax4.set_ylim(60,100)
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('(d) Accuracy Comparison', fontweight='bold', fontsize=12)
ax4.set_xticklabels(mnames, rotation=12, ha='right', fontsize=9)
for bar,val in zip(bars,accs):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f'{val}%', ha='center', fontsize=9, fontweight='bold')
ax4.spines[['top','right']].set_visible(False)

fig.suptitle('Flood Risk Prediction — Experimental Results',
             fontsize=15, fontweight='bold', y=1.01)
plt.savefig('paper_combined_figure.png', dpi=200, bbox_inches='tight')
plt.close()
print("✅ Saved: paper_combined_figure.png")

# ============================================================
# SAVE CSVs
# ============================================================
results_df.to_csv('all_model_results.csv')
shap_df.sort_values('SHAP',ascending=False).to_csv('shap_importance.csv',index=False)
print("✅ Saved: all_model_results.csv, shap_importance.csv")

print("\n"+"="*60)
print("🎉 DAY 3 COMPLETE!")
print("="*60)
print("\n📁 Files for your paper:")
print("  shap_beeswarm.png         → Figure 3")
print("  shap_bar_importance.png   → Figure 4")
print("  shap_dependence.png       → Figure 5")
print("  shap_waterfall.png        → Figure 6")
print("  roc_all_models.png        → Figure 1")
print("  confusion_matrices_all.png→ Figure 2")
print("  paper_combined_figure.png → ⭐ Single combined figure")
print("  paper_table1.png          → Table 1")
print(f"\n🏆 Best: {best_name} | Acc={results[best_name]['Accuracy']}% | AUC={results[best_name]['AUC-ROC']}")
print("="*60)