# Toksisite değerlendirmesi

PR-AUC burada average precision olarak hesaplanır. Eksik etiketler maskelenir. ROC/PR-AUC tek sınıflı görevlerde tanımsız (NaN); bootstrap %95 percentile, tek sınıflı çekimler atlanır. Eşik 0.5; olasılık kalibrasyonu uygulanmadı. CI molekül bazlıdır, scaffold-kümeli CI değildir.

          ROC-AUC    PR-AUC        F1  Balanced Accuracy  evaluable_assays
model                                                                     
MLP      0.744105  0.346993  0.321205           0.683230                12
XGBoost  0.706299  0.301203  0.278990           0.639067                12

En düşük ve en yüksek test ROC-AUC:
  model     assay  ROC-AUC
XGBoost     NR-ER 0.603495
    MLP NR-AR-LBD 0.865465