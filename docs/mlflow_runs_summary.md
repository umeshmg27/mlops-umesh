# MLflow Runs Summary

The repository includes the local MLflow file store under `mlruns/`. The latest training run was executed
after the dataset and pipeline were generated, and MLflow logged separate runs for Logistic Regression and
Random Forest.

## Latest Selected Model

- Selected model: `logistic_regression`
- Selection metric: `test_roc_auc`
- Trained at: `2026-05-10T15:42:55.516862+00:00`
- Model artifact: `models/model.joblib`
- Metadata artifact: `models/metadata.json`

## Latest Metrics

| Model | Accuracy | Precision | Recall | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.869 | 0.812 | 0.929 | 0.966 |
| Random Forest | 0.852 | 0.806 | 0.893 | 0.951 |

## MLflow Evidence

- Experiment store: `mlruns/827868829186113828/`
- Run artifacts include confusion matrices and ROC curves.
- Run parameters include `model_name`, `cv_folds`, and `test_size`.
- Run metrics include cross-validation accuracy, precision, recall, ROC-AUC, and test metrics.

Open locally after starting Compose:

```bash
docker compose up --build
```

Then visit:

```text
http://localhost:5001
```
