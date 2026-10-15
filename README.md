# StockPredictor

Daily probabilistic equity forecasting with reproducible datasets, immutable forecasts, and Vercel deployment through GitHub Actions.

Status: the Python pipeline extracts all planned source families and prepares the daily research dataset: cleaning, 34 shared trailing features, five-session labels, purged train/validation/calibration/test blocks and separate cleaned evidence. Immutable private archives retain source snapshots, checksums and versioned manifests. Historical availability and security identity are reconstructed research assumptions. The four LightGBM models, Ridge/logistic and per-stock ARIMA comparisons are trained, calibrated and evaluated. The candidate fails the direction baseline gate and remains research-only. The macro/neural accuracy experiment is complete; its ensemble did not improve on the original logistic baseline, and 90% accuracy remains unestablished. The SEC/news feature comparison is also complete; its measured results are linked below. Application tables and Vercel deployment are subsequent stages.

- [Build and deployment plan](BUILD_PLAN.md)
- [Data extraction commands, checkpoints and GitHub Actions setup](DATA_EXTRACTION.md)
- [Measured collection results and source coverage gaps](DATA_COLLECTION_REPORT.md)
- [Dataset cleaning and feature specification](DATASET_FEATURE_PLAN.md)
- [Implemented preparation commands and policies](DATASET_PREPARATION.md)
- [Measured prepared-dataset results](DATASET_PREPARATION_REPORT.md)
- [Shared training commands and evaluation policy](MODEL_TRAINING.md)
- [Measured trained-model results](MODEL_TRAINING_REPORT.md)
- [Direction challenger comparison](DIRECTION_CHALLENGERS_REPORT.md)
- [Accuracy research, free macro features and neural ensemble](ACCURACY_RESEARCH.md)
- [Measured accuracy research results](ACCURACY_RESEARCH_REPORT.md)
- [SEC and historical-news experiment](FUNDAMENTAL_NEWS_EXPERIMENT.md)
- [Measured SEC/news results](FUNDAMENTAL_NEWS_REPORT.md)
- [FI-2010 DeepLOB/TransLOB research and Colab GPU training](LOB_BENCHMARK.md)
- [Colab notebook and complete dataset instructions](COLAB_TRAINING.md)
- [Environment variables and service ownership](ENVIRONMENT_VARIABLES.md)
- [Environment template with fixed defaults](.env.example)

Populate the private local `.env` manually. Configure the corresponding scoped GitHub Actions/Vercel settings before deployment. Never commit `.env` or credentials. The daily MVP uses free Alpaca historical data. Accuracy research also uses collected free ALFRED/FRED vintages; SEC and historical-news feature comparisons are implemented; live serving remains a later phase.

Target: GitHub Actions runs Python collection/training/batch inference; Vercel hosts the Next.js dashboard and FastAPI serving API; external PostgreSQL and private S3-compatible storage hold durable records/artifacts. The `stock-predictor` and `stock-predictor-api` Vercel projects have been created through CLI login in `codera647s-projects`. Application configuration and deployment await implementation. Project IDs are documented in the environment guide; GitHub Actions requires its own deployment token.
