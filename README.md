# StockPredictor

Daily probabilistic equity forecasting with reproducible datasets, immutable forecasts, and Vercel deployment through GitHub Actions.

Status: the Python extraction pipeline covers Alpaca bars/actions, Nasdaq/SEC directories, SEC facts/filing metadata and selected originals, FRED/ALFRED vintages, recent GDELT DOC/GKG metadata and prospective company announcements. It has immutable private archives, resumable requests, a frozen 50-stock + SPY universe, Parquet staging and coverage reports. The daily historical backfill completed. Feature generation, application tables, training and Vercel deployment are subsequent stages.

- [Build and deployment plan](BUILD_PLAN.md)
- [Data extraction commands, checkpoints and GitHub Actions setup](DATA_EXTRACTION.md)
- [Measured collection results and source coverage gaps](DATA_COLLECTION_REPORT.md)
- [Dataset cleaning and feature specification](DATASET_FEATURE_PLAN.md)
- [Environment variables and service ownership](ENVIRONMENT_VARIABLES.md)
- [Environment template with fixed defaults](.env.example)

Populate the private local `.env` manually. Configure the corresponding scoped GitHub Actions/Vercel settings before deployment. Never commit `.env` or credentials. The daily MVP uses free Alpaca historical data; macro/news/live integrations are later phases.

Target: GitHub Actions runs Python collection/training/batch inference; Vercel hosts the Next.js dashboard and FastAPI serving API; external PostgreSQL and private S3-compatible storage hold durable records/artifacts. The `stock-predictor` and `stock-predictor-api` Vercel projects have been created through CLI login in `codera647s-projects`. Application configuration and deployment await implementation. Project IDs are documented in the environment guide; GitHub Actions requires its own deployment token.
