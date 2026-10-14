# StockPredictor

Daily probabilistic equity forecasting with reproducible datasets, immutable forecasts, and Vercel deployment through GitHub Actions.

Status: planning and configuration prepared. Application code, market-data backfill, training and deployment have not started.

- [Build and deployment plan](BUILD_PLAN.md)
- [Dataset cleaning and feature specification](DATASET_FEATURE_PLAN.md)
- [Environment variables and service ownership](ENVIRONMENT_VARIABLES.md)
- [Environment template with fixed defaults](.env.example)

Populate the private local `.env` manually. Configure the corresponding scoped GitHub Actions/Vercel settings before deployment. Never commit `.env` or credentials. The daily MVP uses free Alpaca historical data; macro/news/live integrations are later phases.

Target: GitHub Actions runs Python collection/training/batch inference; Vercel hosts the Next.js dashboard and FastAPI serving API; external PostgreSQL and private S3-compatible storage hold durable records/artifacts. The `stock-predictor` and `stock-predictor-api` Vercel projects have been created through CLI login in `codera647s-projects`. Application configuration and deployment await implementation. Project IDs are documented in the environment guide; GitHub Actions requires its own deployment token.

