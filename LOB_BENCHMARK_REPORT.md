# FI-2010 benchmark status

The DeepLOB/TransLOB implementation, full Colab GPU notebook, complete dataset ZIP
and handoff package have been prepared. **Full benchmark training and final test
evaluation have not run.** No reproduced accuracy or claimed 80–90% result exists yet.

The dataset checksum and dimensions are verified: 394,337 observations across the
official first-seven-day training file and three test-day files. The selected GPU
protocol yields 203,601 training, 50,751 validation and 138,990 test windows per horizon.
The package contains no `.env` or provider credentials.

The notebook will produce actual training histories, selected checkpoint hashes,
test predictions, per-file and pooled metrics, and recomputation checks when run
on Colab. Follow [COLAB_TRAINING.md](COLAB_TRAINING.md). The manual GitHub Actions
workflow is provided but has not been dispatched. Colab results remain in your
Drive until you download or transfer them.

The daily forecasting bundle, daily holdout and Vercel deployment are unchanged.
