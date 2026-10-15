# Colab GPU training handoff

Open [StockPredictor_FI2010_Colab.ipynb](notebooks/StockPredictor_FI2010_Colab.ipynb)
in Google Colab using **File → Upload notebook**. Select **Runtime → Change runtime
type → GPU** and run the cells in order. The notebook contains its training code,
so it needs no repository login, `.env`, provider credentials or paid API.

The notebook trains DeepLOB and TransLOB at horizons 10 and 100 on the FI-2010
order-book benchmark. This is the separate intraday experiment discussed in
[LOB_BENCHMARK.md](LOB_BENCHMARK.md); it does not train the project's daily models.
Published headline accuracies are not guaranteed for this implementation.

## Files already created on your computer

| File under `F:\StockPridictor` | Purpose |
| --- | --- |
| `notebooks/StockPredictor_FI2010_Colab.ipynb` | Self-contained Colab notebook |
| `artifacts/colab/fi2010_complete.zip` | Complete four-file dataset for the supplied decimal-normalized seven-day/three-day setup; 56,278,154 bytes |
| `artifacts/colab/StockPredictor_Colab_Complete.zip` | Notebook, complete dataset, readable source, configs, tests and manifest in one package |
| `artifacts/colab/dataset_manifest.json` | Source checksum, member names, schema and observation counts |

The simplest route is to upload only the notebook. It automatically downloads the
same public dataset and verifies its checksum. To use the supplied dataset file,
upload `fi2010_complete.zip` to your Google Drive, then set `DATASET_ZIP` in the
notebook to its full Drive path. Do not put the outer `StockPredictor_Colab_Complete.zip`
in that variable; it is the handoff package, not the original dataset ZIP.

Google's [Colab FAQ](https://research.google.com/colaboratory/faq.html) explains
that free GPU availability, usage limits and runtime duration vary. A GPU session
is needed; the notebook stops with instructions if Colab provides no CUDA GPU.
The dependency cell preserves Colab's installed CUDA PyTorch, rather than installing
the project's optional CPU Torch wheel. No Colab subscription is required by this
notebook, but access to a free GPU is not guaranteed.

## Dataset and split

The complete supplied setup contains **394,337 observations**:

| Original file | Role | Observations |
| --- | --- | ---: |
| `Train_Dst_NoAuction_DecPre_CF_7.txt` | Official first seven training days | 254,750 |
| `Test_Dst_NoAuction_DecPre_CF_7.txt` | First test day | 55,478 |
| `Test_Dst_NoAuction_DecPre_CF_8.txt` | Second test day | 52,172 |
| `Test_Dst_NoAuction_DecPre_CF_9.txt` | Third test day | 31,937 |

Each text file has 149 rows, with observations across columns: the first 40 rows
are raw order-book features, 104 engineered rows are unused, and the final five
rows are native UP/STATIONARY/DOWN labels for horizons 10/20/30/50/100. The loader
maps native labels 1/2/3 to internal UP=2, FLAT=1, DOWN=0. It checks finite features,
schema, label values and alignment and rejects corrupt input.

The GPU recipe uses every eligible 100-observation input window. After purging
the training/validation boundary and dropping terminal-horizon endpoints:

| Split | Windows per horizon |
| --- | ---: |
| Training | 203,601 |
| Validation | 50,751 |
| Test, pooled across the three separate files | 138,990 |

Windows never cross file or train/validation boundaries. Internal stock/session
boundaries are absent from the supplied files and cannot be verified. Source ZIP
SHA-256: `7a5564290a504bdc104dccaf44aa6c0b3857bf1db6b0517b9b88577c2def3e5f`.
The source commit and limitations are documented in the benchmark specification.
This package contains the complete supplied decimal-normalized setup, not every
normalization/fold variant of the original FI-2010 release.

## Training, resuming and evaluation

The GPU config is [lob.fi2010.colab.json](configs/lob.fi2010.colab.json). Each of the
four models has a maximum of 150 epochs, patience 15 and a 90-minute budget checked
between epochs. Batch size is 128; model selection uses validation log loss.
Expect up to six hours of training plus parsing and evaluation if all four exhaust
their budgets. Actual time depends on Colab's GPU and CPU allocation.

The default persistent folder is `MyDrive/StockPredictor_FI2010`. The notebook
saves model weights, optimizer state and random-generator state after each complete
epoch. If disconnected, reconnect a GPU, reopen the same notebook and run it with
the same folder, bundled source, versions and configuration. It resumes the last
committed epoch; an incomplete epoch repeats. Version/config changes make a new
recipe and can prevent continuation. Keep checkpoints in Drive instead of `/content`.

The training cell uses `--train-only`: it selects all four models without opening
the test files. The next table shows validation results. The final evaluation cell
freezes all four selected checkpoint hashes and then evaluates every eligible test
window. Set `RUN_FINAL_TEST = False` before that cell if you want to stop at training.
After test exposure, the study's archive rejects changed recipes/model sets.

Results include accuracy, balanced accuracy, macro F1, per-class confusion matrices,
per-file metrics and a training-class-prior baseline. Saved prediction files permit
independent metric recomputation. The verification cell checks archived hashes and
recomputes metrics without retraining. No target score is substituted for a measured
score, and there is no automatic daily-model promotion.

The final cell saves `StockPredictor_FI2010_results.zip` in your Drive. It contains
model weights, predictions, histories, reports, config and hash records. Large source
data and optimizer-resume checkpoints stay in Drive. Download the results ZIP when
training finishes so it can be reviewed and integrated with this project.

## Rebuild the handoff

```powershell
.venv\Scripts\python.exe scripts/build_colab_bundle.py
```

The builder uses an explicit file whitelist, embeds a checksummed source bundle
in the notebook, compiles every code cell, checks the full dataset's SHA-256 and
records its file dimensions. `.env`, private data from the daily project and
credentials are excluded. Generated ZIPs stay ignored under `artifacts/`; only
the notebook, source, configs and documentation are intended for GitHub.

The local machine has no CUDA GPU. Full Colab GPU training has not been executed
here; checks cover the source, CPU forward/backward/reload behavior, split policy,
training-only holdout exclusion, epoch recovery, and the notebook/package structure.
