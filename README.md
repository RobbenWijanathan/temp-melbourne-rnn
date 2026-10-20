# temp-melbourne-rnn

Exploring daily minimum temperatures in Melbourne before building an RNN forecast.

Open `eda.ipynb` from this folder, select a Python kernel, and run all cells in order.
The notebook covers data quality, missing dates, temperature distributions,
seasonality, yearly changes, and relationships with previous days.

Install the notebook dependencies in your Python environment if needed:

```sh
python -m pip install -r requirements.txt
```

The original dataset is included in `data/daily-min-temperatures.csv` so the notebook
runs without downloading data. Source:
[Jason Brownlee's Datasets repository](https://github.com/jbrownlee/Datasets/blob/master/daily-min-temperatures.csv).

## Vanilla RNN notebook

Upload `vanilla_rnn.ipynb` to Google Colab and select **Runtime → Run all**.
Authorize Google Drive when prompted; a CPU runtime is sufficient.
The notebook downloads the dataset directly from GitHub.

Uses TensorFlow/Keras `SimpleRNN`, following the
[GeeksforGeeks tutorial](https://www.geeksforgeeks.org/machine-learning/time-series-forecasting-using-recurrent-neural-networks-rnn-in-tensorflow/).
It compares 16 window/feature combinations and six additional model settings.
Selection uses 1987–1988 validation RMSE; the final model is refitted on the
first 80% and evaluated once on the final 20% (1989–1990).

Colab results go to `MyDrive/RNN_Exercise/output/`: models, predictions,
metrics, comparison tables, plots, and `summary.md`.
Local execution saves to `./output/`. Rerunning replaces matching output files.

Results and conclusions are generated from actual runs inside the notebook.
The exact downloaded dataset, random seeds, library versions, and chosen
configuration are saved for reproducibility. Scores can vary slightly across
platforms. Do not use test results to make further tuning decisions.

## LSTM notebook

Open `lstm.ipynb` in Colab and run all. Uses Keras `LSTM` with the same splits,
feature comparisons, and validation-based tuning. Results are saved to
`MyDrive/RNN_Exercise/output/lstm/` (locally: `output/lstm/`).

## GRU notebook

Open `gru.ipynb` in Colab and run all. Uses Keras `GRU` with the same splits,
feature comparisons, and validation-based tuning. Results are saved to
`MyDrive/RNN_Exercise/output/gru/` (locally: `output/gru/`).
