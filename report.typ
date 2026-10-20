#set document(title: "Melbourne Temperature Forecasting", author: "")
#set page(paper: "a4", margin: (x: 17mm, y: 15mm), numbering: "1")
#set text(font: "Libertinus Serif", size: 10pt)
#set par(justify: true, leading: 0.5em, spacing: 0.65em)
#set heading(numbering: "1.")
#show heading: set text(font: "Arial", size: 11pt, weight: "bold")
#show heading: set block(above: 0.9em, below: 0.45em)
#set figure(gap: 4pt)
#show figure.caption: set text(size: 8.5pt)
#set table(inset: (x: 6pt, y: 4pt), stroke: 0.4pt + luma(78%),
  fill: (x, y) => if y == 0 { luma(94%) })

#align(center)[
  #text(font: "Arial", size: 19pt, weight: "bold")[Melbourne Temperature Forecasting]
  #v(3pt)
  Vanilla RNN, LSTM, and GRU · One-day-ahead prediction
]
#v(5pt)
All experiments were run on Google Colab with Google Drive mounted.
Outputs were saved to `MyDrive/RNN_Exercise/output/`.

= Dataset
The GitHub dataset contains *3,650 daily minimum temperatures* in Melbourne,
1 January 1981–31 December 1990. `Date` is the timestamp; `Temp` is the recorded
minimum temperature in °C. The target is the next day's minimum temperature,
using past temperatures and derived calendar features. There are no empty cells
or duplicate dates; 31 December 1984 and 1988 are absent. Temperatures range
from 0.0 to 26.3°C, with mean 11.18°C. Recorded extremes are retained.

= EDA
`eda.ipynb` shows strong *annual seasonality* with daily noise: February averages
15.37°C and July 6.69°C. Annual means fluctuate without a steady trend; there is
no clear separate multi-year cycle. Calendar-day correlations are 0.77 at lag 1
and 0.53 at lag 365, supporting recent-history and seasonal inputs.

#figure(image("report_assets/eda_temperature.png", width: 100%),
  caption: [Original EDA plot: daily observations and trailing 30-day mean. Full-series descriptive analysis.])

= Vanilla RNN
Keras `SimpleRNN` updates a tanh hidden state from the current input and previous
state; `Dense(1)` produces the temperature forecast.

*Protocol.* Following the assignment, the first 80% (2,920 observations) is
development and the final 20% (730 days, 1989–1990) is test. Development is split
into training (1981–1986) and validation (1987–1988). A common 424-day warm-up
leaves 1,766 training and 730 validation targets. Missing input dates are
forward-filled using past values; missing targets are excluded. Min–max scalers
are fitted on training data only. Forecasts use the previous day's actual
temperature, including during test; weights stay fixed during evaluation.

*Experiments.* Each model has 22 trials: 16 feature/window combinations, then
six settings on its best feature/window pair. Windows are 7, 14, 30, and 60 days.
Features are temperature history; history plus annual sine/cosine; history plus
t−365; or all of these plus trailing 7/30-day means through t−1. The table gives
the best validation RMSE per feature group; parentheses show its window.

#table(columns: (1.25fr, 1fr, 1fr, 1fr),
  [*Inputs*], [*Vanilla RNN*], [*LSTM*], [*GRU*],
  [Recent temperature], [2.4158 (60d)], [2.4820 (30d)], [2.4016 (30d)],
  [History + calendar], [2.3740 (14d)], [2.5492 (30d)], [2.3636 (7d)],
  [History + annual lag], [2.4052 (14d)], [2.5219 (60d)], [2.3871 (60d)],
  [Combined], [2.3689 (7d)], [2.5542 (30d)], [2.3516 (14d)],
)

Adam minimizes MSE with batch size 64, gradient clipping at 1, seed 42, and a
60-epoch cap. Early stopping has patience 8 and restores the best validation
weights. Tuning varies units, one/two layers, and learning rates 0.0003–0.003.
The RNN winner uses combined inputs, a 7-day window, and 32 units in one layer.

#pagebreak()
= LSTM & GRU
LSTM uses a cell state with input, forget, and output gates. GRU uses update and
reset gates with a hidden state. Both follow the same data and selection protocol.
Initial models are one 8-unit LSTM layer and two 50-unit GRU layers; the RNN starts
with two 50-unit layers. LSTM/GRU tuning tests 16/32/64 units; RNN tuning tests
16/32/50 units. All selected learning rates are 0.001.

*Selected configurations.* Units × layers describes the recurrent architecture.
Each winner is refitted on eligible targets in the full first 80%, with newly
fitted development-only scalers and its validation-selected epoch count.

#table(columns: (1.05fr, 1fr, 0.6fr, 0.85fr, 0.6fr, 0.9fr),
  [*Model*], [*Inputs*], [*Window*], [*Units × layers*], [*Epochs*], [*Val. RMSE*],
  [Vanilla RNN], [combined], [7d], [32 × 1], [60], [2.3686],
  [LSTM], [temperature], [30d], [32 × 1], [60], [2.4010],
  [GRU], [combined], [14d], [50 × 2], [45], [2.3516],
)

= Results Comparison
RMSE is $sqrt(1/n sum_(i=1)^n (hat(y)_i - y_i)^2)$ in °C; lower is better.
All models are evaluated on the same 730 test dates. Settings and epochs were
selected using validation RMSE only; test results below compare the final models.

#table(columns: (1.8fr, 1fr, 1.3fr),
  [*Model / baseline*], [*Test RMSE (°C)*], [*Reduction vs. persistence*],
  [Vanilla RNN], [2.3064], [7.04%],
  [*LSTM*], [*2.2013*], [*11.27%*],
  [GRU], [2.2641], [8.74%],
  [Persistence (t−1)], [2.4809], [—],
  [Trailing 7-day mean], [2.5955], [—],
  [Seasonal (t−365)], [3.7404], [—],
)

*LSTM performs best on test*, reducing RMSE by 11.27% versus persistence.
Vanilla RNN and GRU also improve on persistence by 7.04% and 8.74%; all three
beat the trailing-mean and seasonal baselines. GRU has the lowest selected
validation RMSE, so the validation and test rankings differ.

#figure(image("report_assets/model_comparison.png", width: 100%),
  caption: [First 90 test days, plotted from the saved predictions of all three models.])

*What helped most?* In the controlled LSTM trials, increasing the temperature-only
window from 7 to 30 days lowered validation RMSE from 2.7002 to 2.4820; increasing
units from 8 to 32 then lowered it to 2.4010, plausibly through more context and
capacity. Combined inputs gave the strongest RNN and GRU feature trials, suggesting
that seasonal context and smoothed recent history were useful for those models.
These are single-seed historical results; they do not establish a general
advantage for one architecture or guarantee the lowest achievable RMSE.

= Sources
#set text(size: 8.5pt)
- `RNN_Exercise.pdf` (provided assignment); `eda.ipynb`; model notebooks and exported `output/` tables, histories, and predictions. LSTM results use its saved exports; its local notebook has no saved cell outputs.
- #link("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")[Daily minimum temperatures — GitHub].
- #link("https://www.geeksforgeeks.org/machine-learning/time-series-forecasting-using-recurrent-neural-networks-rnn-in-tensorflow/")[Vanilla RNN forecasting — GeeksforGeeks].
- LSTM: #link("https://medium.com/data-science-collective/future-forecasting-of-time-series-using-lstm-a-quick-guide-for-business-leaders-370661c574c9")[Forecasting guide] · #link("https://medium.com/apes-ai/time-series-forecasting-using-lstm-code-walkthrough-3aae01645491")[Code walkthrough].
- GRU: #link("https://sharmasaravanan.medium.com/time-series-forecasting-using-gru-a-step-by-step-guide-b537dc8dcfba")[Forecasting guide] · #link("https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/")[Gated recurrent unit networks — GeeksforGeeks].
