# Study-HIP

This repository contains a Python adaptation and analysis of the [HIP: Hawkes Intensity Process](https://github.com/andrei-rizoiu/hip-popularity) model originally proposed by Rizoiu et al. for modeling the popularity of online content.

>  **Course Project**: ANU COMP8880 - Network Science  
>  **Focus**: Time series modeling of YouTube video popularity using self-exciting point processes and ARIMA baselines

---

## Project Structure

```
Study-HIP/
├── data/                  # Serialized dataset (.p file)
├── src/                   # Supplementary course files (sliders,requirement,video)
├── comparison.py          # ARIMAX vs HIP comparison and evaluation (RMSE-based)
├── pyhip.py               # Core HIP model logic (intensity, prediction, fitting)
├── pyhip_example.py       # Basic usage of HIP fitting and forecasting
├── tool.ipynb             # Jupyter notebook for visual analysis and case studies
├── README.md              # This file
└── .gitignore             # Git ignore config
```

---

## About This Project

This project:

- **Evaluates HIP vs ARIMAX (baseline)** using RMSE on a sampled subset of videos;
- Supports **batch evaluation**, **visual case studies**, and **parameter sensitivity tests** (`num_initialization`);
- Provides Jupyter notebook `tool.ipynb` for plotting results and inspecting model behavior.

---

## Getting Started

### 1. Prepare Data

Ensure the `active-dataset.p` file is placed under the `./data` directory. You can obtain this from the original [HIP repository](https://github.com/andrei-rizoiu/hip-popularity) or by generating it yourself from YouTube metadata.

This file is a serialized Python dictionary where:

```python
active_videos: Dict[str, Tuple[List[int], List[int], List[int]]]
```
Each key is a YouTube video ID, and the corresponding value is a tuple of three sequences:

- `daily_share`: number of shares per day for the video  
- `daily_view`: number of views per day for the video  
- `daily_watch`: total watch time (in seconds) per day  
- 
### 2. Run Experiments

To run batch evaluation on a sample of 100 videos:

```bash
python comparison.py
```

This will fit both HIP and ARIMAX models on each video and print RMSE comparisons.

To conduct an in-depth case study on a single video:

```python
# Inside comparison.py
case_study(data_source=active_videos, video_id='X0ZEt_GZfkA', num_train=90, num_test=30)
```

---

## Visualization

Use `tool.ipynb` to:

- Plot original view/share series;
- Visualize prediction results across different models;
- Explore how HIP performance varies with `num_initialization`.

---

## Evaluation Metric

- **Root Mean Squared Error (RMSE)** is used as the main evaluation criterion.
- Comparison is done between:
  - **ARIMAX** (autoregressive model with exogenous share features)
  - **HIP** (self-exciting point process based model)

---

## Reference

> Rizoiu M A, Xie L, Sanner S, et al.  
> **"Expecting to be HIP: Hawkes Intensity Processes for Social Media Popularity."**  
> Proceedings of the 26th international conference on world wide web. 2017: 735-744. [[link]](https://arxiv.org/pdf/1602.06033)

Original codebase: [https://github.com/andrei-rizoiu/hip-popularity](https://github.com/andrei-rizoiu/hip-popularity)

