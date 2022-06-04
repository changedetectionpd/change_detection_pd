# Change Detection with Probabilistic Models on Persistence Diagrams

This repository contains the code associated with the paper "Change Detection with Probabilistic Models on Persistence Diagrams"

- circular_dataset.ipynb is a jupyter notebook to run the experiments on a synthetic circular dataset.
- chaotic_time_series.ipynb is a jupyter notebook to run the experiments on a synthetic chaotic time-series dataset.
- financial_time_series.ipynb is a jupyter notebook to run the experiments on a real-world financial time-series dataset.

We use a part of code at https://github.com/IbarakikenYukishi/differential-mdl-change-statistics for SMDL, BOCD, and some evaluation functions, with slight modification. We also use a modified version of TimeDelayEmbedding in GUDHI (https://gudhi.inria.fr/).