# DL2025_Team2_StockPrediction
Introduction to Deep Learning Course 2025 of CCU Team Project

## Model introduction

A GRU based stock price prediction model with GRU layers.
Implemented with Pytorch and trained on [stocknet-dataset](https://github.com/yumoxu/stocknet-dataset/)


{xu-cohen-2018-stock,
    title = "Stock Movement Prediction from Tweets and Historical Prices",
    author = "Xu, Yumo  and
      Cohen, Shay B.",
    editor = "Gurevych, Iryna  and
      Miyao, Yusuke",
    booktitle = "Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2018",
    address = "Melbourne, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P18-1183/",
    doi = "10.18653/v1/P18-1183",
    pages = "1970--1979",
    abstract = "Stock movement prediction is a challenging problem: the market is highly stochastic, and we make temporally-dependent predictions from chaotic data. We treat these three complexities and present a novel deep generative model jointly exploiting text and price signals for this task. Unlike the case with discriminative or topic modeling, our model introduces recurrent, continuous latent variables for a better treatment of stochasticity, and uses neural variational inference to address the intractable posterior inference. We also provide a hybrid objective with temporal auxiliary to flexibly capture predictive dependencies. We demonstrate the state-of-the-art performance of our proposed model on a new stock movement prediction dataset which we collected."
}