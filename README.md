The Free Pizza Project
For CS4662 - Advanced Machine & Deep Learning Project

Dataset: https://www.kaggle.com/datasets/kaggle/random-acts-of-pizza/data

The goal of this project is to analyze Reddit post data from the r/Random_Acts_Of_Pizza subreddit to predict whether a user's request for free pizza will be successfully fulfilled. The dataset originates from a Kaggle competition based on a Stanford research study on altruistic behavior in online communities.

The primary objective is to maximize the area under the ROC curve (AUC) of our models!

We use conda/mamba for dependencies:
Install conda or mamba and run `conda-lock install --name cs4662-env conda-lock.yml` 

and then `mamba activate cs4662-env`

if you want to add dependencies, change the `environment.yml` and then run `make lock`
