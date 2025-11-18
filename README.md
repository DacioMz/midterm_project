Problem Description

This project uses a public Mercado Libre dataset containing 100,000 product listings in JSON Lines format. Each record includes multiple attributes about a marketplace item: its title, description, category, seller information, price, shipping features, and other metadata.

A key challenge in the dataset is the inconsistent and often missing classification of product condition. Many listings do not clearly specify whether the item being sold is new or used, and the available textual fields are noisy, unstructured, and difficult to analyze manually. This lack of reliable labeling limits the ability to study market dynamics, compare price distributions, or improve product recommendation systems.

To address this issue, the goal of the project is to build a machine learning model capable of predicting whether a product listing corresponds to a new or used item based on the attributes available in the dataset. The dataset includes a variable that labels a subset of listings as “new” or “used”, created through internal tagging, and this serves as the ground truth for supervised learning.

By analyzing the structured and unstructured fields of each listing, the project aims to:

Understand which features help distinguish new products from used ones

Build a classification model that can generalize to unlabeled listings

Improve the characterization of marketplace items

Provide a tool that can support sellers, buyers, and automated systems in estimating item condition

This project includes exploratory data analysis (EDA), feature engineering, model development, and model evaluation. The final outcome is a classifier that predicts whether a product listing is new or used based on its available metadata.# midterm_project