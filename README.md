# 🛒 Problem Description

This project uses a **public Mercado Libre dataset** containing **100,000 product listings** in **JSON Lines** format. Each record includes multiple attributes about a marketplace item: its title, description, category, seller information, price, shipping features, and other metadata.

## ⚠️ Problem Context

A key challenge in the dataset is the **inconsistent or missing classification of product condition**. Many listings do not clearly specify whether the item being sold is **new** or **used**, and the available textual fields are often noisy, unstructured, and difficult to analyze manually.

This lack of reliable labeling limits the ability to:

- Study market dynamics  
- Compare price distributions  
- Improve product recommendation systems  
- Understand buyer/seller behavior  

## 🎯 Project Objective

The goal of this project is to **build a machine learning classifier** capable of predicting whether a product listing corresponds to a **new** or **used** item based on the attributes available in the dataset.

The dataset includes a variable created through internal tagging that labels a subset of listings as “new” or “used”. This serves as the **ground truth** for supervised learning.

## 🔍 What the Project Aims to Achieve

By analyzing the structured and unstructured fields of each listing, this project aims to:

- 🧩 **Identify which features best distinguish new products from used ones**  
- 🤖 **Train a classification model** that can generalize to unlabeled listings  
- 🏷️ **Improve the characterization of marketplace items**  
- ⚙️ **Provide a tool that supports sellers, buyers, and automated systems** in estimating product condition  

## 📊 Scope of Work

The project includes:

- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model development  
- Model evaluation  

The final outcome is a **classifier** capable of predicting whether a product listing is **new or used** using only the metadata provided in the dataset.
