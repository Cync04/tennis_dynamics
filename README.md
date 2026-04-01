# tennis_dynamics
This project aims to compute tennis dynamics for (fill this in later)

---

# **aceserve_model.py — Summary**

`aceserve_model` builds a complete machine‑learning pipeline to analyze how serve characteristics influence point outcomes in professional tennis.
### **What aceserve_model Does**

#### **1. Data Cleaning & Preparation**
`aceserve_model` loads the Wimbledon dataset and performs all preprocessing steps needed for modeling:
- Removes invalid or missing values  
- Standardizes serve‑related fields (speed, serve number, serve width)  
- Converts the dataset into a *server‑perspective* format:
  - Whether the server hit an ace  
  - Whether the server won the point  
  - Whether the server faced break‑point pressure  

#### **2. Feature Engineering**
`aceserve_model` constructs the features used for prediction:
- Normalized serve speed  
- First vs second serve  
- Break‑point pressure  
- Rally length  
- One‑hot encoded serve width categories (C, BC, B, BW, W)

Two feature sets are created:
- **Ace model features**  
- **Point‑win model features** (ace features + rally length)

#### **3. Logistic Regression Models**
`aceserve_model` trains two separate logistic regression models:
- **Ace Model** → predicts probability of an ace  
- **Point‑Win Model** → predicts probability the server wins the point  

Each model includes:
- Train/test split evaluation  
- 5‑fold cross‑validation for more reliable accuracy  
- Coefficient outputs to show which features matter most  

#### **4. Visualizations**
`aceserve_model` generates three plots to help interpret the models:
1. **Point‑Win Model Coefficients**  
2. **Point‑Win Probability vs Serve Speed**  
3. **Ace Probability vs Serve Speed (by Serve Width)**  

---
