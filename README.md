# What Makes Me Happy – A Personal Data Science Project

## 📌 Overview

This side project was born from a **New Year's resolution in January 2025**:  
> _"I want to be as happy as possible every single day this year."_

The challenge? I didn’t actually know what makes me happy.

So, I turned to data.

Over the course of four weeks, I recorded daily binary data about simple habits and activities, along with a happiness score (0–10). This project is an early attempt to understand and model what affects my daily wellbeing.

---

## 🧠 Hypothesis

> _"I don't know what makes me happy on a given day—so I need to test it."_

I started with tracking:

1. **Wake up early** (target: 6:30 AM)  
2. **Did my morning routine**  
3. **Went to bed early** (target: 10:00 PM)  
4. **Did house chores** (≥1 hour)  
5. **Worked out** (≥30 minutes)  
6. **Read personal emails**  
7. **Happiness score** (0–10)  

---

## 📊 Data

- Manually collected and stored in a CSV file: `How_was_my_day.csv`
- The dataset includes 28 days (January 1–28, 2025)
- A synthetic `date` column was generated, as the original data lacked timestamps

---

## 📈 Exploratory Data Analysis (EDA)

- Distribution of daily happiness scores  
- Trend over time  
- Correlation heatmaps between habits and happiness

### Key Findings:
- **Workouts** and **house chores** had positive correlation with happiness
- **Waking up early** was surprisingly negatively correlated with other productive habits
- No clear upward/downward trend in overall mood during the 4-week period

---

## 🔬 Experiment 1: Do Weekends Make Me Happier?

### Method:
- Labeled days as **Weekday** or **Weekend**
- Used the **Mann-Whitney U test** (non-parametric)
- Also conducted an **A/A test** for robustness

### Result:
> ❌ **No significant difference** in happiness between weekends and weekdays

---

## 🤖 Predictive Modeling

Trained a **CatBoost Regressor** to understand which habits are most predictive of my daily happiness.

### Setup:
- Features: All tracked habits + weekend indicator
- Target: Happiness score (`val`)
- Model: CatBoostRe
