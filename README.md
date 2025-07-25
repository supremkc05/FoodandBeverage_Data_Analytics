**food and beverage data analysis**
This project analyzes a dataset of food and beverage shops, performing data cleaning, exploratory data analysis (EDA), visualizations, and predictive modeling using machine learning.

---

## 🗂️ Dataset

- **File:** `Dataset_for_Food_and_Beverages.csv`
- **Description:** Contains records of various shops (e.g., cafes, bakeries, restaurants) with attributes like shop type, ratings, foot traffic, marketing efforts, and yearly sales.

---

## 🛠️ Tools & Libraries

- Python 3.x
- `pandas` - Data manipulation and analysis
- `matplotlib` - Static plotting
- `seaborn` - Statistical data visualization
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning
- `streamlit` - Web dashboard framework
- `numpy` - Numerical computing

---
## 📌 Project Workflow

### 1️⃣ Data Cleaning

- Sorted data by `Shop_Name`.
- Dropped unnecessary columns (`Shop_Id`).
- Reset dataframe index.
- Standardized `Shop_Type` values.
- Mapped `Shop_Website` and `Marketing` to binary (0 = No, 1 = Yes).
- Categorized `Rating` into `Low`, `Medium`, `High`.

---
### 2️⃣ Exploratory Data Analysis & Visualizations

✅ **Shop Type Distribution**  
- Pie chart of shop type counts.

✅ **Foot Traffic by Shop Type**  
- Histogram showing average foot traffic for each shop type.

✅ **Rating by Shop Type**  
- Line plot of mean ratings by shop type.

✅ **Marketing vs Yearly Sales**  
- Scatter plot of marketing presence vs sales.

✅ **Website vs Yearly Sales**  
- Scatter plot of website presence vs sales.

✅ **Foot Traffic vs Yearly Sales**  
- Scatter plot analyzing correlation.

✅ **Rating vs Yearly Sales**  
- Line plot of rating and sales relationship.

---
### 3️⃣ Predictive Modeling

- **Target:** `Yearly_Sales`
- **Features:** Shop type (encoded), website, marketing, foot traffic, rating.
- **Model:** Random Forest Regressor
- **Validation:** Train-test split (80-20), cross-validation
- **Metrics:** R² score, Mean Absolute Error

---

## 📝 Results Summary

- **R² Score:** Model performance metric indicating variance explained.
- **MAE:** Average prediction error in sales values.
- **Insights:** Shops with marketing, website presence, higher foot traffic, and better ratings generally have higher sales.

---

## 🚀 How to Run

### Option 1: Jupyter Notebook
1️⃣ Clone/download the notebook.  
2️⃣ Ensure dependencies are installed:
```bash
pip install pandas matplotlib seaborn plotly scikit-learn
```
3️⃣ Run the notebook:
```bash
jupyter notebook ITS69304_SupremKhatri_IndividualAssignment.ipynb
```
4️⃣ Make sure `Dataset_for_Food_and_Beverages.csv` is in the same directory.

### Option 2: Interactive Streamlit Dashboard
1️⃣ Install required dependencies:
```bash
pip install -r requirements.txt
```
2️⃣ Run the Streamlit dashboard:
```bash
streamlit run dashboard.py
```
3️⃣ Open your browser to the provided local URL (usually http://localhost:8501)

---

## 🎯 Dashboard Features

The Streamlit dashboard includes:
- **📈 Overview Page:** Key business metrics and summary statistics
- **🔍 Data Explorer:** Interactive data filtering and exploration
- **📊 Visualizations:** Interactive charts and plots
- **🤖 Machine Learning:** Model performance and feature importance
- **🔮 Predictions:** Real-time predictions for new shop scenarios


