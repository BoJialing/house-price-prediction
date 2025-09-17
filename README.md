# House Price Prediction - Washington State

Predictive modeling for Washington state house prices using Ridge Regression and XGBoost with comprehensive feature engineering (100+ features). This project demonstrates advanced data preprocessing, exploratory data analysis, and feature engineering techniques applied to real estate price prediction.

## 🎯 Project Overview

This project tackles house price prediction using a dataset of 9,200 residential property transactions from 2014 in the Seattle metropolitan area. The focus is on pushing the limits of linear regression through strategic feature engineering while benchmarking against advanced tree-based models.

**Key Results:**
- **Ridge Regression**: 17.08% MAPE with 90 engineered features
- **XGBoost Benchmark**: 11.24% MAPE 
- **Feature Engineering**: 100+ features created from 18 original variables

## 🔧 My Contributions

**Primary Role: Feature Engineering Lead & EDA Specialist**

### Exploratory Data Analysis
- Conducted comprehensive statistical analysis of 9,200 property records
- Identified and addressed data quality issues (zero prices, inconsistent measurements)
- Performed distributional analysis and outlier detection for all numerical features
- Analyzed categorical feature distributions and rare category consolidation

### Advanced Feature Engineering (100+ Features)
- **Log Transformations**: Applied to price and size variables to address right-skewed distributions
- **Ratio Features**: Created meaningful proportions (living_to_lot, basement_ratio, avg_room_size)
- **Interaction Terms**: Engineered bedroom-bathroom and condition-age interactions
- **Spatial Features**: Implemented KNN-based neighborhood pricing using property characteristics
- **Categorical Encoding**: Optimized handling of 44 cities and 77 zip codes with frequency-based consolidation
- **Premium Property Indicators**: Combined features for waterfront + excellent view properties

### Data Preprocessing Pipeline
- Systematic missing value analysis and treatment
- Outlier capping at 99th percentile for extreme values
- Multicollinearity assessment using Variance Inflation Factor (VIF)
- Train-test split with stratified sampling to ensure representative distributions

## 📁 Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/
│   │   └── house_dataset.csv          # Original dataset (9,200 records)
│   └── processed/
│       ├── train_final.csv            # Processed training set
│       └── test_final.csv             # Processed test set
├── notebooks/
│   ├── EDA_and_feature_engineering.ipynb  # Primary analysis notebook
│   └── EDA_and_feature_engineering.html   # HTML export for easy viewing
├── src/                               # Source code modules
├── results/                           # Model outputs and visualizations
└── requirements.txt                   # Dependencies
```

## 🛠 Technical Implementation

### Data Processing Techniques
- **Statistical Validation**: Verified internal consistency of size measurements
- **Feature Scaling**: Standardized numerical features for linear modeling
- **Category Consolidation**: Reduced dimensionality while preserving market segmentation
- **Assumption Testing**: Linearity, normality, and homoscedasticity validation for regression models

### Feature Selection Methods
- **RFECV**: Recursive Feature Elimination with Cross-Validation (90 features selected)
- **LassoCV**: L1 regularization for automatic feature selection
- **VIF Analysis**: Multicollinearity detection and mitigation

### Model Development
- **Ridge Regression**: L2 regularization for robust linear modeling
- **XGBoost Benchmark**: Tree-based model for performance comparison
- **Cross-Validation**: 5-fold CV for reliable performance estimation

## 📊 Key Technical Insights

### Feature Importance Hierarchy
1. **log_sqft_living**: Strongest predictor (r=0.67 with log_price)
2. **knn_avg_price**: Local market context integration
3. **age**: Property depreciation effects
4. **ratio_living_to_lot**: Land utilization efficiency
5. **bed_bath_interact**: Configuration optimization

### Data Quality Improvements
- Removed records with zero prices and bedroom counts
- Corrected 2 inconsistent size measurements
- Consolidated 44 cities into 18 categories (including 'Other')
- Reduced 77 zip codes to 58 meaningful segments

### Statistical Transformations
- Log transformation improved price prediction linearity
- Ratio features captured property efficiency metrics
- Interaction terms revealed non-linear relationships
- KNN features incorporated neighborhood effects

## 🚀 Technical Skills Demonstrated

**Data Science Pipeline:**
- End-to-end data preprocessing from raw to model-ready
- Advanced feature engineering with domain knowledge integration
- Statistical assumption validation and model diagnostics
- Performance benchmarking across algorithm families

**Python Libraries:**
- **pandas**: Data manipulation and transformation
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning pipeline and model selection
- **matplotlib/seaborn**: Statistical visualization and EDA
- **xgboost**: Gradient boosting implementation

**Statistical Methods:**
- Regression diagnostics and assumption testing
- Cross-validation for model selection
- Feature importance analysis
- Multicollinearity assessment

## 📈 Business Impact

This feature engineering approach demonstrates how thoughtful data preprocessing can significantly improve model interpretability while maintaining competitive performance. The 17.08% MAPE achieved with Ridge Regression provides a transparent, explainable model suitable for:

- **Policy Analysis**: Clear coefficient interpretation for regulatory decisions
- **Investment Strategy**: Feature importance insights for property valuation
- **Market Understanding**: Quantified relationships between property characteristics and prices

## 🔄 Future Enhancements

- **Polynomial Features**: Higher-order terms for capturing non-linear relationships
- **Temporal Features**: Seasonal and market cycle variables
- **External Data**: Economic indicators and demographic features
- **Ensemble Methods**: Combining interpretable and high-performance models

## 📋 Requirements

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
xgboost>=1.5.0
jupyter>=1.0.0
```

## 📧 Contact

**Jialing Bo**  
Data Science Graduate Student | National University of Singapore  
Email: bojialing@gmail.com  
LinkedIn: [Connect with me](https://linkedin.com/in/your-profile)
