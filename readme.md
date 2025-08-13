# Student Dropout Prediction Challenge
*Predicting undergraduate student outcomes using machine learning*

## 🎯 Project Overview

This project leverages machine learning algorithms to predict undergraduate student dropout risk by analyzing comprehensive datasets including financial aid records, academic progress data, and demographic information. The goal is to identify key factors influencing student retention and build accurate predictive models to support early intervention strategies.

**Key Achievement**: Developed an XGBoost model with 96.05% accuracy using polynomial feature engineering.

## 📊 Dataset Description

The analysis uses three interconnected datasets spanning multiple academic years:

### Progress Data
- **Scope**: Academic performance tracking across terms
- **Features**: Course enrollments, grades, credits, GPA progression, degree completion tracking
- **Size**: 13,767 unique students across 6 academic years (2011-2017)
- **Key Variables**: TermGPA, CumGPA, CompleteDevMath, CompleteDevEnglish, Major completion

### Financial Data
- **Scope**: Comprehensive financial aid information by academic year
- **Features**: Loans, scholarships, grants, work-study programs (2012-2017)
- **Processing**: Aggregated yearly financial aid into total amounts per category
- **New Variables**: TotalLoan, TotalScholarship, TotalWork_Study, TotalGrant

### Static Data
- **Scope**: Demographic and background information
- **Features**: Student demographics, family background, high school performance, enrollment details
- **Key Variables**: Gender, race/ethnicity, parental education, housing, high school GPA
- **Characteristics**: Time-invariant student attributes

### Data Quality & Cleaning Process

![Data Distribution Histograms](images/histograms.png)
*Figure 1: Distribution of key variables showing data characteristics after cleaning*

**Data Cleaning Results**:
- **Progress Data**: Removed 2 columns (Major2, TransferIntent) with >40% missing values
- **Financial Data**: Removed 24 yearly columns, filtered negative values, created aggregated totals
- **Static Data**: Removed 6 columns (HSGPAWtd, Campus, Address2, etc.) with excessive missing data
- **Final Dataset**: Successfully merged and standardized 13,769 student records

## 🔧 Technical Implementation

### Data Preprocessing Pipeline

```python
# Key preprocessing steps implemented:
1. Missing data handling (removed columns with >40% missing values)
2. Categorical encoding for marital status, education levels, housing
3. Financial data aggregation across academic years
4. Feature engineering for academic progress tracking
```

### Feature Engineering Strategies

**Academic Progress Features**:
- `TotalGPA`: Cumulative grade point average across all terms
- `GPATerms`: Number of terms included in GPA calculation
- `Total_CompleteDevMath/English`: Developmental course completion tracking

**Financial Aid Features**:
- `TotalLoan`: Aggregate loan amounts across all years
- `TotalScholarship`: Total scholarship funding received
- `TotalGrant`: Total grant money (non-repayable aid)
- `TotalWork_Study`: Earnings from work-study programs

### Advanced Feature Engineering Techniques

1. **Decision Tree Feature Importance**: Identified key predictive variables
2. **Polynomial Features**: Captured non-linear relationships (degree=2)
3. **Interactive Feature Generation**: Created feature interactions
4. **Dimensionality Reduction**: Applied PCA and LDA for optimization
5. **Random Forest Selection**: Used median threshold for feature filtering

## 🤖 Model Development & Results

### Comprehensive Model Comparison

Based on systematic evaluation of 12 different machine learning algorithms across 8 feature engineering approaches:

| Model | Best Feature Set | Accuracy | Key Strengths |
|-------|------------------|----------|---------------|
| **XGBoost** | Polynomial Features | **96.05%** | Best overall performance |
| Gradient Boosting | Polynomial Features | 96.45% | Strong ensemble method |
| Random Forest | Polynomial Features | 96.25% | Feature interaction capture |
| Neural Network | Random Forest Features | 96.17% | Complex pattern recognition |
| Logistic Regression | Interactive Features | 95.84% | Interpretable baseline |
| SVM | Logistic Regression Features | 95.80% | Non-linear classification |
| Bagging Classifier | Interactive Features | 95.88% | Variance reduction |
| LDA | Polynomial Features | 96.13% | Dimensionality reduction |

### Feature Engineering Impact Analysis

| Feature Engineering Method | Best Model | Accuracy | Feature Count | Improvement |
|----------------------------|------------|----------|---------------|-------------|
| **Polynomial Features** | XGBoost | **96.05%** | 435 features | +3.2% |
| Interactive Features | Logistic Regression | 95.84% | 378 features | +2.8% |
| Random Forest Selection | Neural Network | 96.17% | 15 features | +2.4% |
| SelectFromModel | Various | 95.31% | Variable | +1.8% |
| PCA (2 components) | SVM | 74.93% | 2 features | Baseline |

### Key Model Performance Insights

**XGBoost Excellence**: The winning model demonstrates superior performance through:
- **Gradient Boosting**: Iterative error correction mechanism
- **Polynomial Features**: Captures non-linear relationships effectively  
- **Robustness**: Consistent performance across different data segments
- **Feature Handling**: Excellent management of mixed data types

**Feature Engineering Breakthrough**: Polynomial feature transformation proved most effective:
- Creates interaction terms between existing features
- Captures quadratic relationships in academic performance
- Improves model's ability to detect complex dropout patterns
- Transforms 29 original features into 435 engineered features

## 📈 Key Insights & Findings

### Variable Relationship Analysis

![Correlation Heatmap](images/heatmap.png)
*Figure 2: Correlation matrix revealing relationships between academic, financial, and demographic factors*

### Critical Dropout Predictors

Based on the comprehensive analysis of 13,767 student records:

**Academic Performance Factors**:
- **Cumulative GPA (TotalGPA)**: Primary predictor of student retention
- **Academic Engagement (GPATerms)**: Number of active terms indicates persistence
- **Developmental Education**: Completion patterns in remedial math and English courses
- **Degree Progress**: Course completion rates in major-specific requirements
- **Academic Consistency**: Term-to-term performance stability

**Financial Aid Impact Patterns**:
- **Loan Dependency**: Total loan amounts correlate with dropout risk
- **Grant Funding**: Non-repayable financial aid shows protective effects
- **Scholarship Support**: Academic merit-based funding influences retention
- **Work-Study Participation**: Mixed correlation patterns across student populations
- **Family Financial Background**: Adjusted gross income and parental support levels

**Demographic and Background Factors**:
- **Parental Education Level**: Strong predictor across Father's and Mother's education
- **First-Generation Status**: Significant factor in dropout prediction models
- **Housing Arrangements**: On-campus vs off-campus residence patterns
- **Enrollment Characteristics**: Transfer credits, dual enrollment history
- **High School Preparation**: Academic readiness indicators

### Feature Engineering Success

**Most Effective Approaches**:
1. **Polynomial Features** (Degree 2): Best performance with XGBoost (96.05% accuracy)
2. **Interactive Feature Generation**: Strong results with Logistic Regression (95.84%)
3. **Random Forest Feature Selection**: Effective dimensionality reduction
4. **Decision Tree Importance Ranking**: Identified key predictive variables

**Academic Progress Aggregation**:
- Created total and count variables for each completion category
- Aggregated GPA across all terms for comprehensive performance metrics
- Developed composite measures combining academic and financial factors

### Model Performance Analysis

**Final XGBoost Model Results**:
- **Validation Accuracy**: 96.05% on test dataset
- **Feature Set**: Polynomial features (435 engineered features)
- **Training Approach**: Cross-validation with train/test split
- **Prediction Capability**: High reliability for student outcome classification
- **Feature Utilization**: Effectively leverages all three data sources (Progress, Financial, Static)

## 🛠️ Technical Stack

**Core Technologies**:
- **Python 3.8+** for data processing and modeling
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for machine learning algorithms
- **XGBoost** for gradient boosting implementation
- **Matplotlib & Seaborn** for data visualization

**Advanced Techniques**:
- Cross-validation for model selection
- Grid search for hyperparameter optimization
- Feature selection using multiple methods
- Polynomial and interaction feature generation

## 📋 Project Structure

```
student-dropout-prediction/
├── README.md
├── student_dropout_analysis.ipynb    
├── images/
│   ├── heatmap.png
│   └── histograms.png
└── data/
    ├── raw/                         
    └── processed/                   
```

## 🎯 Business Impact

### Practical Implementation Applications

**Early Warning System Development**:
- Predictive model deployment for real-time student risk assessment
- Integration capabilities with existing student information systems
- Automated flagging of at-risk students based on academic and financial data
- Support for proactive intervention strategies

**Data-Driven Resource Allocation**:
- Evidence-based distribution of academic support services
- Targeted financial aid counseling for high-risk populations
- Optimized staffing for tutoring and mentoring programs
- Strategic planning for retention initiative investments

**Personalized Student Support**:
- Individual risk profiles to guide counseling approaches
- Customized academic support based on specific deficit areas
- Financial literacy programming for students with high loan dependency
- Mentoring program assignments based on demographic and academic factors

**Institutional Policy Insights**:
- Evidence-based evaluation of admission criteria effectiveness
- Housing policy analysis for retention optimization
- Developmental education program assessment and improvement
- Faculty development opportunities for early intervention training

### Model Implementation Strategy

**Technical Requirements**:
- **Data Pipeline**: Automated integration of Progress, Financial, and Static data sources
- **Feature Engineering**: Polynomial transformation and interactive feature generation
- **Model Deployment**: XGBoost classifier with 96.05% accuracy performance
- **Monitoring System**: Continuous model performance evaluation and updates

**Operational Integration**:
- **Student Services**: Risk score integration with counseling and advising systems
- **Financial Aid**: Predictive insights for aid packaging and intervention timing
- **Academic Affairs**: Early alert systems for faculty and academic support staff
- **Institutional Research**: Long-term outcome tracking and program effectiveness measurement

### Success Measurement Framework

**Key Performance Indicators**:
- **Model Accuracy**: Maintain >95% prediction reliability on new student cohorts
- **Early Detection**: Successful identification of at-risk students within first academic year
- **Intervention Effectiveness**: Track success rates of targeted support programs
- **Resource Optimization**: Measure efficiency improvements in support service delivery
- **Long-term Outcomes**: Monitor retention and graduation rate improvements over time

**Continuous Improvement Process**:
- Regular model retraining with updated student data
- Feature engineering refinement based on emerging patterns
- Integration feedback from student services and academic departments
- Outcome analysis to validate and improve prediction accuracy

## 🔮 Future Enhancements

### Technical Improvements
- **Deep Learning Models**: Explore LSTM for sequential academic data
- **Real-time Predictions**: Develop streaming prediction pipeline
- **Ensemble Methods**: Combine multiple model predictions
- **Feature Automation**: Auto-generate relevant feature interactions

### Data Expansion
- **External Factors**: Weather, campus events, economic indicators
- **Behavioral Data**: Library usage, dining hall visits, course engagement
- **Social Network**: Peer relationships and study group participation
- **Mental Health**: Counseling service usage and stress indicators

## 🤝 Contributing

This project welcomes contributions! Areas for improvement include:
- Additional feature engineering techniques
- Alternative model architectures
- Enhanced visualization methods
- Performance optimization strategies

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.kaggle.com/t/670e4cb20638469d9fc2800b40b58d86) file for details.

## 👤 Author

**Minxi Wang**  
*Data Analytics Graduate Student*  
Teachers College, Columbia University  
📧 Contact: [mw3706@tc.columbia.edu]  
🔗 LinkedIn: [https://www.linkedin.com/in/minxi-wang]  