# Instructor Performance and Course Quality Evaluation on EduPro

## Author details
Author: Mirthesh. M
E-mail: ``` mirtheshmurugaiah1105@gmail.com ```
Employer: Unified Mentor

---

## Overview

This project analyzes instructor performance on an online education platform by combining course data, instructor attributes, and student feedback. The goal is to understand how much instructor quality influences course ratings and whether experience or course category plays a role in perceived teaching effectiveness.

Instead of treating ratings as isolated scores, the project models expected course ratings using instructor and course features, then analyzes where instructors consistently over- or under-perform.

---

### Questions Addressed

#### Which instructors consistently deliver high-quality courses?
Identified by high average ratings with low prediction error across courses.

#### Does teaching experience translate into better-rated courses?
Measured using correlation and experience-bucket comparisons.

#### Are some course categories more dependent on instructor quality?
Inferred through prediction error sensitivity to instructor features.

#### How evenly is teaching performance distributed across the platform?
Evaluated using summary statistics and a Gini coefficient over ratings.

---

### Dataset

The data is stored in a single Excel file with three sheets:

1. Teachers
Instructor demographics and experience

2. Courses
Course metadata such as category, level, duration, and price

3. Transactions
Enrollment and course rating information

All sheets are merged using TeacherID and CourseID.

---

### Approach

1. Merge instructor, course, and transaction data
2. Clean missing values and select relevant features
3. Encode categorical variables and scale numeric features
4. Train a neural network to predict course ratings
5. Compare predicted vs actual ratings at the instructor level
6. Perform statistical analysis on instructor performance

The model is not used for deployment or recommendation, but as an analytical tool to study instructor impact.

---

### Model Details
- Type: Feed-forward neural network
- Loss: Mean Squared Error
- Optimizer: Adam
- Metrics: MAE, RMSE, R²

Regularization is handled using dropout layers to reduce overfitting.

---

### Repository Structure
```
├── data/
│   └── EduPro_Online_Platform.xlsx
├── models/
│   └── instructor_performance_model.h5
├── src/
│   └── train_and_analyze.py
└── README.md
```

---

### How to Run

1. Install dependencies
```
pip install requirements.txt
```

2. Run the training and analysis script
```
python src/train_and_analyze.py
```

---

### The script will:

- Train the model
- Print evaluation metrics
- Output instructor-level performance analysis
- Save the trained model to the models/ directory

---

### Output and Interpretation

#### The script prints:
- Model performance metrics (RMSE, R²)
- A list of consistently high-performing instructors
- Correlation between experience and ratings
- Average ratings by experience group
- Indicators of instructor-dependent course performance
- Distribution statistics of instructor ratings
These outputs are intended for analysis and insight, not automated decision-making.

---

### Limitations
- Ratings are subjective and may be influenced by factors outside instructor control
- The dataset does not capture student background or expectations
- Model predictions are used for comparison, not as ground truth

---

### Future Improvements
- Add temporal analysis of instructor performance over time
- Include student engagement metrics
- Extend analysis to recommendation or quality assurance systems
- Replace Excel input with a database pipeline

---

### Author Notes
- This project was built as an applied machine learning analysis exercise, focusing on clean data handling, reasonable modeling choices, and interpretable results rather than aggressive optimization.