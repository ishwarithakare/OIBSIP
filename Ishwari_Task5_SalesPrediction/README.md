Sales Prediction Using Advertising Data

This project builds a regression model to predict product sales based on advertising budget allocated across TV, Radio, and Newspaper channels. Built as part of the Oasis Infobyte Data Science Internship.

Dataset
The classic Advertising dataset (200 records), containing advertising budgets across three channels and the resulting sales figures.

Approach
Correlation analysis showed TV had the strongest individual relationship with sales, followed by Radio, while Newspaper showed a weak relationship. A TV and Radio interaction feature was engineered by multiplying the two budgets together, which showed a much stronger correlation with sales than either channel alone, revealing a synergy effect between the two channels.

Five regression models were compared using 5-fold cross-validation: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Gradient Boosting. Gradient Boosting performed best and was further tuned using GridSearchCV. Residual analysis was conducted to confirm the model's errors were random rather than systematic.

Results
The final tuned Gradient Boosting model achieved a R² score of 0.9914 on the test set, with an average prediction error of just $0.37 in sales. Feature importance analysis confirmed that the TV and Radio interaction was by far the most influential predictor, more important than TV, Radio, or Newspaper individually. This suggests that running TV and Radio advertising together drives sales more effectively than either channel alone.

Interactive app
An interactive Streamlit app is included in the app folder, where advertising budgets can be adjusted using sliders to see predicted sales update live, along with an explanation of the TV-Radio synergy effect.

Project structure
notebooks folder contains the full analysis and model building notebook.
models folder contains the saved trained model and generated charts.
app folder contains the Streamlit web app.

To run the app
Navigate to the app folder and run: streamlit run sales_app.py

Tools used
Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit

Author
Ishwari Thakare