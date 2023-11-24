# Stock Prediction using Linear Regression

## Overview
Stock Price Prediction using machine learning aims to forecast the future value of company stocks and other financial assets traded on an exchange. Predicting stock prices is a challenging task influenced by various factors including physical, psychological, rational, and irrational behaviors, resulting in dynamic and volatile share prices.

This project utilizes the Linear Regression algorithm to predict stock prices, leveraging Python and key libraries like pandas, numpy, matplotlib, and scikit-learn (sklearn).

## Linear Regression
Linear Regression is a fundamental supervised machine learning algorithm used for modeling the relationship between a dependent variable (target) and one or more independent variables (predictors). Its formula is expressed as: Y = mx + c
Where:
- Y represents the dependent variable (target).
- x represents the independent variable (predictor).
- m is the coefficient of the independent variable.
- c is the y-intercept.
The algorithm aims to find the best-fitting linear relationship between the input variables and the output, enabling predictions based on this relationship.

## Project Workflow
- **Data Splitting**: The dataset is divided into training and testing sets, with 75% of the data allocated to the training set and the remaining 25% to the testing set.
- **Model Training**: The Linear Regression model is trained using the training set. It predicts the coefficients of dependent variables and the y-intercept to establish the relationship between input features and target values.
- **Testing and Predictions**: The trained model is evaluated using the testing set. The testing set data is fed into the trained model to generate predictions based on the established relationship.
- **Accuracy Assessment**: The accuracy of the model is calculated and displayed, showcasing the model's performance in predicting stock prices. Achieving an accuracy of >99% demonstrates the effectiveness of the trained model in predicting stock prices.
![image](https://github.com/JanviBagrecha/Stock-prediction/assets/111588269/05a57edb-717e-4b42-a269-812e04a6ef04)

## Technologies Used
- **Python**: Language used for coding the project.
- **Pandas, NumPy**: Libraries used for data manipulation and analysis.
- **Matplotlib**: Library utilized for data visualization.
- **Scikit-learn (sklearn)**: Framework used for implementing machine learning algorithms.

## Usage
1. Clone the repository.
2. Ensure you have the required dependencies installed (`pip install pandas numpy matplotlib scikit-learn`).
3. Run the Python code for stock prediction using Linear Regression.
4. Explore the accuracy results displayed after model evaluation.

## Support or Contact
For any inquiries or support regarding the Stock Prediction using Linear Regression project, please contact janvi.bagrecha@gmail.com
