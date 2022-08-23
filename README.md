# Hotel Predict Customer Check-in

# 1. Business/Real World Problem:
A hotel(s) needs to predict whether a cutsomer would check in or not.

# 2. Mapping to Machine Learning Problem:
## 2.1 Machine Learning Problem Statement: Given the customer and booking details, predict which customer would be checking in the hotel.

## 2.2 Type of Machine Learning Problem: There is data provided about the check-ins of a customer. There are 2 possibilities whether the customer will check in or not. This is a Binary Classfication Problem.

## 2.3 Performance Metric:
- Accuracy - If Class Distribution is Balanced
- F1 Score - If Class Distribution is Imbalanced
- Based on domain knowledge we can select a suitable performance metric considering below cases:
    - If we predict that the customer won't be checking in (pred: 0) and the room is given to another person, but the customer actually checks in(true: 1) then that    
      won't be good for the reputation of the hotel. Here, we may consider Recall as our metric.
    - If we predict that the customer will be checking in(pred: 1) and the room is kept reserved, but the customer actually does not check in(true: 0) then that would         affect the hotel's revenue. Here, we may consider Precision as our metric.
    - If both Precision and Recall both are important so we can consider F1-Score.
    - Also, Balanced Accuracy could be a good performance metric here which accounts for Sensitivity and Specificity.

## 2.4 Machine Learing Objectives and Constraints:

### 2.4.1 Objecive: Predict the probability of each data point belonging to Class 1 (i.e. Customer checks-in)

### 2.4.2 Constraints:
- Class Probability is needed
- No Low Latency Requirement
