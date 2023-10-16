# stroke_prediction
BYU Math 402 Group Project
Stroke is a significant health concern, with alarming statistics in the United States: a stroke occurs every 40 seconds, and someone dies from a stroke every 4 minutes. Identifying individuals at risk of stroke is vital for timely preventive measures. This paper delves into the world of predictive modeling for stroke risk assessment, aiming to determine which models and corresponding parameters are most effective.

A variety of models have been used for stroke prediction, such as linear regression and Cox regression, achieving varying degrees of success. However, the study reveals a notable gap: a lack of comparative research among different modeling techniques. This project fills this void by applying multiple modeling techniques to the same dataset to ascertain which model, or combination of models, yields the best performance.

Given the severe consequences of false negatives in stroke prediction, accuracy is crucial. However, we also incorporate recall as a performance measure in our evaluation.

In addition to model evaluation, we investigate which features are most correlated with stroke risk. This provides guidance for patients to prioritize risk factors for intervention. Our dataset includes ten features, including gender, age, hypertension, heart disease, and lifestyle factors, collected from over 5,000 individuals.

The study starts by cleaning the dataset, addressing issues like insufficient observations and handling missing values. We describe a generalized data cleaning function to facilitate future data cleaning efforts. Our approach to data cleaning is documented thoroughly for transparency.

The paper presents visualizations that offer insights into the relationships between stroke risk and various continuous and binary features. These visualizations help identify the most important risk factors.

We explore the performance of several classification algorithms, including Logistic Regression, K-Nearest Neighbor, Naive Bayes, and Random Forests, assessing them through accuracy, recall, and F1 scores. Given the imbalanced dataset, we employ the Synthetic Minority Over-sampling Technique (SMOTE) to enhance the recall rate.

Our results indicate that the Gaussian Naive Bayes model stands out due to its high recall, which is crucial in identifying individuals at risk. However, it is essential to consider the potential ethical and practical implications when applying these models to medical diagnosis. The study also highlights data collection and transparency challenges and calls for better practices in this regard.

In conclusion, the study demonstrates the effectiveness of various models in predicting stroke risk. It underscores the importance of feature selection and the need to account for the practical implications and ethics in implementing predictive models for medical diagnoses. The findings provide a valuable foundation for further research and model development in the field of stroke prediction.
