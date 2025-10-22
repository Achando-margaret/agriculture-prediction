AGRICULTURE_PREDICTION
Using Machine Learning to Forecast Crop Yields and Ensure Food Security in Kenya.

This project is an AI-driven solution designed to address the UN Sustainable Development Goal 2: Zero Hunger. By leveraging historical agricultural and climate data, Mavuno AI predicts crop yields for staple crops in Kenya, providing valuable insights to farmers and policymakers.

🌍 The Problem: Food Insecurity in Kenya
Kenyan farmers, particularly smallholders, face significant uncertainty due to unpredictable weather patterns and limited access to data-driven farming techniques. This unpredictability can lead to inefficient resource allocation, reduced harvests, and increased food insecurity. Our project aims to bridge this information gap with a reliable forecasting tool.

🤖 Our Solution: An AI-Powered Forecasting Model
We developed a supervised machine learning model that uses a RandomForestRegressor algorithm to predict crop yield in kilograms per hectare.

The model analyzes key factors to make its predictions:

Climate Data: Average rainfall and temperature.
Geographical Data: Region within Kenya.
Agronomic Data: Soil type and crop type.
Temporal Data: Year of harvest.
🛠️ Tech Stack
Language: Python
Libraries:
Pandas for data manipulation
NumPy for numerical operations
Scikit-learn for building and evaluating the ML model
Matplotlib & Seaborn for data visualization
Environment: Jupyter Notebook / Google Colab
📈 Results & Performance
Our model has demonstrated strong predictive power, achieving the following results on our test dataset:

R-squared (
R
2
) Score: 0.84
This means our model can explain 84% of the variability in crop yields, indicating a high level of accuracy.
Mean Absolute Error (MAE): 191.05 kg/hectare
On average, our model's predictions are off by only 191 kg/hectare, which is a strong result for agricultural forecasting.
Visualizations
1. Actual vs. Predicted Yield
This scatter plot shows a strong correlation between our model's predictions and the actual harvest data.

![Actual vs. Predicted Plot](Screenshot1.jpg)

2. Key Predictive Factors
Rainfall and region were identified as the most significant factors influencing crop yield, highlighting the importance of location-specific climate data.

![Feature Importance Plot](Screenshot2.jpg)

⚙️ How to Run This Project
Clone the repository:
git clone https://github.com/Achando-margaret/agriculture-prediction.git
Navigate to the project directory:
cd agriculture_prediction
Install the required libraries:
pip install pandas numpy scikit-learn matplotlib seaborn
Run the Jupyter Notebook: Open and run the .ipynb file to see the entire workflow from data preprocessing to model evaluation.
🤔 Ethical Considerations
Data Bias: The model's accuracy is dependent on the quality and diversity of the training data. A lack of data from smaller, more remote farms could lead to less accurate predictions for those regions.
Accessibility: To be truly effective, insights from this model must be accessible to all farmers, including those without internet access. Future work should explore delivery via SMS or local community partnerships.
