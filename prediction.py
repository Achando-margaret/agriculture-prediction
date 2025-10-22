
# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the full model pipeline
# 1. Preprocess data
# 2. Train a RandomForestRegressor model
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Train the model!
model_pipeline.fit(X_train, y_train)
print("\nModel training complete!")

# STEP 5: MODEL EVALUATION


# Make predictions on the test data
y_pred = model_pipeline.predict(X_test)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")
print("--------------------------")
print("Note for Person B: MAE means our model's predictions are, on average, off by this many kg/hectare.")
print("Note for Person B: R² of ~0.9+ is excellent, it means our model explains over 90% of the variance in crop yield.")



# STEP 6: GENERATE VISUALIZATIONS (FOR PERSON B)
# 1. Actual vs. Predicted Yield Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Perfect prediction line
plt.title('Actual vs. Predicted Crop Yield', fontsize=16)
plt.xlabel('Actual Yield (kg/hectare)', fontsize=12)
plt.ylabel('Predicted Yield (kg/hectare)', fontsize=12)
plt.grid(True)
plt.savefig('actual_vs_predicted.png') 
plt.show()


# 2. Feature Importance Plot
# Get feature names after one-hot encoding
feature_names = numerical_features + \
                list(model_pipeline.named_steps['preprocessor']
                     .named_transformers_['cat']
                     .get_feature_names_out(categorical_features))

importances = model_pipeline.named_steps['regressor'].feature_importances_
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False).head(10) # Top 10

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Top 10 Most Important Features for Predicting Crop Yield', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.savefig('feature_importance.png') 
plt.show()# Make predictions on the test data
