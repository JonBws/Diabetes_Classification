## Diabetes Prediction

![GIF](resource/web-app.gif)

## Step By Step on This Project
1. Data Ingestion
   * Read the dataset after clean for missing value
   * split cleaned dataset into training and testing
     
2. Data Transformation
   * make pipeline for numerical column and apply standard scaler for this column
   * save this pipeline as preprocessor.pkl
     
3. Model Trainer
   * train several model for classification on training dataset
   * the best model is random forest
   * this best model save as model.pkl
     
4. Predict Pipeline
   * convert input data as dataframe
   * open preprocessor.pkl and model.pkl
   * use this new data for prediction using model.pkl and preprocessor.pkl
     
5. Make Flask App
   * make UI for predict wheter person have diabetes or don't have diabetes
