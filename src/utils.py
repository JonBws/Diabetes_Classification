import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from src.execption import CustomException
from src.logger import logging
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
           
            ## Train the model
            model.fit(x_train,y_train)
            
            ## predict train and test data
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            ## Get accuracy score for train and test
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e, sys)
    
def model_metrics (true, predicted):
    try:
        accuracy = accuracy_score(true, predicted)
        confusionmatrix = confusion_matrix(true, predicted)
        precision = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        return accuracy, confusionmatrix, precision, recall
    except Exception as e:
        logging.info('Exception Occured while evaluating metric')
        raise CustomException(e,sys)
        
def print_evaluated_results(xtrain,ytrain,xtest,ytest,model):
    try:
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)

        # Evaluate Train and Test dataset
        model_train_accuracy , model_train_confusionmatrix, model_train_precision, model_train_recall = model_metrics(ytrain, ytrain_pred)
        model_test_accuracy , model_test_confusionmatrix, model_test_precision, model_test_recall = model_metrics(ytest, ytest_pred)

        # Printing results
        print('Model performance for Training set')
        print("- Accuracy: {:.4f}".format(model_train_accuracy))
        print("- Confusion Matrix:\n {}".format(model_train_confusionmatrix))
        print("- Precision: {:.4f}".format(model_train_precision))
        print("- Recall: {:.4f}".format(model_train_recall))
        
        print('----------------------------------')
        
        print('Model performance for Test set')
        print("- Accuracy: {:.4f}".format(model_test_accuracy))
        print("- Confusion Matrix:\n {}".format(model_test_confusionmatrix))
        print("- Precision: {:.4f}".format(model_test_precision))
        print("- Recall: {:.4f}".format(model_test_recall))
        
        
    except Exception as e:
        logging.info('Exception occured during printing of evaluated results')
        raise CustomException(e,sys)
        
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)