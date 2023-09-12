import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split

from src.execption import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, model_metrics, print_evaluated_results

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and testing data')
            xtrain,ytrain,xtest,ytest=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],  
                test_array[:,-1],
            )
            models = {
                'Decision Tree': DecisionTreeClassifier(),
                'KNN': KNeighborsClassifier(),
                'Logistic Regression': LogisticRegression()
            }
            
            model_report:dict = evaluate_models(xtrain,ytrain,xtest,ytest,models)
            
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6 :
                logging.info('Best model has Accuracy Score less than 60%')
                raise CustomException('No Best Model Found')
            
            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
 
            ## Final Model
            print('Final Model Evaluation :\n')
            print_evaluated_results(xtrain,ytrain,xtest,ytest,best_model)
        
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            logging.info('pickle file for model saved')
            
            ytest_predict = best_model.predict(xtest)
            
            accuracy, confusionmatrix, precision, recall = model_metrics(ytest, ytest_predict)
            logging.info(f'Test accuracy : {accuracy}')
            logging.info(f'Test confusion matrix : {confusionmatrix}')
            logging.info(f'Test precision : {precision}')
            logging.info(f'Test recall : {recall}')
            logging.info('Final Model Training Completed')
            
            return accuracy
            
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e, sys)
        
