from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)

## Route for a home page
@app.route('/')
def home_page():
    return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict_datapoint():
        data=CustomData(
            Pregnancies = int(request.form.get('Pregnancies')),
            Glucose = int(request.form.get('Glucose')),
            BloodPressure = int(request.form.get('BloodPressure')),
            SkinThickness = int(request.form.get('SkinThickness')),
            Insulin = int(request.form.get('Insulin')),
            BMI = float(request.form.get('BMI')),
            DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction')),
            Age = int(request.form.get('Age'))
        )
        
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(pred_df)
        
        return render_template('result.html',results=pred)
    

if __name__ == "__main__":
    app.run(debug=True)        