from flask import Flask, render_template, request, session
import os
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import pickle
# Ensuring the static/plots directory exists
os.makedirs('static/plots', exist_ok=True)

app = Flask(__name__)  

# Loading the trained models
models = {
    'Random Forest': pickle.load(open('RFmodel.pkl', 'rb')),
    'SVM': pickle.load(open('SVMmodel.pkl', 'rb')),
    'Decision Tree': pickle.load(open('DTmodel.pkl', 'rb')),
    'Logistic Regression': pickle.load(open('LOGmodel.pkl', 'rb')),
}

# Prediction function
def predict_single_row(model, row):
    row_df = pd.DataFrame([row])  
    prediction = model.predict(row_df)
    return int(prediction[0])


@app.route('/', methods=['GET', 'POST'])
def prediction():
    combo_box_options = {
        'sex': [0, 1],
        'cp': [0, 1, 2, 3],
        'fbs': [0, 1],
        'restecg': [0, 1, 2],
        'exang': [0, 1],
        'slope': [0, 1, 2],
        'ca': [0, 1, 2, 3],
        'thal': [0, 1, 2, 3]
    }
    model_options = list(models.keys())
    result = None
    if request.method == 'POST':
        selected_model_name = request.form.get('model')
        selected_model = models[selected_model_name]
        user_input = {
            'age': int(request.form.get('age', 0)),
            'sex': int(request.form.get('sex', 0)),
            'cp': int(request.form.get('cp', 0)),
            'trestbps': int(request.form.get('trestbps', 0)),
            'chol': int(request.form.get('chol', 0)),
            'fbs': int(request.form.get('fbs', 0)),
            'restecg': int(request.form.get('restecg', 0)),
            'thalach': int(request.form.get('thalach', 0)),
            'exang': int(request.form.get('exang', 0)),
            'oldpeak': float(request.form.get('oldpeak', 0.0)),
            'slope': int(request.form.get('slope', 0)),
            'ca': int(request.form.get('ca', 0)),
            'thal': int(request.form.get('thal', 0))
        }
        result = predict_single_row(selected_model, user_input)
        result = f"The patient {'has' if result == 1 else 'does not have'} a heart attack."
    return render_template('prediction.html', combo_box_options=combo_box_options, model_options=model_options, result=result)

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    box_plot_files = {
        'age': "static/plots/box_plot_age.html",
        'chol': "static/plots/box_plot_chol.html",
        'oldpeak': "static/plots/box_plot_oldpeak.html",
        'thalach': "static/plots/box_plot_thalach.html",
        'trestbps': "static/plots/box_plot_trestbps.html"
    }
    histogram_files = {
        'age': "static/plots/histogram_age.html",
        'chol': "static/plots/histogram_chol.html",
        'oldpeak': "static/plots/histogram_oldpeak.html",
        'thalach': "static/plots/histogram_thalach.html",
        'trestbps': "static/plots/histogram_trestbps.html"
    }
    categorical_count_files = {
        'sex': "static/plots/categorical_count_plot_sex.html",
        'exang': "static/plots/categorical_count_plot_exang.html",
        'ca': "static/plots/categorical_count_plot_ca.html",
        'cp': "static/plots/categorical_count_plot_cp.html",
        'fbs': "static/plots/categorical_count_plot_fbs.html",
        'restecg': "static/plots/categorical_count_plot_restecg.html",
        'slope': "static/plots/categorical_count_plot_slope.html",
        'thal': "static/plots/categorical_count_plot_thal.html"
    }
    correlation_file = "static/plots/correlation_heatmap.html"
    bar_plot_file="static/plots/barplot.html"
    session['selected_attribute'] = request.form.get('attribute', session.get('selected_attribute'))
    session['selected_histogram_attribute'] = request.form.get('histogram_attribute', session.get('selected_histogram_attribute'))
    session['selected_cattribute'] = request.form.get('categorical_attribute', session.get('selected_cattribute'))
    return render_template('analysis.html', 
                           selected_attribute=session['selected_attribute'],
                           selected_histogram_attribute=session['selected_histogram_attribute'],
                           selected_cattribute=session['selected_cattribute'],
                           box_plot_files=box_plot_files,
                           histogram_files=histogram_files,
                           categorical_count_files=categorical_count_files, 
                           correlation_file=correlation_file,
                           bar_plot_file=bar_plot_file)
if __name__ == '__main__':
    app.run(debug=True)
 
