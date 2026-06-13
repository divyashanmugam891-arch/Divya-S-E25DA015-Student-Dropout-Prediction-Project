from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__, template_folder='templats')  
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
gender_map = {"Male": 1, "Female": 0}
yes_no_map = {"Yes": 1, "No": 0}
@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        try:
            gender = gender_map[request.form['gender']]
            family_income = float(request.form['family_income'])
            study_hours = float(request.form['study_hours'])
            attendance = float(request.form['attendance'])
            part_time = yes_no_map[request.form['part_time_job']]
            scholarship = yes_no_map[request.form['scholarship']]
            stress_index = float(request.form['stress_index'])
            gpa = float(request.form['gpa'])
            features = np.array([[
                gender,
                family_income,
                study_hours,
                attendance,
                part_time,
                scholarship,
                stress_index,
                gpa
            ]])
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)
            probability = model.predict_proba(features_scaled)
            dropout_prob = probability[0][1]
            if dropout_prob >= 0.5:
                result = f"Likely to Dropout ({dropout_prob*100:.2f}%)"
            elif dropout_prob >= 0.3:
                result = f"Medium DROPOUT Risk ({dropout_prob*100:.2f}%)"
            else:
                result = f"Not Likely to Dropout ({(1-dropout_prob)*100:.2f}%)"
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('index.html', result=result, form=request.form)
if __name__ == '__main__':
    app.run(debug=True)