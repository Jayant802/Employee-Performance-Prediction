from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        quarter = int(request.form['quarter'])
        department_1 = int(request.form['department'])      # First department
        day = int(request.form['day'])
        team = int(request.form['team'])
        targeted_productivity = float(request.form['targeted_productivity'])
        smv = float(request.form['smv'])
        wip = float(request.form['wip'])
        over_time = int(request.form['over_time'])
        incentive = int(request.form['incentive'])
        idle_time = float(request.form['idle_time'])
        idle_men = int(request.form['idle_men'])
        no_of_style_change = int(request.form['no_of_style_change'])
        no_of_workers = float(request.form['no_of_workers'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        weekday = int(request.form['weekday'])
        department_2 = int(request.form['department_2'])   # Second department encoded field

        # 17 total features
        input_features = np.array([[quarter, department_1, day, team, targeted_productivity, smv,
                                   wip, over_time, incentive, idle_time, idle_men, no_of_style_change,
                                   no_of_workers, year, month, weekday, department_2]])

        prediction = model.predict(input_features)[0]
        prediction = round(prediction, 3)

        return render_template('submit.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
