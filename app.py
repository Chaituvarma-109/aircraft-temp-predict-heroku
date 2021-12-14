from flask import Flask, request, render_template
from flask_cors import cross_origin
from Air_temp.model import Model

app = Flask(__name__)


mod = Model('ai4i2020.csv')


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def take_inputs():
    if request.method == 'POST':
        try:
            process_temperature = float(request.form['Process temperature [K]'])
            rotational_speed = int(request.form["Rotational speed [rpm]"])
            torque = float(request.form["Torque [Nm]"])
            tool_wear = int(request.form["Tool wear [min]"])
            twf = int(request.form["TWF"])
            hdf = int(request.form["HDF"])
            pwf = int(request.form["PWF"])
            osf = int(request.form["OSF"])
            rnf = int(request.form["RNF"])
            pred = mod.prediction([[process_temperature, rotational_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf]])
            return render_template('result.html', prediction=pred[0][0])
        except Exception as e:
            print(f'Exception has occurred: {e}')
            return "Something is wrong."
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
