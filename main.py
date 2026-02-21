from flask import Flask, render_template, request, redirect, url_for, flash
import driver
import Rainfall
import alerter

app = Flask(__name__)
app.secret_key = '5791628bb0b13ce0c676dfde280ba245'

# ------------------------
# Basic Pages
# ------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about_team():
    return render_template('about_team.html')

@app.route('/contacts')
def contact():
    return render_template('contact.html')

@app.route('/services')
def service():
    return render_template('service.html')

# ------------------------
# Flood Prediction
# ------------------------

@app.route('/floodHome')
def floodHome():
    """
    Flood homepage with notice board alerts
    """
    alerts = alerter.alerting()  # returns list of rivers
    alerts = [f"Flood ALERT for {river}" for river in alerts]

    return render_template(
        'floodHome.html',
        result=alerts
    )

@app.route('/refreshFlood')
def refreshFlood():
    """
    Refresh flood alert data (12 month forecast)
    """
    alerter.water_level_predictior()
    return redirect(url_for('floodHome'))

@app.route('/floodResult', methods=['POST'])
def floodResult():
    """
    Flood prediction result page
    """
    user_date = request.form.get('DATE')
    river = request.form.get('SEL')

    if not user_date or not river:
        return redirect(url_for('floodHome'))

    # Core ML logic delegated to driver.py
    results_dict = driver.drive(river, user_date)

    # Convert dict to ordered list for table rendering
    table_data = list(results_dict.values())

    return render_template(
        'floodResult.html',
        result=table_data
    )

# ------------------------
# Rainfall Prediction
# ------------------------

@app.route('/rainfallHome')
def rainfallHome():
    return render_template('rainfallHome.html')

@app.route('/rainfallResult', methods=['POST'])
def rainfallResult():
    """
    Rainfall prediction result page
    """
    year = request.form.get('Year')
    region = request.form.get('SEL')

    if not year or not region:
        flash("Please enter all required data")
        return redirect(url_for('rainfallHome'))

    # ML logic delegated to Rainfall.py
    mae, score = Rainfall.rainfall(year, region)

    return render_template(
        'rainfallResult.html',
        Mae=mae,
        Score=score
    )

# ------------------------
# App Runner
# ------------------------

if __name__ == '__main__':
    app.run(debug=True)
