from flask import Flask, render_template, request, redirect, url_for
import sys
import os
import time
import random
import string
sys.path.append('./')
# import stages.utils.pdaa as pdaa

app = Flask(__name__)

user_reports = []

def make_section_id():
    timestamp = int(time.time() * 1000)
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}-{random_str}"

def compute_adverse_outcomes_result(args):
    return render_template('section/adverse-outcomes-result.html', section_id=make_section_id())

compute_sections = {"adverse-outcomes-result": compute_adverse_outcomes_result}

@app.route("/section/<section_name>")
def section(section_name):
    section_id = make_section_id()
    if section_name in compute_sections:
        return compute_sections[section_name](request.args)
    else:
        return render_template(f'section/{section_name}.html', section_id=section_id)

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    # Example login logic
    return redirect(url_for('home_page'))

@app.route('/home')
def home_page():
    # Example data
    username = "john_doe"
    reports = [
        {"name": "Report 1", "updated": "Updated yesterday"},
        {"name": "Report 2", "updated": "Updated last week"}
    ]
    return render_template('home.html', username=username, reports=reports)

@app.route('/<username>/reports/new')
def new_report(username):
    return render_template('new_report.html', username=username)

@app.route('/create-report', methods=['POST'])
def create_report():
    report_name = request.form['report_name']
    description = request.form.get('description', '')
    visibility = request.form['visibility']
    user_reports.append({
        'name': report_name,
        'description': description,
        'visibility': visibility
    })
    return redirect(url_for('home_page'))

@app.route("/debug/templates")
def debug_templates():
    print(os.listdir("templates/section"))
    return {"templates": os.listdir("templates/section")}


if __name__ == "__main__":
    from werkzeug.serving import run_simple
    from werkzeug.middleware.shared_data import SharedDataMiddleware

    # app = SharedDataMiddleware(app, {"webdashboard/templates": "./webdashboard/templates"})
    run_simple("localhost", 5000, app, use_reloader=True, use_debugger=True)
