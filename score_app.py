# don't write bytecodes!
import sys; sys.dont_write_bytecode = True

# web stuff
from flask import Flask, url_for, request, json, render_template
from bson.objectid import ObjectId
import requests

# back end logic
import model.predict

# data/database
import pymongo
import cPickle
import json


app = Flask(__name__)

# With this application we have managed to break every single rule of modern web
# development with a single function! Never do this! (We were pressed for time
# and it was our first time developing for the web: We're Data Scientists!)
@app.route('/')
def api_show():

    result  = '<body style="font-family:sans-serif;">'
    result += '<table>'
    result += '<th style="background-color:#D3D3D3;">Event Name</th>'
    result += '<th style="background-color:#D3D3D3;">Likelihood of Fraud</th>'
    result += '<th style="background-color:#D3D3D3;">Prediction</th>'

    for event in coll.find().sort("prediction", pymongo.DESCENDING):
        
        p = float(event["prediction"])
        #r = int(event["prediction"] * 255.0)
        if p < SAFE_THRESH:
            pred = 'Safe'
            color = "background-color:#32CD32;"
            font_color = "color:rgb(255,255,255);"
        elif p < FRAUD_THRESH:
            pred = 'Investigate'
            color = "background-color:#FFFF66;"
            font_color = "color:rgb(50,50,50);"
        else:
            pred = 'Fraud!'
            color = "background-color:rgb(255,0,0);"
            font_color = "color:rgb(255,255,255);"
            
        event_uri = '<a href="/find/' + str(event["_id"]) + '">'

        result += '<tr>'
        result += '<td>%s %s</a></td><td align="center" style=\"%s%s\">%.4f' % (event_uri, event["name"], color, font_color, float(event["prediction"])*100) + '%</td>'
        result += '<td align="center" style=\"%s%s\">%s' % (color, font_color, pred) + '</td>'
        result += '</tr>'

    result += '</table>'
    result += '</body>'
    
    return result 

@app.route('/find/<event_id>')
def api_find_event(event_id):
    event = coll.find({ "_id": ObjectId(event_id) })[0]
    return '<p><a href="../..">Dashboard</p><a>' \
            + '<p><a href="../seeall/' + str(event["_id"]) + '">See All</a>' \
            + event["description"]

@app.route('/seeall/<event_id>')
def api_show_all_info(event_id):
    event = coll.find({ "_id": ObjectId(event_id) })[0]
    res  = '<p><a href="/">Dashboard</p><a>'
    res +=  '<p><a href="/find/' + str(event["_id"]) + '">See Less</p><a>'
    res += '<table>'
    for col in event.keys():
        if col == 'description':
            continue

        res += '<tr><td><b>' + str(col) + '</b></td>'
        try:
            obj = ''.join([i if ord(i) < 128 else '' for i in event[col]])
        except TypeError:
            obj = str(event[col])
        res += '<td>' + obj + '</td></tr>'

    res += '</table>'
    res += '<br>'
    res += event['description']

    return res

@app.route('/hello')
def api_hello_world():
    return '<h1>Hello World!</h1>'


def api_register():
    # To use this service: * Send a POST request to /register with your IP 
    # and port (as ip and port parameters in JSON). We'll announce what the 
    # ip address of the service machine is in class. Write this code in the 
    # if name block at the bottom of your Flask script (i.e. it should register
    # each time you run the Flask script)
    requests.post('http://10.0.1.87:5000/register', data = { 'ip': '10.0.1.71', 'port': 5000 })


@app.route('/score', methods = ['GET', 'POST'])
def api_scoring_app():
    js = request.json
    prediction = model.predict.predict_one(js, tfidf_clf, tfidf_lr_clf, final_model, cols)
    js["prediction"] = prediction
    coll.insert(js)
    return str(prediction)


if __name__ == '__main__':
    # Make sure mongo is running:
        # mongod --dbpath db
    conn = pymongo.MongoClient("localhost", 27017)
    coll = conn.scores.data

    # connect to posting service
    api_register()

    # Read in model specifications from pickle file
    tfidf_clf, tfidf_lr_clf, final_model, cols = cPickle.load(open("model/model.pkl", "r"))
    SAFE_THRESH = 0.05
    FRAUD_THRESH = 0.95

    # curl -H 'Content-Type: application/json' -X POST 127.0.0.1:5000/score --data @ex2.json
    app.run(debug=True, host='0.0.0.0')