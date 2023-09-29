import numpy as np
import psycopg2
from flask import Flask, render_template, render_template_string
import pandas as pd
import json
#################################################
# Database Setup
#################################################

#Hardcoded credentials for local use only
conn = psycopg2.connect(host='localhost', database="nba_stats", user='postgres', password='postgres')


#################################################
# Flask Setup
#################################################
app = Flask(__name__)


#################################################
# Flask Routes
#################################################
def getTeam(team: str):
    cur = conn.cursor()

    #Using Limit to test query without bloating console
    cur.execute('SELECT * FROM nba WHERE nba.Team = %s LIMIT 2;', (team, ))
    team_data = cur.fetchall()
    cur.close()
    return team_data

@app.route("/")
def home():
    return (
        render_template('future.html')
    )

@app.route("/data/<team1>/<team2>")
def data(team1 = None, team2 = None):
    if team1 == None or team2 == None:
        return render_template_string('Team Not Found {{ errorCode }}', errorCode='404'), 404
    team1_data = getTeam(team1)
    team2_data = getTeam(team2)
    
    if team1_data == [] or team2_data == []:
        return render_template_string('Team Not Found {{ errorCode }}', errorCode='404'), 404

    print(pd.DataFrame(team1_data))
    print(pd.DataFrame(team2_data))
    # conn.close()
    return json.loads(json.dumps({
        "Found": True
    }))

if __name__ == '__main__':
    app.run(debug=True)