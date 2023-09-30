import numpy as np
import psycopg2
from flask import Flask, render_template, render_template_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf
import json
#################################################
# Database Setup
#################################################

#Hardcoded credentials for local use only
conn = psycopg2.connect(host='localhost', database="nba_stats", user='postgres', password='postgres')

columns = ['Id','Team', 'PTS', 'FG', 'FGA', '2PA', '3P', '3PA', 'eFG%', 'Home/Away',
       'ORtg', 'FTr', '3PAr', 'TS%', 'FT/FGA', 'ORtg_opponent', 'FTr_opponent',
       '3PAr_opponent', 'TS%_opponent', 'eFG%_opponent', 'FT/FGA_opponent']
#################################################
# Flask Setup
#################################################
app = Flask(__name__)

nn_model = load_model("NBA_Model.h5")

#################################################
# Flask Routes
#################################################
def getTeam(team: str):
    cur = conn.cursor()

    #Using Limit to test query without bloating console
    cur.execute('SELECT * FROM nba WHERE nba.Team = %s LIMIT 1;', (team, ))
    team_data = cur.fetchall()
    cur.close()
    return team_data

def changeTypes(df):
    return df.astype({
        'Team': "string",
        'Home/Away': "string"
    })

def transformAndPredict(df, i, k):
    y = df["PTS"]
    X = df.drop(["Id", "Team", "PTS", "Home/Away"], axis=1)
    X["Home/Away_Home"] = i
    X["Home/Away_Away"] = k
    print(X.head())
    # Create a StandardScaler instances
    scaler = StandardScaler()


    # Fit the StandardScaler
    X_scaler = scaler.fit(X)

    # Scale the data
    X_train_scaled = X_scaler.transform(X)


    prediction = nn_model.predict(X_train_scaled)

    return int(round(prediction[0][0]))





@app.route("/")
def home():
    return (
        render_template('future.html')
    )

@app.route("/data/<team1>/<team2>")
def data(team1 = None, team2 = None):
    if team1 == None or team2 == None:
        return render_template_string('Team Not Found {{ errorCode }}', errorCode='404'), 404
    
    #Get Team Data from database
    team1_data = getTeam(team1)
    team2_data = getTeam(team2)
    
    if team1_data == [] or team2_data == []:
        return render_template_string('Team Not Found {{ errorCode }}', errorCode='404'), 404

    #Transform to DF
    team1_data_df = pd.DataFrame(team1_data, columns=columns)
    team2_data_df = pd.DataFrame(team2_data, columns=columns)

    #Change from Object to String
    team1_data_df = changeTypes(team1_data_df)
    team2_data_df = changeTypes(team2_data_df)
    
    #Hardcoded home vs away
    i = 0
    k = 1
    
    team1_score = transformAndPredict(team1_data_df, i, k)
    i+=1
    k-=1
    team2_score = transformAndPredict(team2_data_df, i, k)
    # conn.close()
    return json.loads(json.dumps({
        "team_1": team1_score,
        "team_2": team2_score
    }))

if __name__ == '__main__':
    app.run(debug=True)