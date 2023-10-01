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

nn_model = load_model("NBA_Model_V2.h5")

#################################################
# Flask Routes
#################################################
def getTeam(team1: str, team2: str):
    cur = conn.cursor()

    #Using Limit to test query without bloating console
    cur.execute('SELECT * FROM nba WHERE nba.Team = %s OR nba.Team = %s', (team1, team2, ))
    team_data = cur.fetchall()
    cur.close()
    return team_data

def changeTypes(df):
    return df.astype({
        'Team': "string",
        'Home/Away': "string"
    })

def transformAndPredict(df):
    y = df["PTS"]
    X = df.drop(["Id", "Team", "PTS"], axis=1)
    X = pd.get_dummies(X, columns=["Home/Away"])
    print(X.head())
    # Create a StandardScaler instances
    scaler = StandardScaler()


    # Fit the StandardScaler
    X_scaler = scaler.fit(X)

    # Scale the data
    X_train_scaled = X_scaler.transform(X)


    prediction = nn_model.predict(X_train_scaled)
    flat_list = [round(item) for sublist in prediction for item in sublist]
    return flat_list





@app.route("/")
def home():
    return (
        render_template('home.html')
    )

@app.route("/data/<team1>/<team2>")
def data(team1 = None, team2 = None):
    if team1 == None or team2 == None:
        return render_template_string('Team Not Found {{ errorCode }}', errorCode='404'), 404
    
    #Get Team Data from database
    teams_data = getTeam(team1, team2)

    if teams_data == []:
        return render_template_string('Team Not Found {{ errorCode }}', errorCode='404'), 404

    #Transform to DF
    teams_data_df = pd.DataFrame(teams_data, columns=columns)

    #Change from Object to String
    teams_data_df = changeTypes(teams_data_df)
    print(teams_data_df)
    #Set Home/Away
    teams_data_df.loc[teams_data_df["Team"] == team1, 'Home/Away'] = 'Home'
    teams_data_df.loc[teams_data_df["Team"] == team2, 'Home/Away'] = 'Away'

    #Check to see if Home team is first in the row, if not swap order
    #This helps keep track of respective prediction
    b, c = teams_data_df.iloc[0], teams_data_df.iloc[1]
    if b["Home/Away"] != "Home":
        temp = teams_data_df.iloc[0].copy()
        teams_data_df.iloc[0] = c
        teams_data_df.iloc[1] = temp
    team_scores = transformAndPredict(teams_data_df)
    # conn.close()
    return json.loads(json.dumps({
        "team_1": team_scores[0],
        "team_2": team_scores[1]
    }))

if __name__ == '__main__':
    app.run(debug=True)