# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 00:22:12 2022

@author: shane
"""

#import requests
#import json
#from bs4 import BeautifulSoup

#url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

#r = requests.get(url)

#db = r.json()
#db = json.loads(r.text)
#df = pd.read_json(r)
#print(db)
#..............................


#import pandas as pd
#URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
#df = pd.read_json(URL)

#leaguesSCh = df.loc[:,"leagueSchedule"]

#df.info()

import requests
import json
#from bs4 import BeautifulSoup
import pandas as pd
#url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

r = requests.get(url)

db = r.json()
#leagueSch = db
#db = json.loads(r.text)
#df = pd.read_json(r)
df = pd.DataFrame.from_dict(db)
for i in range(4):
    HomeTeamName =df["leagueSchedule"]['gameDates'][i]['games'][0]['homeTeam']["teamName"]
    AwayTeamName =df["leagueSchedule"]['gameDates'][i]['games'][0]['awayTeam']["teamName"]
    print(HomeTeamName)
    print(AwayTeamName)
