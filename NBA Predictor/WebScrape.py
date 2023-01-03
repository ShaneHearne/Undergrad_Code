# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:29:44 2022

@author: shane
"""


import requests
import pandas as pd
url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
r = requests.get(url)
db = r.json()
df = pd.DataFrame.from_dict(db)
homeTeams = []
awayTeams = []
# for i in range(4):
#     HomeTeamName =df["leagueSchedule"]['gameDates'][i]['games'][0]['homeTeam']["teamName"]
#     AwayTeamName =df["leagueSchedule"]['gameDates'][i]['games'][0]['awayTeam']["teamName"]
#     homeTeams.append(HomeTeamName)
#     awayTeams.append(AwayTeamName)
  #  print(AwayTeamName)
  #  print(HomeTeamName)
    
for i in range(8):
    # print("index_i",i)
    var1 =df["leagueSchedule"]['gameDates'][i]['games']
    # print(len(var1))
    for j in range(len(var1)):
    #     print("index_j",j)   
        HomeTeamName = var1[j]['homeTeam']["teamName"]
        AwayTeamName = var1[j]['awayTeam']["teamName"]
        homeTeams.append(HomeTeamName)
        awayTeams.append(AwayTeamName)

Schedule = pd.DataFrame(
    {'Home Teams' : homeTeams,
     'Away Teams' : awayTeams
     }
    )
print(Schedule)