# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 00:22:12 2022

@author: shane
"""

import requests
#import json
#from bs4 import BeautifulSoup
import pandas as pd
url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"

r = requests.get(url)

db = r.json()

df = pd.read_json(db)

print("I am cunt")

#leagueSchedule = db.leagueSchedule 
#print(leagueSchedule)

#print(df)