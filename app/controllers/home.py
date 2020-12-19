# -*- coding: utf-8 -*-
from flask import Blueprint, render_template

blueprint = Blueprint('home', __name__)

@blueprint.route('/')
def index():
    return render_template('home/index.html')

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl

import seaborn as sns
import tensorflow as tf
import io as io
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import re
import json

@blueprint.route('/predictTomorrow', methods=['GET'])
def predictCasesTomorrow():
    # Register converters to avoid warnings
    pd.plotting.register_matplotlib_converters()
    plt.rc("figure", figsize=(16,8))
    plt.rc("font", size=14)

    url="https://api.covidtracking.com/v1/states/daily.csv"
    s=requests.get(url).content
    c=pd.read_csv(io.StringIO(s.decode('utf-8')))
    predictorList = []
    for state in c['state'].unique():
        df = c.loc[c['state'] == state]
        #df['DateTime'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
        df = df[::-1]
        mod = sm.tsa.statespace.SARIMAX(df['positive'], trend='c', order=(1,1,1),simple_differencing=True)
        res = mod.fit(disp=False)
        result = res.forecast()
        result = re.search(r"[\u0030-\u0039]+\u002E[\u0030-\u0039]+", str(result))
        result = result.group()
        predictorList.append([state,str(result)])
        # print(res.summary())
        print("State: "+str(state)+" , predicted next day positive cases : "+str(result))#.format(state, res.forecast()))
    #print("State: "+ state + "Predicted next day positive cases"+res.forecast())
    js = json.dumps(predictorList)
    return js

#able to show weighted average for population density for every state
# @blueprint.route('/vaccineUrgency', methods=['GET'])
# def vaccineUrgency():
#     df = pd.read_csv('sample_data/uscities.csv')
#     df_state = df.groupby('state_id')

#     data = []
#     for state in df_state['state_id'].unique():
#     cur_state = df_state.get_group(state[0]) 
#     cur_state_copy = cur_state.copy()
#     cur_state['weighted_average'] = (cur_state_copy['population'] / cur_state_copy.population.sum()) * cur_state_copy['density']
#     data.append([str(state[0]),cur_state.weighted_average.sum()])

#     state_df = pd.DataFrame(data, columns=['state', 'weight'])    
    