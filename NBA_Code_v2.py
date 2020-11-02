#!/usr/bin/env python
# coding: utf-8

# In[353]:


get_ipython().run_line_magic('matplotlib', 'inline')
from selenium import webdriver
from pandas import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import *
from bs4 import BeautifulSoup
import requests


# In[354]:


urls = []
months = ['october','november','december','january','february','march','april']

for i in months:
    a = "https://www.basketball-reference.com/leagues/NBA_2020_games-" + str(i) + ".html" #change se
    urls.append(a)
    continue


# In[355]:


Totals = pd.DataFrame([["Date", "Visitor/Neutral","PTS","Home/Neutral","PTS.1"]])
new_header = Totals.iloc[0] 
Totals = Totals[1:] 
Totals.columns = new_header


# In[356]:


for u in urls:
    url = str(u)
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "html.parser")
    results = soup.find("table",id='schedule')
    df = pd.read_html(str(results))
    df = df[0]
    Sced = df[["Date", "Visitor/Neutral","PTS","Home/Neutral","PTS.1"]]
    Totals = Totals.append(Sced,ignore_index=True)
    continue


# In[357]:


Totals = Totals.dropna()
Totals = Totals.astype(str)


# In[358]:


Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Mon'))
Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Tue'))
Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Wed'))
Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Thu'))
Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Fri'))
Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Sat'))
Totals['Date'] = Totals['Date'].map(lambda x: x.lstrip('Sun'))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace(',',""))
Totals['Date'] = Totals['Date'].map(lambda x: x.strip(' '))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace(' ',"/"))


# In[359]:


Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Nov',"11"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Oct',"10"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Jan',"01"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Feb',"02"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Mar',"03"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Apr',"04"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('May',"05"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Jun',"06"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('July',"07"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Aug',"08"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Sep',"09"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('Dec',"12"))


# In[360]:


Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/1/',"/01/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/2/',"/02/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/3/',"/03/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/4/',"/04/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/5/',"/05/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/6/',"/06/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/7/',"/07/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/8/',"/08/"))
Totals['Date'] = Totals['Date'].map(lambda x: x.replace('/9/',"/09/"))


# In[361]:


#Totals['Date'] = pd.to_datetime(Totals['Date'], format='%m/%d/%Y')  


# In[362]:


Totals["PTS"] = Totals["PTS"].apply(pd.to_numeric)
Totals["PTS.1"] = Totals["PTS.1"].apply(pd.to_numeric)


# In[363]:


Totals["Home_Result"]=Totals["PTS.1"]-Totals["PTS"]
Totals["Home_Binary_Result"] = np.where(Totals["Home_Result"]>0,"1","0")


# In[364]:


UniqueDates = Totals["Date"].unique()


# In[1]:


UniqueDates = UniqueDates.tolist()


# In[2]:


UniqueDates = UniqueDates[5:8]


# In[366]:


Concatenated = pd.DataFrame()


# In[368]:


#for i in UniqueDates:
   # try:
        left = i[:2]
        middle = i[3:5]
        right = i[6:]
        driver = webdriver.Chrome(executable_path=r"/Users/mac/Desktop/chromedriver")
        url = 'https://stats.nba.com/teams/advanced/?sort=W&dir=-1&Season=2019-20&SeasonType=Regular%20Season&DateTo=' + str(left) + '%2F' + str(middle) + '%2F' + str(right) 
        driver.get(url)
        htmlSource = driver.page_source
        soup = BeautifulSoup(htmlSource, 'html.parser')
        table = soup.find('div', class_='nba-stat-table__overflow')
        df_list = pd.read_html(table.prettify())
        df_list1 = df_list[0]
        df_list2 = df_list1[["TEAM","OffRtg","DefRtg","PACE","PIE","AST  Ratio"]]
        driver.close();
        Date_with_Stats = Totals[Totals['Date']==str(i)]
        Date_with_Stats_Complete = Date_with_Stats.merge(df_list2, left_on = 'Home/Neutral', right_on = 'TEAM')
        Date_with_Stats_Complete = Date_with_Stats_Complete.merge(df_list2, left_on = 'Visitor/Neutral', right_on = 'TEAM')
        Concatenated = Concatenated.append(Date_with_Stats_Complete,ignore_index=True)
  #  except:
    #    continue
    


# In[377]:


ML_Inputs = Concatenated[["Date","Home/Neutral","OffRtg_x","DefRtg_x","PACE_x","PIE_x","AST  Ratio_x","Visitor/Neutral","OffRtg_y","DefRtg_y","PACE_y","PIE_y","AST  Ratio_y","Home_Result","Home_Binary_Result"]]


# In[378]:


ML_Inputs["Net_PACE"]=ML_Inputs["PACE_x"].apply(pd.to_numeric)-ML_Inputs["PACE_y"].apply(pd.to_numeric)#x is home #y is away
ML_Inputs["Net_AST_Ratio"]=ML_Inputs["AST  Ratio_x"].apply(pd.to_numeric)-ML_Inputs["AST  Ratio_y"].apply(pd.to_numeric)
ML_Inputs["Net_OFFRtg"]=ML_Inputs["OffRtg_x"].apply(pd.to_numeric)-ML_Inputs["OffRtg_y"].apply(pd.to_numeric)
ML_Inputs["Net_DefRtg"]=ML_Inputs["DefRtg_x"].apply(pd.to_numeric)-ML_Inputs["DefRtg_y"].apply(pd.to_numeric)
ML_Inputs["Net_PIE"]=ML_Inputs["PIE_x"].apply(pd.to_numeric)-ML_Inputs["PIE_y"].apply(pd.to_numeric)


# In[379]:


ML_Inputs = ML_Inputs[["Date","Home/Neutral","Visitor/Neutral",'Net_PACE','Net_AST_Ratio','Net_OFFRtg','Net_DefRtg','Net_PIE',"Home_Result","Home_Binary_Result"]]


# In[380]:


ML_Inputs


# In[385]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model


# In[386]:


x = ML_Inputs.values[:,3:8]
y = ML_Inputs.values[:,8]


# In[387]:


x = x.astype(float)
y = y.astype(float)


# In[388]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 100)


# In[389]:


LINREG = linear_model.LinearRegression()


# In[390]:


LINREG.fit(x_train,y_train)


# In[391]:


LINREG.coef_


# In[392]:


LINREG.intercept_ #home team advantage


# In[396]:


7.89502805e-04,  2.22729102e-01,  9.67607119e-01, -8.90571319e-01, -8.35687652e-02


# In[399]:


LINREG.score(x_test,y_test)


# In[400]:


ML_Inputs.to_csv("/Users/mac/Desktop/Ml_inputs2.csv")


# In[409]:


DF = pd.DataFrame(LINREG.predict(x_test))


# In[416]:


LINREG.fit(x_test,y_test)


# In[433]:


import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(x_test)
est = sm.OLS(y_test, X2)
est2 = est.fit()
print(est2.summary())


# In[ ]:




