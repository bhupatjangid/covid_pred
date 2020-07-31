import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import sklearn
from sklearn.linear_model import LinearRegression

#preprocessing methods
from sklearn.model_selection import train_test_split

confirmed_cases=pd.read_csv("confirmed_global.csv")
death_reported=pd.read_csv('deaths_global.csv')
recovered_cases=pd.read_csv('recovered_global.csv')

cols=confirmed_cases.keys()

confirmed=confirmed_cases.loc[:,cols[4]:cols[-1]]
#print(confirmed)
#265 contries
recovered=recovered_cases.loc[:,cols[4]:cols[-1]]
#print(recovered)
#265 contries
deaths=death_reported.loc[:,cols[4]:cols[-1]]
#print(deaths)
#265 contries


dates=confirmed.keys()
dates

world_cases=[]
total_deaths=[]
martality_rate=[]
total_recovered=[]
india_cases=[]

for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recovered[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    martality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)
    india_cases.append(confirmed_cases[confirmed_cases['Country/Region']=='India'][i].sum())
    #print(confirmed[i])
#print(world_cases)

#changing days into days
v=1
day_date=[]
for i in range(len(dates)):
    v=i*1
    day_date.append(v)
#day_date

#plot
plt.figure(figsize=(20,12))
plt.plot(day_date,world_cases)
plt.plot(day_date,total_recovered,color='green')
plt.plot(day_date,total_deaths,color="red")
plt.title("corona cases,deaths with time",size=30)
plt.xlabel("days",size=30)
plt.ylabel("count of cases",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
#plt.show()

#check each day cases
def eachday_increase(records):
    d=[]
    for i in range(len(records)):
        if i==0:
            d.append(records[0])
        else:
            d.append(records[i]-records[i-1])
    return d

world_daily_increase=eachday_increase(world_cases)
#world_daily_increase

#polynomial reg.
#print(type(world_cases))

world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)
days=np.array(day_date).reshape(-1,1)
#print(days.shape)

days_in_future=10
future_forcast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forcast[:-10]

#adjusted_dates
x_train_confirmed,x_test_confirmed,y_train_confirmed,y_test_confirmed = train_test_split(days,world_cases,test_size=0.25,shuffle=False)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
poly_x_train_confirmed=poly.fit_transform(x_train_confirmed)
poly_x_test_confirmed=poly.fit_transform(x_test_confirmed)
poly_future_forcast=poly.fit_transform(future_forcast)

model=LinearRegression()
model.fit(poly_x_train_confirmed,y_train_confirmed)
test_pred=model.predict(poly_x_test_confirmed)
predictions=model.predict(poly_future_forcast)

from sklearn.metrics import mean_squared_error, mean_absolute_error
mean_squared_error(test_pred,y_test_confirmed)
mean_absolute_error(test_pred,y_test_confirmed)
plt.plot(y_test_confirmed,color='red')
plt.plot(test_pred)