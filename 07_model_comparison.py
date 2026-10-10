"""
Model comparison on the same Sep-Oct 2025 holdout (train on Jan-Aug, cleaned data,
same features without market columns). Lower MAE is better.
Linear Regression and Random Forest need filled missing values, so we use the median.
"""
import time, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error as mae
t=pd.read_csv('data/train_test.csv',parse_dates=['date']); t['weight']=t.weight.abs(); t['rpm']=t.posted_rate/t.distance
c=t[(t.rpm>=0.8)&(t.rpm<=6)].copy(); c['month']=c.date.dt.month; c['dow']=c.date.dt.dayofweek
c=pd.concat([c,pd.get_dummies(c.equipment,dtype=float)],axis=1)
num=['distance','weight','month','dow','pickup_lat','pickup_lon','delivery_lat','delivery_lon','Dry Van','Flatbed','Reefer']
fit=c[c.date<'2025-09-01']; ho=c[c.date>='2025-09-01']
med=fit[num].median(); Xf=fit[num].fillna(med); Xh=ho[num].fillna(med)
res=[]
def run(name,m,Xa,Xb):
    s=time.time(); m.fit(Xa,fit.posted_rate); p=m.predict(Xb); res.append((name,round(mae(ho.posted_rate,p),1),round(time.time()-s,1)))
res.append(('Baseline (distance x avg $/mile)',round(mae(ho.posted_rate,ho.distance*fit.rpm.mean()),1),0))
run('Linear Regression',LinearRegression(),Xf,Xh)
run('Random Forest (200 trees)',RandomForestRegressor(n_estimators=200,n_jobs=-1,random_state=0),Xf,Xh)
run('sklearn HistGradientBoosting',HistGradientBoostingRegressor(max_iter=500,learning_rate=0.05,random_state=0),fit[num],ho[num])
run('LightGBM (our model)',lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,verbose=-1),fit[num],ho[num])
for r in res: print(r)