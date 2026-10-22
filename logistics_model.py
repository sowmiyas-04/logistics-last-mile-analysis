"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Problem: forecast last-mile delivery time (minutes) and use the forecast to optimise dispatch.
"""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)
R = {}   # results store
C = {"navy":"#1F3A5F","teal":"#2A9D8F","amber":"#E9A23B","red":"#C8553D","grey":"#8A94A0"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})

# ---------------------------------------------------------------- 1. DATA SIMULATION
def true_minutes(d):
    """Noise-free delivery time (the 'physics' of the simulator)."""
    base = 8 + 2.6*d.distance_km
    base = base*np.where(d.zone=="Urban",1.15,np.where(d.zone=="Suburban",1.0,0.9))
    cong = 1 + 0.55*(d.traffic_index/10)**2*np.where(d.zone=="Urban",1.4,0.7)
    wx = d.weather.map({"Clear":1.0,"Rain":1.18,"Storm":1.45}).values
    veh = d.vehicle.map({"Bike":1.0,"Van":0.9,"Truck":1.1}).values
    veh_w = np.where(d.vehicle=="Bike", 1+0.05*np.maximum(d.weight_kg-5,0), 1.0)
    exp = 1.25 - 0.25*(1-np.exp(-d.driver_exp_yrs/3))
    stops = 3.2*d.stops_on_route
    peak = 1 + 0.12*d.is_peak
    return (base*cong*wx*veh*veh_w*exp*peak + stops + 0.6*d.weight_kg)

def simulate(n):
    d = pd.DataFrame({
        "distance_km": np.clip(rng.gamma(3.0,2.2,n),0.5,35),
        "weight_kg":   np.clip(rng.lognormal(1.2,0.7,n),0.2,40),
        "hour":        rng.choice(np.arange(7,22),n,p=np.array([4,7,8,6,5,7,9,6,5,6,9,10,8,5,3])/np.array([4,7,8,6,5,7,9,6,5,6,9,10,8,5,3]).sum()),
        "day_of_week": rng.integers(0,7,n),
        "zone":        rng.choice(["Urban","Suburban","Rural"],n,p=[.5,.35,.15]),
        "weather":     rng.choice(["Clear","Rain","Storm"],n,p=[.72,.22,.06]),
        "vehicle":     rng.choice(["Bike","Van","Truck"],n,p=[.35,.5,.15]),
        "driver_exp_yrs": np.clip(rng.gamma(2.2,1.6,n),0.1,15),
        "stops_on_route": rng.integers(1,16,n),
    })
    d["is_peak"] = d.hour.isin([8,9,12,13,17,18,19]).astype(int)
    d["traffic_index"] = np.clip(3 + 4*d.is_peak + 1.2*(d.zone=="Urban") + 0.9*(d.day_of_week<5) + rng.normal(0,1.2,n),0,10)
    mu = true_minutes(d)
    d["delivery_min"] = mu*rng.lognormal(0,0.12,n) + rng.normal(0,1.5,n)   # multiplicative + additive noise
    d["delivery_min"] = d.delivery_min.clip(lower=5)
    return d

df = simulate(6000)
# inject realistic data-quality issues
miss = rng.choice(df.index,150,replace=False); df.loc[miss[:90],"traffic_index"] = np.nan
df.loc[miss[90:],"driver_exp_yrs"] = np.nan
R["n"]=len(df); R["missing"]=df.isna().sum()[lambda s:s>0].to_dict()
R["target_desc"]=df.delivery_min.describe().round(2).to_dict()
R["skew"]=float(stats.skew(df.delivery_min))
df.to_csv("/home/claude/work/simulated_deliveries.csv",index=False)

# ---------------------------------------------------------------- 2. EDA + STATISTICAL TESTS
eda = {}
# Welch t-test peak vs off-peak
a,b = df[df.is_peak==1].delivery_min, df[df.is_peak==0].delivery_min
t,p = stats.ttest_ind(a,b,equal_var=False); pooled=np.sqrt((a.var()+b.var())/2)
eda["peak"] = dict(mean_peak=a.mean(),mean_off=b.mean(),t=t,p=p,cohen_d=(a.mean()-b.mean())/pooled)
# Kruskal-Wallis weather and zone (+ eta-squared via H)
for col in ["weather","zone","vehicle"]:
    groups=[g.delivery_min.values for _,g in df.groupby(col)]
    H,p=stats.kruskal(*groups); k=len(groups); eta=(H-k+1)/(len(df)-k)
    eda[col]=dict(H=H,p=p,eta2=eta,means=df.groupby(col).delivery_min.mean().round(1).to_dict())
# Spearman correlations
eda["spearman"]={c:dict(zip(["rho","p"],stats.spearmanr(df[c],df.delivery_min,nan_policy="omit"))) for c in ["distance_km","weight_kg","traffic_index","driver_exp_yrs","stops_on_route"]}
# Chi-square SLA breach (>60 min) by weather
df["breach60"]=(df.delivery_min>60).astype(int)
ct=pd.crosstab(df.weather,df.breach60); chi2,p,dof,_=stats.chi2_contingency(ct)
eda["chi_weather_breach"]=dict(chi2=chi2,p=p,dof=dof,rates=df.groupby("weather").breach60.mean().round(3).to_dict())
ct=pd.crosstab(df.zone,df.breach60); chi2,p,dof,_=stats.chi2_contingency(ct)
eda["chi_zone_breach"]=dict(chi2=chi2,p=p,dof=dof,rates=df.groupby("zone").breach60.mean().round(3).to_dict())
eda["breach_overall"]=df.breach60.mean()
R["eda"]=eda

# ---------------------------------------------------------------- 3. PIPELINE + MODELS
num=["distance_km","weight_kg","hour","day_of_week","traffic_index","driver_exp_yrs","stops_on_route","is_peak"]
cat=["zone","weather","vehicle"]
from sklearn.impute import SimpleImputer
def prep(scale):
    n=[("imp",SimpleImputer(strategy="median"))]+([("sc",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(n),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
X=df[num+cat]; y=df.delivery_min
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
models={
 "Linear Regression":Pipeline([("p",prep(True)),("m",LinearRegression())]),
 "Ridge (alpha=10)":Pipeline([("p",prep(True)),("m",Ridge(alpha=10))]),
 "Decision Tree (depth 8)":Pipeline([("p",prep(False)),("m",DecisionTreeRegressor(max_depth=8,min_samples_leaf=10,random_state=42))]),
 "Random Forest":Pipeline([("p",prep(False)),("m",RandomForestRegressor(n_estimators=100,min_samples_leaf=3,n_jobs=-1,random_state=42))]),
 "Gradient Boosting":Pipeline([("p",prep(False)),("m",GradientBoostingRegressor(random_state=42))]),
}
kf=KFold(5,shuffle=True,random_state=42)
cvres={}; fold_rmse={}
for name,m in models.items():
    s=cross_validate(m,Xtr,ytr,cv=kf,scoring=("neg_root_mean_squared_error","neg_mean_absolute_error","r2"),n_jobs=-1)
    cvres[name]=dict(rmse=-s["test_neg_root_mean_squared_error"].mean(),rmse_sd=s["test_neg_root_mean_squared_error"].std(),
                     mae=-s["test_neg_mean_absolute_error"].mean(),r2=s["test_r2"].mean())
    fold_rmse[name]=-s["test_neg_root_mean_squared_error"]
R["cv"]=cvres

# hyper-parameter tuning (Gradient Boosting)
grid={"m__n_estimators":[150,250,350],"m__learning_rate":[0.03,0.05,0.1],"m__max_depth":[2,3,4],
      "m__subsample":[0.7,0.85,1.0],"m__min_samples_leaf":[5,10,20]}
rs=RandomizedSearchCV(models["Gradient Boosting"],grid,n_iter=12,cv=3,scoring="neg_root_mean_squared_error",random_state=42,n_jobs=-1)
rs.fit(Xtr,ytr)
R["best_params"]={k.replace("m__",""):v for k,v in rs.best_params_.items()}; R["tuned_cv_rmse"]=-rs.best_score_
tuned=rs.best_estimator_
s=cross_validate(tuned,Xtr,ytr,cv=kf,scoring="neg_root_mean_squared_error",n_jobs=-1)
fold_rmse["Tuned Gradient Boosting"]=-s["test_score"]
cvres["Tuned Gradient Boosting"]=dict(rmse=-s["test_score"].mean(),rmse_sd=s["test_score"].std(),mae=np.nan,r2=np.nan)

# final test evaluation
def mape(a,b): return float(np.mean(np.abs((a-b)/a))*100)
test={}
fitted={}
for name,m in list(models.items())+[("Tuned Gradient Boosting",tuned)]:
    m.fit(Xtr,ytr) if name!="Tuned Gradient Boosting" else None
    pr=m.predict(Xte); fitted[name]=pr
    test[name]=dict(rmse=float(np.sqrt(mean_squared_error(yte,pr))),mae=float(mean_absolute_error(yte,pr)),r2=float(r2_score(yte,pr)),mape=mape(yte,pr))
R["test"]=test
best=tuned; pbest=fitted["Tuned Gradient Boosting"]
# fill CV mae / r2 for tuned
s2=cross_validate(tuned,Xtr,ytr,cv=kf,scoring=("neg_mean_absolute_error","r2"),n_jobs=-1)
cvres["Tuned Gradient Boosting"]["mae"]=-s2["test_neg_mean_absolute_error"].mean(); cvres["Tuned Gradient Boosting"]["r2"]=s2["test_r2"].mean()

# paired tests on CV fold RMSE
def paired(a,b):
    t,p=stats.ttest_rel(fold_rmse[a],fold_rmse[b]); w=stats.wilcoxon(fold_rmse[a],fold_rmse[b])
    return dict(mean_diff=float((fold_rmse[a]-fold_rmse[b]).mean()),t=float(t),p_t=float(p),p_wilcoxon=float(w.pvalue))
R["paired"]={"Linear vs Tuned GB":paired("Linear Regression","Tuned Gradient Boosting"),
             "Random Forest vs Tuned GB":paired("Random Forest","Tuned Gradient Boosting"),
             "Default GB vs Tuned GB":paired("Gradient Boosting","Tuned Gradient Boosting")}

# bootstrap CI on test RMSE and MAE
idx=np.arange(len(yte)); bs=[]
yt=yte.values
for _ in range(2000):
    i=rng.choice(idx,len(idx)); bs.append((np.sqrt(np.mean((yt[i]-pbest[i])**2)),np.mean(np.abs(yt[i]-pbest[i]))))
bs=np.array(bs); R["boot"]=dict(rmse_ci=np.percentile(bs[:,0],[2.5,97.5]).tolist(),mae_ci=np.percentile(bs[:,1],[2.5,97.5]).tolist())

# residual diagnostics
res=yt-pbest
sw=stats.shapiro(rng.choice(res,500,replace=False))
lr=stats.linregress(pbest,res**2)  # simplified Breusch-Pagan style heteroscedasticity check
rel=np.abs(res)/yt
R["diag"]=dict(res_mean=float(res.mean()),res_sd=float(res.std()),shapiro_p=float(sw.pvalue),skew=float(stats.skew(res)),
               het_slope=float(lr.slope),het_p=float(lr.pvalue),het_r=float(lr.rvalue))
# error by segment
seg=pd.DataFrame({"zone":Xte.zone.values,"weather":Xte.weather.values,"peak":Xte.is_peak.values,"abs_err":np.abs(res),"y":yt})
R["seg_mae"]={c:seg.groupby(c).abs_err.mean().round(2).to_dict() for c in ["zone","weather","peak"]}

# permutation importance
pi=permutation_importance(best,Xte,yte,n_repeats=15,random_state=42,scoring="neg_root_mean_squared_error",n_jobs=-1)
imp=pd.Series(pi.importances_mean,index=X.columns).sort_values(ascending=False)
imp_sd=pd.Series(pi.importances_std,index=X.columns)[imp.index]
R["importance"]={k:[float(v),float(imp_sd[k])] for k,v in imp.items()}

# ---------------------------------------------------------------- 4. OPTIMISATION
# 4A. Order-to-driver assignment (Hungarian / linear assignment on predicted minutes)
N=120
orders=Xte.dropna(subset=["traffic_index"]).iloc[:N].reset_index(drop=True)
drv=pd.DataFrame({"vehicle":rng.choice(["Bike","Van","Truck"],N,p=[.35,.5,.15]),"driver_exp_yrs":np.clip(rng.gamma(2.2,1.6,N),0.1,15)})
def pair_frame(o,dr):
    f=o.copy(); f["vehicle"]=dr.vehicle; f["driver_exp_yrs"]=dr.driver_exp_yrs; return f
pred_M=np.zeros((N,N)); true_M=np.zeros((N,N))
for j in range(N):
    f=pair_frame(orders,pd.DataFrame({"vehicle":[drv.vehicle[j]]*N,"driver_exp_yrs":[drv.driver_exp_yrs[j]]*N}))
    pred_M[:,j]=best.predict(f); true_M[:,j]=true_minutes(f)
r,c=linear_sum_assignment(pred_M); r0,c0=linear_sum_assignment(true_M)
opt_true=true_M[r,c].sum()
oracle=true_M[r0,c0].sum()
rand_costs=[true_M[np.arange(N),rng.permutation(N)].sum() for _ in range(500)]
naive=true_M[np.arange(N),np.arange(N)].sum()
R["assign"]=dict(N=N,random_mean=float(np.mean(rand_costs)),naive=float(naive),model=float(opt_true),oracle=float(oracle),
                 saving_vs_random_pct=float((np.mean(rand_costs)-opt_true)/np.mean(rand_costs)*100),
                 gap_to_oracle_pct=float((opt_true-oracle)/oracle*100),
                 per_order_saving=float((np.mean(rand_costs)-opt_true)/N))
wrand=[ (true_M[np.arange(N),p]).max() for p in [rng.permutation(N) for _ in range(500)]]
R["assign"]["max_random"]=float(np.mean(wrand)); R["assign"]["max_model"]=float(true_M[r,c].max())
# paired test across random baselines
R["assign"]["pct_random_worse"]=float(np.mean(np.array(rand_costs)>opt_true)*100)

# 4B. Promise-time (ETA) setting with quantile regression
bp=rs.best_params_
qm={}
for q in [0.5,0.9]:
    g=Pipeline([("p",prep(False)),("m",GradientBoostingRegressor(loss="quantile",alpha=q,n_estimators=bp["m__n_estimators"],learning_rate=bp["m__learning_rate"],max_depth=bp["m__max_depth"],subsample=bp["m__subsample"],min_samples_leaf=bp["m__min_samples_leaf"],random_state=42))])
    g.fit(Xtr,ytr); qm[q]=g.predict(Xte)
fixed=pbest+10
q90=np.maximum(qm[0.9],pbest)
eta={}
for lab,pr in [("Mean + fixed 10-min buffer",fixed),("Mean + fixed 15-min buffer",pbest+15),("P90 quantile model",q90)]:
    eta[lab]=dict(on_time=float((yt<=pr).mean()*100),avg_promise=float(pr.mean()),late_by=float(np.maximum(yt-pr,0)[yt>pr].mean()) if (yt>pr).any() else 0.0)
R["eta"]=eta
# calibration of P90 by zone/weather
R["eta_by_weather"]={w:dict(fixed=float((yt[(Xte.weather==w).values]<=fixed[(Xte.weather==w).values]).mean()*100),
                            q90=float((yt[(Xte.weather==w).values]<=q90[(Xte.weather==w).values]).mean()*100)) for w in ["Clear","Rain","Storm"]}
R["eta_by_zone"]={w:dict(fixed=float((yt[(Xte.zone==w).values]<=fixed[(Xte.zone==w).values]).mean()*100),
                         q90=float((yt[(Xte.zone==w).values]<=q90[(Xte.zone==w).values]).mean()*100)) for w in ["Urban","Suburban","Rural"]}

# 4C. Peak-load shifting what-if + 4D capacity/cost LP-style allocation by vehicle
sc=Xte.copy(); base_pred=best.predict(sc); peak_mask=(sc.is_peak==1).values
res_shift={}
for share in [0,0.1,0.2,0.3,0.4]:
    s2=sc.copy(); mv=peak_mask & (rng.random(len(sc))<share)
    s2.loc[mv,"hour"]=10; s2.loc[mv,"is_peak"]=0
    s2.loc[mv,"traffic_index"]=np.clip(s2.loc[mv,"traffic_index"]-4,0,10)
    p=best.predict(s2); res_shift[share]=dict(mean=float(p.mean()),breach=float((p>60).mean()*100),total_hours=float(p.sum()/60))
R["shift"]=res_shift
# vehicle-mix what-if: for each test order, predicted time and cost per vehicle; choose cheapest meeting 60-min SLA
cost_per_min={"Bike":0.18,"Van":0.30,"Truck":0.55}; fixed_cost={"Bike":0.6,"Van":1.4,"Truck":3.2}
alloc=sc.copy(); costs={}; times={}
for v in cost_per_min:
    a=sc.copy(); a["vehicle"]=v; times[v]=best.predict(a); costs[v]=fixed_cost[v]+cost_per_min[v]*times[v]
T=np.column_stack([times[v] for v in cost_per_min]); Cm=np.column_stack([costs[v] for v in cost_per_min])
# feasibility: truck only if weight>20, bike only if weight<=15
w=sc.weight_kg.values; feas=np.ones_like(Cm,bool); feas[:,2]=w>20*0+10; feas[:,0]=w<=15
cur=sc.vehicle.map({"Bike":0,"Van":1,"Truck":2}).values
cur_cost=Cm[np.arange(len(sc)),cur]
Cm_f=np.where(feas&(T<=60),Cm,np.where(feas,Cm+3,1e9)); new=Cm_f.argmin(1)
new_cost=Cm[np.arange(len(sc)),new]
R["vehicle_alloc"]=dict(cur_cost=float(cur_cost.sum()),new_cost=float(new_cost.sum()),saving_pct=float((cur_cost.sum()-new_cost.sum())/cur_cost.sum()*100),
    cur_mix=pd.Series(cur).map({0:"Bike",1:"Van",2:"Truck"}).value_counts(normalize=True).round(3).to_dict(),
    new_mix=pd.Series(new).map({0:"Bike",1:"Van",2:"Truck"}).value_counts(normalize=True).round(3).to_dict(),
    cur_breach=float((T[np.arange(len(sc)),cur]>60).mean()*100),new_breach=float((T[np.arange(len(sc)),new]>60).mean()*100))

# ---------------------------------------------------------------- 5. FIGURES
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
ax[0].hist(df.delivery_min,bins=50,color=C["navy"],alpha=.9); ax[0].axvline(60,color=C["red"],ls="--",lw=1.2); ax[0].text(61,ax[0].get_ylim()[1]*.9,"60-min SLA",color=C["red"],fontsize=8)
ax[0].set_xlabel("Delivery time (min)"); ax[0].set_ylabel("Orders"); ax[0].set_title("Target is right-skewed",loc="left",fontsize=10)
order=["Rural","Suburban","Urban"]; ax[1].boxplot([df[df.zone==z].delivery_min for z in order],labels=order,patch_artist=True,boxprops=dict(facecolor=C["teal"],alpha=.6),medianprops=dict(color=C["navy"]),flierprops=dict(markersize=2,alpha=.4))
ax[1].set_ylabel("Delivery time (min)"); ax[1].set_title("Delivery time by zone",loc="left",fontsize=10)
plt.tight_layout(); plt.savefig("/home/claude/work/fig/eda.png"); plt.close()

names=list(cvres.keys()); vals=[cvres[n]["rmse"] for n in names]; errs=[cvres[n]["rmse_sd"] for n in names]
fig,ax=plt.subplots(figsize=(7.5,3.3)); cols=[C["grey"]]*len(names); cols[-1]=C["teal"]
ax.barh(names[::-1],vals[::-1],xerr=errs[::-1],color=cols[::-1],capsize=3)
for i,v in enumerate(vals[::-1]): ax.text(v+0.4,i,f"{v:.2f}",va="center",fontsize=8)
ax.set_xlabel("10-fold CV RMSE (minutes, lower is better)"); plt.tight_layout(); plt.savefig("/home/claude/work/fig/models.png"); plt.close()

fig,ax=plt.subplots(1,2,figsize=(9,3.6))
ax[0].scatter(yt,pbest,s=5,alpha=.35,color=C["navy"]); lim=[0,max(yt.max(),pbest.max())]; ax[0].plot(lim,lim,color=C["red"],lw=1)
ax[0].set_xlabel("Actual (min)"); ax[0].set_ylabel("Predicted (min)"); ax[0].set_title("Predicted vs actual (test set)",loc="left",fontsize=10)
ax[1].scatter(pbest,res,s=5,alpha=.35,color=C["teal"]); ax[1].axhline(0,color=C["red"],lw=1)
ax[1].set_xlabel("Predicted (min)"); ax[1].set_ylabel("Residual (min)"); ax[1].set_title("Residuals widen with trip length",loc="left",fontsize=10)
plt.tight_layout(); plt.savefig("/home/claude/work/fig/fit.png"); plt.close()

fig,ax=plt.subplots(figsize=(7.5,3.6)); top=imp.head(9)[::-1]
ax.barh(top.index,top.values,xerr=imp_sd[top.index].values,color=C["navy"],capsize=3); ax.set_xlabel("Increase in test RMSE when feature is shuffled (min)")
plt.tight_layout(); plt.savefig("/home/claude/work/fig/importance.png"); plt.close()

fig,ax=plt.subplots(1,2,figsize=(9,3.5))
lab=["Naive\n(as listed)","Random\n(mean)","Model-based\n(Hungarian)","Oracle"]; v=[naive,np.mean(rand_costs),opt_true,oracle]
ax[0].bar(lab,[x/60 for x in v],color=[C["grey"],C["grey"],C["teal"],C["navy"]]); ax[0].set_ylabel("Total delivery hours (120 orders)")
ax[0].set_title("Order-driver assignment",loc="left",fontsize=10)
for i,x in enumerate(v): ax[0].text(i,x/60+.5,f"{x/60:.1f}",ha="center",fontsize=8)
shares=list(res_shift.keys()); ax[1].plot([s*100 for s in shares],[res_shift[s]["breach"] for s in shares],marker="o",color=C["amber"],lw=2)
ax[1].set_xlabel("% of peak orders shifted to mid-morning"); ax[1].set_ylabel("Predicted >60 min (%)"); ax[1].set_title("Peak-load shifting what-if",loc="left",fontsize=10)
plt.tight_layout(); plt.savefig("/home/claude/work/fig/optim.png"); plt.close()

def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,np.integer)): return o.item()
    return o
json.dump(clean(R),open("/home/claude/work/results.json","w"),indent=1)
print(json.dumps(clean(R),indent=1))
