"""Follow-up: SLA=90 min, conformal-calibrated P90 ETA, capacity-constrained LP vehicle allocation, bottleneck what-ifs."""
import json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings("ignore"); rng=np.random.default_rng(7)
C={"navy":"#1F3A5F","teal":"#2A9D8F","amber":"#E9A23B","red":"#C8553D","grey":"#8A94A0"}
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
df=pd.read_csv("simulated_deliveries.csv")
num=["distance_km","weight_kg","hour","day_of_week","traffic_index","driver_exp_yrs","stops_on_route","is_peak"]; cat=["zone","weather","vehicle"]
X=df[num+cat]; y=df.delivery_min
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42); yt=yte.values
def prep(): return ColumnTransformer([("num",SimpleImputer(strategy="median"),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
P=dict(n_estimators=350,learning_rate=0.05,max_depth=4,subsample=0.7,min_samples_leaf=10,random_state=42)
best=Pipeline([("p",prep()),("m",GradientBoostingRegressor(**P))]).fit(Xtr,ytr)
pb=best.predict(Xte); R={"rmse_check":float(np.sqrt(np.mean((yt-pb)**2)))}
SLA=90; R["SLA"]=SLA
# ---- SLA breach by segment (observed, test+train = full data)
df["breach"]=(df.delivery_min>SLA).astype(int)
R["breach_overall"]=float(df.breach.mean())
R["breach_zone"]=df.groupby("zone").breach.mean().round(3).to_dict()
R["breach_weather"]=df.groupby("weather").breach.mean().round(3).to_dict()
R["breach_peak"]=df.groupby("is_peak").breach.mean().round(3).to_dict()
seg=df.groupby(["zone","is_peak"]).agg(n=("breach","size"),mean_min=("delivery_min","mean"),breach=("breach","mean")).round(3).reset_index()
R["seg_zone_peak"]=seg.to_dict("records")
worst=df.groupby(["zone","weather","is_peak"]).agg(n=("breach","size"),breach=("breach","mean"),mean_min=("delivery_min","mean")).reset_index()
worst=worst[worst.n>=40].sort_values("breach",ascending=False).head(5).round(3); R["worst_segments"]=worst.to_dict("records")
share=df.groupby(["zone","is_peak"]).breach.sum()/df.breach.sum(); R["breach_share_urban_peak"]=float(share.get(("Urban",1),0))
R["urban_peak_order_share"]=float(((df.zone=="Urban")&(df.is_peak==1)).mean())
# ---- Conformalised P90 ETA (CQR-style split conformal on the point-forecast residuals scaled by prediction)
Xfit,Xcal,yfit,ycal=train_test_split(Xtr,ytr,test_size=0.25,random_state=1)
pm=Pipeline([("p",prep()),("m",GradientBoostingRegressor(**P))]).fit(Xfit,yfit)
qm=Pipeline([("p",prep()),("m",GradientBoostingRegressor(loss="quantile",alpha=0.9,**P))]).fit(Xfit,yfit)
pc=pm.predict(Xcal); qc=qm.predict(Xcal)
score=(ycal.values-qc)           # CQR conformity score
k=int(np.ceil((len(score)+1)*0.9))-1; qhat=np.sort(score)[k]
pt=pm.predict(Xte); qt=qm.predict(Xte)
rel=(ycal.values-pc)/pc; relq=np.sort(rel)[k]            # scaled conformal on mean model
variants={"Mean + 10-min buffer":pt+10,"Mean + 15-min buffer":pt+15,"Raw P90 quantile model":np.maximum(qt,pt),
          "Conformal P90 (quantile + offset)":np.maximum(qt+qhat,pt),"Scaled conformal (mean x (1+q))":pt*(1+relq)}
eta={}
for lab,pr in variants.items():
    late=yt>pr
    eta[lab]=dict(on_time=float((~late).mean()*100),avg_promise=float(pr.mean()),
       urban=float((yt[(Xte.zone=="Urban").values]<=pr[(Xte.zone=="Urban").values]).mean()*100),
       rural=float((yt[(Xte.zone=="Rural").values]<=pr[(Xte.zone=="Rural").values]).mean()*100),
       storm=float((yt[(Xte.weather=="Storm").values]<=pr[(Xte.weather=="Storm").values]).mean()*100),
       late_by=float((yt-pr)[late].mean()) if late.any() else 0.0)
R["eta"]=eta; R["qhat"]=float(qhat); R["relq"]=float(relq)
# ---- Capacity-constrained LP: assign orders to vehicle classes
cost_min={"Bike":0.18,"Van":0.30,"Truck":0.55}; fixed={"Bike":0.6,"Van":1.4,"Truck":3.2}; V=list(cost_min)
T=np.column_stack([best.predict(Xte.assign(vehicle=v)) for v in V]); n=len(Xte)
cur=Xte.vehicle.map({v:i for i,v in enumerate(V)}).values
caps=np.ceil(np.bincount(cur,minlength=3)*1.15).astype(int)
w=Xte.weight_kg.values
Cm=np.column_stack([fixed[v]+cost_min[v]*T[:,i] for i,v in enumerate(V)])
Cm=Cm+8.0*(T>SLA)                      # SLA penalty ($) per predicted breach
ub=np.ones((n,3)); ub[:,0]=(w<=15).astype(float)          # bikes cannot carry >15 kg
c=Cm.ravel(); Aeq=np.zeros((n,3*n)); 
for i in range(n): Aeq[i,3*i:3*i+3]=1
Aub=np.zeros((3,3*n))
for j in range(3): Aub[j,j::3]=1
res=linprog(c,A_ub=Aub,b_ub=caps,A_eq=Aeq,b_eq=np.ones(n),bounds=list(zip([0]*(3*n),ub.ravel())),method="highs")
Xlp=res.x.reshape(n,3).argmax(1)
def stats_(a):
    tt=T[np.arange(n),a]; cost=(np.array([fixed[V[j]] for j in a])+np.array([cost_min[V[j]] for j in a])*tt)
    return dict(cost=float(cost.sum()),mean_min=float(tt.mean()),breach=float((tt>SLA).mean()*100),
                mix={V[j]:float((a==j).mean()) for j in range(3)})
R["lp"]=dict(status=int(res.status),caps=dict(zip(V,map(int,caps))),current=stats_(cur),optimised=stats_(Xlp))
R["lp"]["saving_pct"]=(R["lp"]["current"]["cost"]-R["lp"]["optimised"]["cost"])/R["lp"]["current"]["cost"]*100
# ---- Peak shifting what-if (SLA 90)
sc=Xte.copy(); peak=(sc.is_peak==1).values; shift={}
for s in [0,0.1,0.2,0.3,0.4]:
    s2=sc.copy(); mv=peak&(rng.random(n)<s); s2.loc[mv,"hour"]=10; s2.loc[mv,"is_peak"]=0; s2.loc[mv,"traffic_index"]=np.clip(s2.loc[mv,"traffic_index"]-4,0,10)
    p=best.predict(s2); shift[s]=dict(mean=float(p.mean()),breach=float((p>SLA).mean()*100),hours=float(p.sum()/60))
R["shift"]=shift
# ---- Route-density what-if: cap stops per route
dens={}
for cap in [15,12,10,8]:
    s2=sc.copy(); s2["stops_on_route"]=np.minimum(s2.stops_on_route,cap); p=best.predict(s2); dens[cap]=dict(mean=float(p.mean()),breach=float((p>SLA).mean()*100))
R["density"]=dens; R["stops_gt10_share"]=float((sc.stops_on_route>10).mean())
# ---- Distance-banded error / bottleneck: share of predicted time explained by stops+distance
R["pred_mean"]=float(pb.mean())
# ---- Figures
fig,ax=plt.subplots(1,3,figsize=(11,3.4))
ax[0].bar(["Random","Model-based\nassignment"],[8214.06/60,7970.63/60],color=[C["grey"],C["teal"]]); ax[0].set_ylabel("Total delivery hours (120 orders)"); ax[0].set_ylim(120,140)
for i,v in enumerate([8214.06/60,7970.63/60]): ax[0].text(i,v+.3,f"{v:.1f}",ha="center",fontsize=8)
ax[0].set_title("Order-driver assignment",loc="left",fontsize=10)
labs=["+10 min","+15 min","Raw P90","Conformal P90"]; keys=["Mean + 10-min buffer","Mean + 15-min buffer","Raw P90 quantile model","Conformal P90 (quantile + offset)"]
x=np.arange(4); ax[1].bar(x,[eta[k]["on_time"] for k in keys],color=[C["grey"],C["grey"],C["amber"],C["teal"]]); ax[1].axhline(90,color=C["red"],ls="--",lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels(labs,fontsize=8); ax[1].set_ylim(80,100); ax[1].set_ylabel("On-time (%)"); ax[1].set_title("Promise-time policies",loc="left",fontsize=10)
for i,k in enumerate(keys): ax[1].text(i,eta[k]["on_time"]+.3,f"{eta[k]['on_time']:.1f}\n(avg {eta[k]['avg_promise']:.0f}m)",ha="center",fontsize=7)
ss=list(shift); ax[2].plot([s*100 for s in ss],[shift[s]["breach"] for s in ss],marker="o",color=C["amber"],lw=2); ax[2].set_xlabel("% of peak orders shifted off-peak"); ax[2].set_ylabel(f"Predicted >{SLA} min (%)"); ax[2].set_title("Peak-load shifting what-if",loc="left",fontsize=10)
plt.tight_layout(); plt.savefig("fig/optim.png"); plt.close()
fig,ax=plt.subplots(figsize=(6.5,3.2)); labs=[f"{z}\n{'peak' if p else 'off-peak'}" for z in ["Rural","Suburban","Urban"] for p in [0,1]]
vals=[df[(df.zone==z)&(df.is_peak==p)].breach.mean()*100 for z in ["Rural","Suburban","Urban"] for p in [0,1]]
ax.bar(labs,vals,color=[C["grey"],C["amber"]]*3); ax.set_ylabel(f"Orders over {SLA} min (%)")
for i,v in enumerate(vals): ax.text(i,v+.5,f"{v:.0f}%",ha="center",fontsize=8)
plt.tight_layout(); plt.savefig("fig/breach_seg.png"); plt.close()
def clean(o):
    if isinstance(o,dict): return {str(k):clean(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [clean(v) for v in o]
    if isinstance(o,(np.floating,np.integer)): return o.item()
    return o
json.dump(clean(R),open("results2.json","w"),indent=1); print("done")
