import json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
C={"navy":"#1F3A5F","teal":"#2A9D8F","amber":"#E9A23B","red":"#C8553D","grey":"#8A94A0"}
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":170})
R1=json.load(open("results.json")); R2=json.load(open("results2.json"))
df=pd.read_csv("simulated_deliveries.csv"); df["b"]=(df.delivery_min>90).astype(int)
out={}
for col in ["weather","zone","is_peak"]:
    ct=pd.crosstab(df[col],df.b); chi2,p,dof,_=stats.chi2_contingency(ct)
    n=ct.values.sum(); v=np.sqrt(chi2/(n*(min(ct.shape)-1)))
    out[col]=dict(chi2=float(chi2),p=float(p),dof=int(dof),cramers_v=float(v))
json.dump(out,open("results3.json","w"),indent=1); print(out)
A=R1["assign"]; eta=R2["eta"]; sh=R2["shift"]; lp=R2["lp"]
fig,ax=plt.subplots(2,2,figsize=(7.6,6.0))
a=ax[0,0]; v=[A["random_mean"]/60,A["model"]/60,A["oracle"]/60]
a.bar(["Random","Model-based","Oracle"],v,color=[C["grey"],C["teal"],C["navy"]]); a.set_ylim(0,max(v)*1.15)
for i,x in enumerate(v): a.text(i,x+1.5,f"{x:.1f} h",ha="center",fontsize=8)
a.set_ylabel("Total delivery hours (120 orders)"); a.set_title("A. Order-driver assignment",loc="left",fontsize=9.5)
a=ax[0,1]; keys=["Mean + 10-min buffer","Mean + 15-min buffer","Raw P90 quantile model","Conformal P90 (quantile + offset)"]; labs=["Mean\n+10 min","Mean\n+15 min","Raw\nP90","Conformal\nP90"]
a.bar(range(4),[eta[k]["on_time"] for k in keys],color=[C["grey"],C["grey"],C["amber"],C["teal"]]); a.axhline(90,color=C["red"],ls="--",lw=1)
a.set_xticks(range(4)); a.set_xticklabels(labs,fontsize=8); a.set_ylim(80,98); a.set_ylabel("On-time rate (%)")
for i,k in enumerate(keys): a.text(i,eta[k]["on_time"]+.4,f"{eta[k]['on_time']:.1f}%\navg {eta[k]['avg_promise']:.0f} min",ha="center",fontsize=7)
a.set_title("B. Promise-time policies (target 90%)",loc="left",fontsize=9.5)
a=ax[1,0]; ss=list(sh); a.plot([float(s)*100 for s in ss],[sh[s]["breach"] for s in ss],marker="o",color=C["amber"],lw=2)
a.set_xlabel("% of peak orders shifted off-peak"); a.set_ylabel("Predicted orders > 90 min (%)"); a.set_title("C. Peak-load shifting what-if",loc="left",fontsize=9.5)
a=ax[1,1]; cats=["Cost","Breach rate"]; cur=[lp["current"]["cost"]/1000,lp["current"]["breach"]]; opt=[lp["optimised"]["cost"]/1000,lp["optimised"]["breach"]]
x=np.arange(2); a.bar(x-.18,cur,.36,color=C["grey"],label="Current mix"); a.bar(x+.18,opt,.36,color=C["teal"],label="LP-optimised")
for i in range(2):
    a.text(i-.18,cur[i]+.5,f"{cur[i]:.1f}",ha="center",fontsize=8); a.text(i+.18,opt[i]+.5,f"{opt[i]:.1f}",ha="center",fontsize=8)
a.set_xticks(x); a.set_xticklabels(["Cost (thousand units)","Breach rate (%)"]); a.legend(frameon=False,fontsize=8); a.set_ylim(0,32); a.set_title("D. Vehicle allocation (LP)",loc="left",fontsize=9.5)
plt.tight_layout(); plt.savefig("fig/optim.png"); plt.close()
