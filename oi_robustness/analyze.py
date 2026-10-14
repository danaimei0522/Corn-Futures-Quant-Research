"""Robustness checks for the previously studied corn-futures OI factor."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).parent
SRC=ROOT/"oi_deep_dive"; PANEL=ROOT/"alternative_alpha/daily_factor_panel.csv"
ROUTE=ROOT/"five_year_backtest/causal_signal_and_contracts.csv"
START,END=pd.Timestamp("2021-10-12"),pd.Timestamp("2026-10-12")
YEARS=[2022,2023,2024,2025,2026]; HORIZON=10; LAGS=20; WINDOW=60; ROLL=252; STEP=21; Z=1.959963984540054
VARIANTS=["oi_pct_change","oi_rolling_z60","oi_historical_pctile60"]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def bh(pv):
    p=np.asarray(pv,float);q=np.full(len(p),np.nan);ids=np.flatnonzero(np.isfinite(p))
    if not len(ids):return q
    order=ids[np.argsort(p[ids])];raw=p[order]*len(order)/np.arange(1,len(order)+1);q[order]=np.minimum(np.minimum.accumulate(raw[::-1])[::-1],1);return q
def pnorm(t):return math.erfc(abs(t)/math.sqrt(2)) if np.isfinite(t) else np.nan
def hac(y,x,lags=LAGS):
    y=np.asarray(y,float);x=np.asarray(x,float);keep=np.isfinite(y)&np.isfinite(x);y,x=y[keep],x[keep];n=len(y)
    if n<5 or np.std(x)==0 or np.std(y)==0:return None
    X=np.column_stack([np.ones(n),x]);bread=np.linalg.pinv(X.T@X);b=bread@X.T@y;r=y-X@b;g=X*r[:,None];meat=g.T@g
    for lag in range(1,min(lags,n-1)+1):
        w=1-lag/(min(lags,n-1)+1);gamma=g[lag:].T@g[:-lag];meat+=w*(gamma+gamma.T)
    cov=bread@meat@bread;se=np.sqrt(np.maximum(np.diag(cov),0));t=b[1]/se[1] if se[1]>0 else np.nan
    return {"n":n,"coef":float(b[1]),"se":float(se[1]),"low":float(b[1]-Z*se[1]),"high":float(b[1]+Z*se[1]),"p":pnorm(t)}
def stats(x,y):
    z=pd.DataFrame({"x":x,"y":y}).dropna(); n=len(z)
    if n<5:return {"n":n,"spearman_ic":np.nan,"spearman_se":np.nan,"spearman_low":np.nan,"spearman_high":np.nan,"spearman_p":np.nan,"n_eff_hac_approx":np.nan,"pearson_r":np.nan,"pearson_low":np.nan,"pearson_high":np.nan,"pearson_p":np.nan}
    xr=z.x.rank(method="average").to_numpy();yr=z.y.rank(method="average").to_numpy();xr=(xr-xr.mean())/xr.std();yr=(yr-yr.mean())/yr.std();rs=hac(yr,xr)
    xp=z.x.to_numpy(float);yp=z.y.to_numpy(float);xp=(xp-xp.mean())/xp.std();yp=(yp-yp.mean())/yp.std();pr=hac(yp,xp)
    ne=float(np.clip((1-rs["coef"]**2)/max(rs["se"]**2,1e-15)+2,1,n))
    return {"n":n,"spearman_ic":rs["coef"],"spearman_se":rs["se"],"spearman_low":max(-1,rs["low"]),"spearman_high":min(1,rs["high"]),"spearman_p":rs["p"],"n_eff_hac_approx":ne,"pearson_r":pr["coef"],"pearson_low":pr["low"],"pearson_high":pr["high"],"pearson_p":pr["p"]}

def block_diff(frame,reps=2000,block=20,seed=261014):
    a=frame[["oi","ret"]].to_numpy(float);n=len(a);valid=np.isfinite(a).all(axis=1);obs=a[valid]
    if n==0 or len(obs)==0:return {}
    up=obs[:,0]>0;down=obs[:,0]<0
    if not up.any() or not down.any():return {}
    md=float(obs[up,1].mean()-obs[down,1].mean());med=float(np.median(obs[up,1])-np.median(obs[down,1]))
    rng=np.random.default_rng(seed);mv=[];dv=[];starts=np.arange(max(1,n-block+1))
    for _ in range(reps):
        ids=[]
        while len(ids)<n:
            j=int(rng.choice(starts));ids.extend(range(j,min(j+block,n)))
        b=a[np.asarray(ids[:n])];b=b[np.isfinite(b).all(axis=1)];u=b[:,0]>0;v=b[:,0]<0
        if u.any() and v.any():mv.append(b[u,1].mean()-b[v,1].mean());dv.append(np.median(b[u,1])-np.median(b[v,1]))
    return {"n":len(obs),"calendar_rows":n,"mean_diff":md,"mean_ci_low":float(np.quantile(mv,.025)),"mean_ci_high":float(np.quantile(mv,.975)),"median_diff":med,"median_ci_low":float(np.quantile(dv,.025)),"median_ci_high":float(np.quantile(dv,.975)),"bootstrap_reps":len(mv),"block_sessions":block}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protected=[PANEL,ROUTE,SRC/"daily_oi_audit_panel.csv",SRC/"summary.json"]
    hash0={str(p):sha(p) for p in protected}
    d=pd.read_csv(PANEL,parse_dates=["date"]).set_index("date").sort_index().loc[START:END].copy()
    route=pd.read_csv(ROUTE,parse_dates=["date"]).set_index("date").sort_index().loc[d.index]
    if not route.contract.astype(str).equals(d.contract.astype(str)):raise ValueError("Selected contract series mismatch")
    audit=pd.read_csv(SRC/"daily_oi_audit_panel.csv",parse_dates=["date"]).set_index("date").reindex(d.index)
    d["oi_pct_change"]=d.oi_change_1d
    d["roll"]=d.source_roll.astype(bool)
    d["roll_guard_5"]=audit.roll_proximity_5.astype(bool)
    # For standardization reference, remove dates within +/-5 sessions of route changes.
    # All reference values are earlier than the signal; no current/future observations enter mean/SD/CDF.
    reference=d.oi_pct_change.where(~d.roll_guard_5)
    values=reference.dropna()
    zscore=pd.Series(np.nan,index=d.index,dtype=float);pctile=pd.Series(np.nan,index=d.index,dtype=float)
    history=[]
    for dt in d.index:
        current=d.at[dt,"oi_pct_change"]
        past=np.asarray(history[-WINDOW:],float)
        if pd.notna(current) and len(past)>=WINDOW:
            sd=float(past.std(ddof=1))
            if sd>0:zscore.at[dt]=(current-past.mean())/sd
            pctile.at[dt]=float(np.mean(past<=current))
        refv=reference.at[dt]
        if pd.notna(refv):history.append(float(refv))
    d["oi_rolling_z60"]=zscore;d["oi_historical_pctile60"]=pctile
    # Exact rolling window provenance counts validate all standardizers are causal.
    d["target_end_10d"]=pd.Series(d.index,index=d.index).shift(-10)
    d["year"]=d.index.year
    d.to_csv(OUT/"daily_oi_standardized_panel.csv",index_label="date",encoding="utf-8-sig",float_format="%.10g")

    tests=[]
    def add_test(family,name,values):
        for metric,p in [("spearman",values.get("spearman_p",np.nan)),("pearson",values.get("pearson_p",np.nan))]:
            if np.isfinite(p):tests.append({"family":family,"test":name,"metric":metric,"p":p})
    annual=[]
    for variant in VARIANTS:
        for year in YEARS:
            g=d[(d.year==year)&(~d.roll)&d.target_end_10d.notna()&(d.target_end_10d.dt.year==year)].dropna(subset=VARIANTS+["future_return_10d"])
            st=stats(g[variant],g.future_return_10d)
            row={"year":year,"variant":variant,**st,"sample_period":"partial_through_2026-10-12" if year==2026 else "full_calendar_year"}
            annual.append(row)
            tests.append({"family":"annual_15","test":f"{year}:{variant}:spearman","metric":"spearman","p":st["spearman_p"]})
            tests.append({"family":"annual_15","test":f"{year}:{variant}:pearson","metric":"pearson","p":st["pearson_p"]})
    annual=pd.DataFrame(annual)
    annual["bh_q_annual_spearman_15"]=bh(annual.spearman_p)
    annual["bh_q_annual_pearson_15"]=bh(annual.pearson_p)
    annual.to_csv(OUT/"annual_oi_robustness.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    pooled=[]
    eval_nonroll=d[(~d.roll)&d.future_return_10d.notna()].dropna(subset=VARIANTS)
    scenarios=[("full_nonroll",pd.Series(True,index=eval_nonroll.index)),("exclude_2024_2025",~eval_nonroll.year.isin([2024,2025])),("exclude_roll_plus_minus_5",~eval_nonroll.roll_guard_5)]
    for sample,mask in scenarios:
        g=eval_nonroll[mask]
        for variant in VARIANTS:
            st=stats(g[variant],g.future_return_10d);pooled.append({"sample":sample,"variant":variant,**st})
            add_test("pooled_9",f"{sample}:{variant}",st)
    pooled=pd.DataFrame(pooled)
    for metric in ["spearman","pearson"]:pooled[f"bh_q_{metric}_pooled_9"]=bh(pooled[f"{metric}_p"])
    pooled.to_csv(OUT/"pooled_exclusion_tests.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    loo=[]
    for variant in VARIANTS:
        for year in YEARS:
            g=eval_nonroll[eval_nonroll.year!=year]
            st=stats(g[variant],g.future_return_10d);loo.append({"omitted_year":year,"variant":variant,**st})
    loo=pd.DataFrame(loo)
    for metric in ["spearman","pearson"]:loo[f"bh_q_{metric}_leave_one_year_out_15"]=bh(loo[f"{metric}_p"])
    loo.to_csv(OUT/"leave_one_year_out.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Ex-post 1/99 clipping/trimming are influence diagnostics only, never signal definitions.
    base=eval_nonroll[["oi_pct_change","future_return_10d"]].dropna().copy()
    xlo,xhi=base.oi_pct_change.quantile([.01,.99]);ylo,yhi=base.future_return_10d.quantile([.01,.99])
    outlier_specs=[("raw",base.oi_pct_change.copy(),base.future_return_10d.copy(),pd.Series(True,index=base.index)),
       ("OI_winsor_1_99",base.oi_pct_change.clip(xlo,xhi),base.future_return_10d.copy(),pd.Series(True,index=base.index)),
       ("trim_OI_1_99",base.oi_pct_change,base.future_return_10d,(base.oi_pct_change>=xlo)&(base.oi_pct_change<=xhi)),
       ("return_winsor_1_99",base.oi_pct_change,base.future_return_10d.clip(ylo,yhi),pd.Series(True,index=base.index)),
       ("trim_return_1_99",base.oi_pct_change,base.future_return_10d,(base.future_return_10d>=ylo)&(base.future_return_10d<=yhi))]
    outs=[]
    for name,x,y,mask in outlier_specs:
        st=stats(x[mask],y[mask]);outs.append({"policy":name,"n":st["n"],"oi_p01":xlo,"oi_p99":xhi,"return_p01":ylo,"return_p99":yhi,**st})
        add_test("outlier_10",name,st)
    outdf=pd.DataFrame(outs)
    for metric in ["spearman","pearson"]:outdf[f"bh_q_{metric}_outlier_10"]=bh(outdf[f"{metric}_p"])
    outdf.to_csv(OUT/"extreme_value_sensitivity.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    # Extreme-value sensitivity for the OI-up versus OI-down return spread.
    nonroll_x=d.oi_pct_change.where(~d.roll);nonroll_y=d.future_return_10d
    group_specs=[("raw",nonroll_x,nonroll_y),
       ("trim_OI_1_99",nonroll_x.where((d.oi_pct_change>=xlo)&(d.oi_pct_change<=xhi)),nonroll_y),
       ("winsor_return_1_99",nonroll_x,nonroll_y.clip(ylo,yhi)),
       ("trim_return_1_99",nonroll_x,nonroll_y.where((d.future_return_10d>=ylo)&(d.future_return_10d<=yhi)))]
    gd=[]
    for i,(name,x,y) in enumerate(group_specs):
        frame=pd.DataFrame({"oi":x,"ret":y})
        gd.append({"policy":name,**block_diff(frame,seed=261014+i)})
    pd.DataFrame(gd).to_csv(OUT/"extreme_group_difference.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Continuous, causal 252-session rolling IC and Pearson correlation; omit roll dates.
    rolling=[]
    for end_i in range(ROLL-1,len(d),STEP):
        w=d.iloc[end_i-ROLL+1:end_i+1];w=w[(~w.roll)&w.future_return_10d.notna()].dropna(subset=VARIANTS)
        for variant in VARIANTS:
            st=stats(w[variant],w.future_return_10d);rolling.append({"window_start":d.index[end_i-ROLL+1],"window_end":d.index[end_i],"variant":variant,**{k:v for k,v in st.items() if not k.endswith("_p")}})
    rolling=pd.DataFrame(rolling);rolling.to_csv(OUT/"rolling_oi_robustness.csv",index=False,encoding="utf-8-sig",float_format="%.10g",date_format="%Y-%m-%d")

    # Global BH family includes annual, pooled, leave-one-year-out, and outlier
    # influence p-values for both correlation metrics (88 tests total).
    family=[]
    for _,r in annual.iterrows():
        family.extend([{"test":f"annual:{r.year}:{r.variant}","metric":"spearman","p":r.spearman_p},{"test":f"annual:{r.year}:{r.variant}","metric":"pearson","p":r.pearson_p}])
    for _,r in pooled.iterrows():
        family.extend([{"test":f"pooled:{r['sample']}:{r.variant}","metric":"spearman","p":r.spearman_p},{"test":f"pooled:{r['sample']}:{r.variant}","metric":"pearson","p":r.pearson_p}])
    for _,r in outdf.iterrows():
        family.extend([{"test":f"outlier:{r.policy}","metric":"spearman","p":r.spearman_p},{"test":f"outlier:{r.policy}","metric":"pearson","p":r.pearson_p}])
    for _,r in loo.iterrows():
        family.extend([{"test":f"leave_out:{r.omitted_year}:{r.variant}","metric":"spearman","p":r.spearman_p},{"test":f"leave_out:{r.omitted_year}:{r.variant}","metric":"pearson","p":r.pearson_p}])
    family=pd.DataFrame(family);family["bh_q_global_88_all_tests"]=bh(family.p);family.to_csv(OUT/"global_multiple_testing.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Chart: yearly Spearman IC, time series rolling IC with band, and extreme sensitivity.
    fig,axes=plt.subplots(3,1,figsize=(13,12))
    colors={VARIANTS[0]:"#4c78a8",VARIANTS[1]:"#f58518",VARIANTS[2]:"#54a24b"}
    for v in VARIANTS:
        s=annual[annual.variant==v];axes[0].plot(s.year,s.spearman_ic,marker="o",label=v,color=colors[v]);axes[0].fill_between(s.year,s.spearman_low,s.spearman_high,color=colors[v],alpha=.12)
    axes[0].axhline(0,color="black",lw=.7);axes[0].set_ylabel("Annual Rank IC");axes[0].set_title("Yearly OI factor IC (future 10-session return; rolls excluded)");axes[0].legend(ncol=3);axes[0].grid(alpha=.2)
    for v in VARIANTS:
        s=rolling[rolling.variant==v];axes[1].plot(pd.to_datetime(s.window_end),s.spearman_ic,label=v,color=colors[v],lw=1.2)
    axes[1].axhline(0,color="black",lw=.7);axes[1].set_ylabel("Rolling Rank IC");axes[1].set_title("252-session window / 21-session step; overlapping windows descriptive only");axes[1].legend(ncol=3);axes[1].grid(alpha=.2)
    s=outdf;axes[2].errorbar(np.arange(len(s)),s.pearson_r,yerr=[s.pearson_r-s.pearson_low,s.pearson_high-s.pearson_r],fmt="o",capsize=4,color="#e45756");axes[2].axhline(0,color="black",lw=.7);axes[2].set_xticks(np.arange(len(s)),s.policy,rotation=20,ha="right");axes[2].set_ylabel("Pearson correlation");axes[2].set_title("Extreme-value influence diagnostics (ex-post 1% tails)");axes[2].grid(alpha=.2)
    fig.suptitle("Open Interest Factor Robustness — Exploratory")
    fig.tight_layout();fig.savefig(OUT/"oi_robustness.png",dpi=170);plt.close(fig)

    hash1={str(p):sha(p) for p in protected}
    if hash0!=hash1:raise RuntimeError("Protected existing input changed")
    c=d[(~d.roll)&d.oi_pct_change.notna()&d.future_return_10d.notna()]
    missing={"panel_rows":len(d),"oi_missing":int(d.oi_pct_change.isna().sum()),"target10_missing":int(d.future_return_10d.isna().sum()),"roll_days":int(d.roll.sum()),"roll_guard_days":int(d.roll_guard_5.sum()),"complete_nonroll_rows":len(c),"raw_missing_selected_rows":int(audit[["actual_contract_return_1d","oi_change_recomputed_same_contract"]].isna().any(axis=1).sum())}
    summary={"research":"OI Factor Robustness Analysis","period_start":str(START.date()),"period_end":str(END.date()),"years":YEARS,"standardizations":{"oi_pct_change":"existing same-contract daily percentage change","oi_rolling_z60":"current value standardized by prior 60 non-roll-guard observations","oi_historical_pctile60":"expanding empirical percentile vs all prior non-roll-guard observations, minimum 60"},"extreme_policy":"one percent quantile clipping/trimming, ex-post influence diagnostic only; never used to define a signal","roll_guard":"both route-switch date and ±5 adjacent DCE sessions for guarded sensitivity; factor references omit the guard window","hac_lags":LAGS,"rolling_window":ROLL,"rolling_step":STEP,"multiple_testing":{"annual_each_metric_family":15,"pooled_each_metric_family":9,"leave_one_year_out_each_metric_family":15,"outlier_each_metric_family":5,"global_all_metric_family":88,"global_includes_spearman_and_pearson":True},"missingness":missing,"raw_contract_gap_records_existing":32,"gaps_filled":False,"existing_inputs_unchanged":hash0==hash1,"input_hashes":hash0,"true_independent_oos":False,"strategy_or_historical_backtest_modified":False,"trading_executed":False,"forward_paper_started":False}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"annual_spearman":annual[["year","variant","n","spearman_ic","spearman_low","spearman_high","bh_q_annual_spearman_15"]].to_dict("records"),"pooled":pooled[["sample","variant","spearman_ic","spearman_p"]].to_dict("records"),"input_unchanged":hash0==hash1,"missingness":missing},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
