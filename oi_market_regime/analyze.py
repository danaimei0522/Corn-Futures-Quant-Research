"""Causal, descriptive regime-conditional OI factor analysis."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).parent
PANEL=ROOT/"alternative_alpha/daily_factor_panel.csv"
REGIME=ROOT/"five_year_trade_research/daily_asof_features.csv"
AUDIT=ROOT/"oi_deep_dive/daily_oi_audit_panel.csv"
ROUTE=ROOT/"five_year_backtest/causal_signal_and_contracts.csv"
START,END=pd.Timestamp("2022-01-01"),pd.Timestamp("2026-10-12")
TREND=["uptrend","downtrend","sideways_transition"]
VOLS=["low","medium","high"]
LAGS=20;VOL_LOOKBACK=60;VOL_MIN=40;Z=1.959963984540054

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normp(t):return math.erfc(abs(t)/math.sqrt(2)) if np.isfinite(t) else np.nan
def bh(pv):
    p=np.asarray(pv,float);q=np.full(len(p),np.nan);ids=np.flatnonzero(np.isfinite(p))
    if not len(ids):return q
    order=ids[np.argsort(p[ids])];raw=p[order]*len(order)/np.arange(1,len(order)+1);q[order]=np.minimum(np.minimum.accumulate(raw[::-1])[::-1],1);return q
def gammaincc(a,x):
    """Regularized upper incomplete gamma, for chi-square Wald p-values."""
    if x<=0:return 1.0
    gln=math.lgamma(a)
    if x<a+1:
        ap=a;term=1/a;total=term
        for _ in range(10000):
            ap+=1;term*=x/ap;total+=term
            if abs(term)<abs(total)*1e-14:break
        p=total*math.exp(-x+a*math.log(x)-gln)
        return max(0.0,min(1.0,1-p))
    b=x+1-a;c=1e300;d=1/max(b,1e-300);h=d
    for i in range(1,10000):
        an=-i*(i-a);b+=2;d=an*d+b
        if abs(d)<1e-300:d=1e-300
        c=b+an/c
        if abs(c)<1e-300:c=1e-300
        d=1/d;delta=d*c;h*=delta
        if abs(delta-1)<1e-14:break
    return max(0.0,min(1.0,math.exp(-x+a*math.log(x)-gln)*h))
def chi2_sf(stat,df):return gammaincc(df/2,max(0,stat)/2)

def hac_fit_gaps(y,X,lags=LAGS):
    """OLS/Newey-West with missing calendar rows retained as zero score rows."""
    y=np.asarray(y,float);X=np.asarray(X,float)
    if X.ndim==1:X=X[:,None]
    valid=np.isfinite(y)&np.isfinite(X).all(axis=1);n=len(y);m=int(valid.sum())
    if m<=X.shape[1]+3:return None
    Xv=np.column_stack([np.ones(m),X[valid]]);yv=y[valid];k=Xv.shape[1]
    bread=np.linalg.pinv(Xv.T@Xv);beta=bread@Xv.T@yv;resid=yv-Xv@beta
    g=np.zeros((n,k));g[valid]=Xv*resid[:,None];meat=g.T@g
    for lag in range(1,min(lags,n-1)+1):
        w=1-lag/(min(lags,n-1)+1);gamma=g[lag:].T@g[:-lag];meat+=w*(gamma+gamma.T)
    cov=bread@meat@bread;se=np.sqrt(np.maximum(np.diag(cov),0));t=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    return {"n":m,"beta":beta,"cov":cov,"se":se,"t":t,"p":np.array([normp(v) for v in t]),"lo":beta-Z*se,"hi":beta+Z*se,"valid":valid}
def rank_ic_gaps(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);valid=np.isfinite(x)&np.isfinite(y);n=int(valid.sum())
    if n<5:return {"n":n,"ic":np.nan,"se":np.nan,"low":np.nan,"high":np.nan,"p":np.nan,"n_eff":np.nan}
    xr=pd.Series(x[valid]).rank(method="average").to_numpy(float);yr=pd.Series(y[valid]).rank(method="average").to_numpy(float)
    xr=(xr-xr.mean())/xr.std();yr=(yr-yr.mean())/yr.std()
    X=np.full((len(x),1),np.nan);Y=np.full(len(y),np.nan);X[valid,0]=xr;Y[valid]=yr
    fit=hac_fit_gaps(Y,X,LAGS);b=float(fit["beta"][1]);se=float(fit["se"][1]);ne=float(np.clip((1-b*b)/max(se*se,1e-15)+2,1,n))
    return {"n":n,"ic":b,"se":se,"low":max(-1,b-Z*se),"high":min(1,b+Z*se),"p":normp(b/se if se else np.nan),"n_eff_hac_approx":ne}
def wald(fit,idx):
    b=fit["beta"][idx];v=fit["cov"][np.ix_(idx,idx)];stat=float(b@np.linalg.pinv(v)@b)
    return {"wald_chi2":stat,"df":len(idx),"p":chi2_sf(stat,len(idx))}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[PANEL,REGIME,AUDIT,ROUTE];h0={str(p):sha(p) for p in sources}
    panel=pd.read_csv(PANEL,parse_dates=["date"]).set_index("date").sort_index()
    regimes=pd.read_csv(REGIME,parse_dates=["date","asof"]).set_index("date").sort_index()
    audit=pd.read_csv(AUDIT,parse_dates=["date"]).set_index("date").sort_index()
    route=pd.read_csv(ROUTE,parse_dates=["date"]).set_index("date").sort_index()
    if panel.index.has_duplicates or regimes.index.has_duplicates:raise ValueError("Duplicate input dates")
    d=panel.join(regimes[["asof","adx","slope20","slope60","rv20","state","vol_state"]],how="left",rsuffix="_regime")
    d=d.join(audit[["roll","roll_proximity_5","oi_change_recomputed_same_contract"]],how="left")
    route_contract=route.contract.astype(str).reindex(d.index);panel_contract=d.contract.astype(str)
    if not route_contract.equals(panel_contract):raise ValueError("Audited route and factor panel contract mismatch")
    # Trend categories are inherited from the prior as-of regime study; merge only its
    # range and transition labels into a broad sideways/transition class.
    d["trend_regime"]=d.state.map({"uptrend":"uptrend","downtrend":"downtrend","range":"sideways_transition","transition":"sideways_transition"})
    # Causal volatility tertiles: compare current as-of RV20 to distribution of prior
    # 60 daily as-of RV20 values, with at least 40 valid prior values.
    hist=d.rv20.shift(1)
    lower=hist.rolling(VOL_LOOKBACK,min_periods=VOL_MIN).quantile(1/3)
    upper=hist.rolling(VOL_LOOKBACK,min_periods=VOL_MIN).quantile(2/3)
    d["vol_tertile"]=np.select([d.rv20<=lower,d.rv20<=upper], ["low","medium"],default="high")
    d.loc[d.rv20.isna()|lower.isna()|upper.isna(),"vol_tertile"]="unknown"
    d["signal_period"]=np.where(d.index.year.isin([2024,2025]),"2024_2025","other_years")
    d["year"]=d.index.year
    d["target_end_10d"]=pd.Series(d.index,index=d.index).shift(-10)
    # Restrict evaluation to 2022-10/2026 and retain all rows in the time index for HAC lags.
    d=d.loc[START:END].copy()
    d["roll"]=d.roll.fillna(False).astype(bool);d["roll_proximity_5"]=d.roll_proximity_5.fillna(False).astype(bool)
    d["oi_factor"]=d.oi_change_recomputed_same_contract
    if not np.allclose(d.oi_factor.to_numpy(float),d.oi_change_1d.to_numpy(float),equal_nan=True,atol=1e-10,rtol=1e-8):raise ValueError("OI factor audit mismatch")
    d.to_csv(OUT/"daily_regime_panel.csv",index_label="date",encoding="utf-8-sig",float_format="%.10g")

    # State-conditioned IC for three trends, three volatility tertiles, 9 intersections,
    # and each one-way regime separately during/away from 2024-25.
    ic_rows=[]
    valid_target=d.future_return_10d.notna()
    for dim,col,cats in [("trend","trend_regime",TREND),("volatility","vol_tertile",VOLS),("trend_x_volatility","joint_state",None)]:
        if dim=="trend_x_volatility":d["joint_state"]=d.trend_regime.astype(str)+"|"+d.vol_tertile.astype(str);cats=[f"{t}|{v}" for t in TREND for v in VOLS]
        for cat in cats:
            mask=d[col].astype(str).eq(cat)&(~d.roll)&valid_target
            st=rank_ic_gaps(d.oi_factor.where(mask),d.future_return_10d.where(mask))
            ic_rows.append({"dimension":dim,"state":cat,"period":"all","n":st["n"],"rank_ic":st["ic"],"hac_se":st["se"],"ci95_low":st["low"],"ci95_high":st["high"],"hac_p_normal":st["p"],"n_eff_hac_approx":st.get("n_eff_hac_approx",np.nan)})
            if dim in ["trend","volatility"]:
                for period in ["2024_2025","other_years"]:
                    m=mask&d.signal_period.eq(period)
                    z=rank_ic_gaps(d.oi_factor.where(m),d.future_return_10d.where(m))
                    ic_rows.append({"dimension":dim,"state":cat,"period":period,"n":z["n"],"rank_ic":z["ic"],"hac_se":z["se"],"ci95_low":z["low"],"ci95_high":z["high"],"hac_p_normal":z["p"],"n_eff_hac_approx":z.get("n_eff_hac_approx",np.nan)})
    icdf=pd.DataFrame(ic_rows);icdf["bh_q_27_state_ic_tests"]=bh(icdf.hac_p_normal);icdf.to_csv(OUT/"state_conditional_oi_ic.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Return distributions per one-way and joint states; HAC regression gives time-order
    # robust state means and CI without compressing the daily calendar.
    dist_rows=[]
    sets=[("trend","trend_regime",TREND),("volatility","vol_tertile",VOLS),("joint","joint_state",[f"{t}|{v}" for t in TREND for v in VOLS])]
    for dim,col,cats in sets:
        if col=="joint_state":d[col]=d.trend_regime.astype(str)+"|"+d.vol_tertile.astype(str)
        groupcol=d[col].astype(str)
        usable=d.future_return_10d.notna()&(~d.roll)&groupcol.isin(cats)
        cats_use=[c for c in cats if (usable&groupcol.eq(c)).sum()>0]
        if not cats_use:continue
        baseline=cats_use[0];dummycats=cats_use[1:]
        X=np.column_stack([(groupcol.eq(c)&usable).astype(float).to_numpy() for c in dummycats]) if dummycats else np.zeros((len(d),0))
        y=d.future_return_10d.where(usable).to_numpy(float)
        fit=hac_fit_gaps(y,X,LAGS)
        for cat in cats_use:
            ids=np.flatnonzero((usable&groupcol.eq(cat)).to_numpy());yy=d.future_return_10d.iloc[ids].to_numpy(float)
            v=np.zeros(1+len(dummycats));v[0]=1
            if cat!=baseline:v[1+dummycats.index(cat)]=1
            mean=float(v@fit["beta"]);se=float(np.sqrt(max(0,v@fit["cov"]@v)))
            dist_rows.append({"dimension":dim,"state":cat,"n":len(ids),"mean_future10":mean,"hac_mean_ci_low":mean-Z*se,"hac_mean_ci_high":mean+Z*se,"median_future10":float(np.median(yy)),"positive_return_share":float(np.mean(yy>0)),"roll_days":int(d.roll.iloc[ids].sum())})
    dist=pd.DataFrame(dist_rows);dist.to_csv(OUT/"state_future_return_distribution.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Regime composition in 2024-25 versus remaining requested years (2022, 2023, 2026).
    comp_rows=[]
    base=d[d.index.year.isin([2022,2023,2024,2025,2026])&(~d.roll)]
    for dim,col,cats in [("trend","trend_regime",TREND),("volatility","vol_tertile",VOLS)]:
        known=base[base[col].astype(str).isin(cats)]
        for cat in cats:
            outcome=known[col].astype(str).eq(cat).astype(float).to_numpy()
            period=known.signal_period.eq("2024_2025").astype(float).to_numpy()
            fit=hac_fit_gaps(outcome,period,LAGS)
            if fit is None:continue
            comp_rows.append({"dimension":dim,"state":cat,"n":fit["n"],"share_other_years":float(known.loc[known.signal_period.eq("other_years"),col].astype(str).eq(cat).mean()),"share_2024_2025":float(known.loc[known.signal_period.eq("2024_2025"),col].astype(str).eq(cat).mean()),"difference_2425_minus_other":float(fit["beta"][1]),"ci95_low":float(fit["lo"][1]),"ci95_high":float(fit["hi"][1]),"p_hac_normal":float(fit["p"][1])})
    comp=pd.DataFrame(comp_rows);comp["bh_q_6_prevalence_tests"]=bh(comp.p_hac_normal);comp.to_csv(OUT/"regime_prevalence_2024_25.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Unified rank-on-rank OI x regime model. Main effects for trend, volatility and
    # period; test OI interactions with each set using Newey-West HAC covariance.
    model_ok=d.future_return_10d.notna()&d.oi_factor.notna()&d.trend_regime.isin(TREND)&d.vol_tertile.isin(VOLS)&(~d.roll)
    oi_rank=np.full(len(d),np.nan);y_rank=np.full(len(d),np.nan)
    oi_rank[model_ok]=pd.Series(d.oi_factor[model_ok].to_numpy()).rank(method="average").to_numpy();y_rank[model_ok]=pd.Series(d.future_return_10d[model_ok].to_numpy()).rank(method="average").to_numpy()
    for arr in [oi_rank,y_rank]:arr[model_ok]=(arr[model_ok]-arr[model_ok].mean())/arr[model_ok].std()
    trbase="sideways_transition";volbase="medium";periodbase="other_years"
    columns=["oi_rank"]
    Xcols=[oi_rank.copy()]
    for c in ["uptrend","downtrend"]:columns.append(f"trend_{c}");Xcols.append((d.trend_regime.eq(c)&model_ok).astype(float).to_numpy())
    for c in ["low","high"]:columns.append(f"vol_{c}");Xcols.append((d.vol_tertile.eq(c)&model_ok).astype(float).to_numpy())
    columns.append("period_2024_2025");Xcols.append((d.signal_period.eq("2024_2025")&model_ok).astype(float).to_numpy())
    int_names=[]
    for c in ["uptrend","downtrend"]:
        name=f"oi_x_trend_{c}";columns.append(name);Xcols.append(oi_rank*(d.trend_regime.eq(c)&model_ok).to_numpy(float));int_names.append(name)
    for c in ["low","high"]:
        name=f"oi_x_vol_{c}";columns.append(name);Xcols.append(oi_rank*(d.vol_tertile.eq(c)&model_ok).to_numpy(float));int_names.append(name)
    name="oi_x_period_2024_2025";columns.append(name);Xcols.append(oi_rank*(d.signal_period.eq("2024_2025")&model_ok).to_numpy(float));int_names.append(name)
    X=np.column_stack(Xcols);fit=hac_fit_gaps(y_rank,X,LAGS)
    term_rows=[]
    for j,name in enumerate(columns,1):
        term_rows.append({"term":name,"coefficient":fit["beta"][j],"se_hac":fit["se"][j],"ci95_low":fit["lo"][j],"ci95_high":fit["hi"][j],"p_hac_normal":fit["p"][j]})
    terms=pd.DataFrame(term_rows);terms["bh_q_11_term_tests"]=bh(terms.p_hac_normal);terms.to_csv(OUT/"interaction_model_terms.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    idx={n:i+1 for i,n in enumerate(columns)}
    joint=[]
    for label,names in [("OI_x_trend_joint",["oi_x_trend_uptrend","oi_x_trend_downtrend"]),("OI_x_volatility_joint",["oi_x_vol_low","oi_x_vol_high"]),("OI_x_2024_2025_period",["oi_x_period_2024_2025"]),("all_OI_interactions_joint",int_names)]:
        w=wald(fit,[idx[n] for n in names]);joint.append({"test":label,"terms":";".join(names),"n":fit["n"],**w})
    joint=pd.DataFrame(joint);joint["bh_q_4_joint_tests"]=bh(joint.p);joint.to_csv(OUT/"interaction_joint_tests.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # One global BH family across every state-IC, prevalence and interaction test.
    alltests=[]
    alltests += [{"family":"state_ic_27","test":f"{r.dimension}:{r.state}:{r.period}","p":r.hac_p_normal} for r in icdf.itertuples()]
    alltests += [{"family":"prevalence_6","test":f"{r.dimension}:{r.state}","p":r.p_hac_normal} for r in comp.itertuples()]
    alltests += [{"family":"interaction_terms_11","test":r.term,"p":r.p_hac_normal} for r in terms.itertuples()]
    alltests += [{"family":"interaction_joint_4","test":r.test,"p":r.p} for r in joint.itertuples()]
    alltests=pd.DataFrame(alltests);alltests["bh_q_global_48_tests"]=bh(alltests.p);alltests.to_csv(OUT/"global_multiple_testing.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Annual raw OI IC to connect the regime study to the robustness baseline.
    annual=[]
    for year in range(2022,2027):
        m=(d.year==year)&(~d.roll)&d.future_return_10d.notna()
        st=rank_ic_gaps(d.oi_factor.where(m),d.future_return_10d.where(m));annual.append({"year":year,**st,"period_description":"partial_to_2026-10-12" if year==2026 else "calendar_year"})
    ann=pd.DataFrame(annual);ann["bh_q_5_annual_baseline"]=bh(ann.p);ann.to_csv(OUT/"annual_baseline_oi_ic.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Visuals: joint IC heatmaps; period-specific ICs; regime prevalence comparison.
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    jointic=icdf[(icdf.dimension=="trend_x_volatility")&(icdf.period=="all")].copy();jointic[["trend","vol"]]=jointic.state.str.split("|",expand=True)
    piv=jointic.pivot(index="trend",columns="vol",values="rank_ic").reindex(index=TREND,columns=VOLS)
    im=axes[0,0].imshow(piv.to_numpy(float),cmap="RdYlBu_r",vmin=-.5,vmax=.5,aspect="auto");axes[0,0].set_xticks(range(3),VOLS);axes[0,0].set_yticks(range(3),TREND)
    for i,t in enumerate(TREND):
        for j,v in enumerate(VOLS):
            row=jointic[(jointic.trend==t)&(jointic.vol==v)].iloc[0];axes[0,0].text(j,i,f"{row.rank_ic:+.2f}\nn={int(row.n)}",ha="center",va="center",fontsize=9)
    axes[0,0].set_title("OI future-10d Rank IC by trend × volatility");fig.colorbar(im,ax=axes[0,0],shrink=.8)
    for period,color in [("2024_2025","#e45756"),("other_years","#4c78a8")]:
        sub=icdf[(icdf.dimension=="trend")&(icdf.period==period)].set_index("state").reindex(TREND)
        axes[0,1].errorbar(np.arange(3)+(0.08 if period=="2024_2025" else -.08),sub.rank_ic,yerr=[sub.rank_ic-sub.ci95_low,sub.ci95_high-sub.rank_ic],fmt="o",capsize=3,label=period,color=color)
    axes[0,1].axhline(0,color="black",lw=.7);axes[0,1].set_xticks(range(3),TREND,rotation=10);axes[0,1].set_ylabel("Rank IC");axes[0,1].set_title("Trend-conditional IC by year block");axes[0,1].legend();axes[0,1].grid(alpha=.2)
    for period,color in [("2024_2025","#e45756"),("other_years","#4c78a8")]:
        sub=icdf[(icdf.dimension=="volatility")&(icdf.period==period)].set_index("state").reindex(VOLS)
        axes[1,0].errorbar(np.arange(3)+(0.08 if period=="2024_2025" else -.08),sub.rank_ic,yerr=[sub.rank_ic-sub.ci95_low,sub.ci95_high-sub.rank_ic],fmt="o",capsize=3,label=period,color=color)
    axes[1,0].axhline(0,color="black",lw=.7);axes[1,0].set_xticks(range(3),VOLS);axes[1,0].set_ylabel("Rank IC");axes[1,0].set_title("Volatility-conditional IC by year block");axes[1,0].legend();axes[1,0].grid(alpha=.2)
    cm=comp.pivot(index="state",columns="dimension",values="difference_2425_minus_other").reindex(index=TREND+VOLS)
    cm.plot(kind="bar",ax=axes[1,1],color=["#54a24b","#f58518"]);axes[1,1].axhline(0,color="black",lw=.7);axes[1,1].set_ylabel("State share change (fraction)");axes[1,1].set_title("2024–25 regime composition vs other years");axes[1,1].tick_params(axis="x",rotation=25);axes[1,1].grid(axis="y",alpha=.2)
    fig.suptitle("Market-Regime Conditional OI Factor Analysis — Exploratory")
    fig.tight_layout();fig.savefig(OUT/"market_regime_conditional_oi.png",dpi=170);plt.close(fig)

    h1={str(p):sha(p) for p in sources}
    if h0!=h1:raise RuntimeError("Protected source files changed")
    unknown_trend=int(d.trend_regime.isna().sum());unknown_vol=int(d.vol_tertile.eq("unknown").sum())
    summary={"research":"Market Regime Conditional Factor Analysis","evaluation_start":str(START.date()),"evaluation_end":str(END.date()),"trend_definition":"existing prior as-of states; range+transition merged as sideways_transition","volatility_definition":f"prior {VOL_LOOKBACK} as-of RV20 values; min {VOL_MIN}; rolling one-third/two-third quantiles; shift one session before thresholds","oi_factor":"same-contract daily OI percentage change","target":"causal additive real-contract index future 10-session return","hac_lags":LAGS,"roll_days":int(d.roll.sum()),"roll_guard_days":int(d.roll_proximity_5.sum()),"unknown_trend_rows":unknown_trend,"unknown_volatility_rows":unknown_vol,"matched_model_n":int(fit["n"]),"global_multiple_test_count":len(alltests),"global_q_lt_05":int((alltests.bh_q_global_48_tests<.05).sum()),"inputs_unchanged":h0==h1,"input_hashes":h0,"true_independent_oos":False,"strategy_version_unchanged":"corn-adx20-atr3-forward-v1.0.0","strategy_or_data_modified":False,"trading_executed":False,"forward_paper_started":False}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"trend_ic":icdf[(icdf.dimension=="trend")][["state","period","n","rank_ic","ci95_low","ci95_high","bh_q_27_state_ic_tests"]].to_dict("records"),"vol_ic":icdf[(icdf.dimension=="volatility")][["state","period","n","rank_ic","ci95_low","ci95_high","bh_q_27_state_ic_tests"]].to_dict("records"),"interactions":joint.to_dict("records"),"global_q05":summary["global_q_lt_05"],"inputs_unchanged":summary["inputs_unchanged"]},ensure_ascii=False,indent=2))

if __name__=="__main__":main()
