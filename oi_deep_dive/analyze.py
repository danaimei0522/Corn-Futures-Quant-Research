"""Exploratory open-interest factor deep dive; reads existing data only."""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).parent
PANEL_PATH = ROOT / "alternative_alpha/daily_factor_panel.csv"
ROUTE_PATH = ROOT / "five_year_backtest/causal_signal_and_contracts.csv"
GAPS_PATH = ROOT / "five_year_backtest/missing_dates.csv"
CONTRACT_FILES = sorted((ROOT / "long_history_audit").glob("C????_daily.csv"))
START, END = pd.Timestamp("2021-10-12"), pd.Timestamp("2026-10-12")
HAC_LAGS, ROLL_WINDOW, ROLL_STEP = 20, 252, 21
BOOTSTRAPS, BLOCK = 2000, 20
SEED = 261014
Z95 = 1.959963984540054

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def bh(pvals):
    p = np.asarray(pvals, dtype=float); q = np.full(len(p), np.nan)
    ids = np.flatnonzero(np.isfinite(p))
    if not len(ids): return q
    order = ids[np.argsort(p[ids])]
    raw = p[order] * len(order) / np.arange(1, len(order)+1)
    q[order] = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1.0)
    return q

def normal_p(t): return math.erfc(abs(t)/math.sqrt(2)) if np.isfinite(t) else np.nan

def hac_ols(y, X, lags=HAC_LAGS):
    """OLS with Newey-West Bartlett covariance; X has no intercept column."""
    y = np.asarray(y, float); X = np.asarray(X, float)
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[keep], X[keep]; n = len(y)
    if n <= X.shape[1] + 3: return None
    X = np.column_stack([np.ones(n), X]); k = X.shape[1]
    bread = np.linalg.pinv(X.T @ X); beta = bread @ X.T @ y
    resid = y - X @ beta; g = X * resid[:, None]; meat = g.T @ g
    L = min(lags, n-1)
    for lag in range(1, L+1):
        w = 1-lag/(L+1); gamma = g[lag:].T @ g[:-lag]
        meat += w*(gamma+gamma.T)
    cov = bread @ meat @ bread
    se = np.sqrt(np.maximum(np.diag(cov),0)); t = np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    lo, hi = beta-Z95*se, beta+Z95*se
    return {"n":n,"beta":beta,"se":se,"t":t,"p":np.array([normal_p(v) for v in t]),"lo":lo,"hi":hi,"r2":1-float(np.sum(resid**2))/max(float(np.sum((y-y.mean())**2)),1e-20),"cov":cov}

def rank_ic(x,y):
    z=pd.DataFrame({"x":x,"y":y}).dropna(); n=len(z)
    if n<5 or z.x.nunique()<2 or z.y.nunique()<2: return {"n":n,"ic":np.nan,"se":np.nan,"low":np.nan,"high":np.nan,"p":np.nan,"n_eff":np.nan}
    xr=z.x.rank(method="average").to_numpy(float); yr=z.y.rank(method="average").to_numpy(float)
    xr=(xr-xr.mean())/xr.std(); yr=(yr-yr.mean())/yr.std()
    fit=hac_ols(yr,xr[:,None]); b,se,t=fit["beta"][1],fit["se"][1],fit["t"][1]
    ne=float(np.clip((1-b*b)/max(se*se,1e-15)+2,1,n))
    return {"n":n,"ic":float(b),"se":float(se),"low":float(max(-1,b-Z95*se)),"high":float(min(1,b+Z95*se)),"p":float(normal_p(t)),"n_eff":ne}

def block_boot_diff(frame, reps=BOOTSTRAPS, block=BLOCK, seed=SEED):
    """Moving-block bootstrap of OI-up minus OI-down return means/medians."""
    a=frame[["oi_change_1d","future_return_10d"]].to_numpy(float)
    valid=np.isfinite(a).all(axis=1); a=a[valid]; n=len(a)
    up=a[:,0]>0; down=a[:,0]<0
    if not up.any() or not down.any(): return {}
    mean_diff=float(a[up,1].mean()-a[down,1].mean())
    median_diff=float(np.median(a[up,1])-np.median(a[down,1]))
    rng=np.random.default_rng(seed); means=[]; medians=[]
    starts=np.arange(max(1,n-block+1))
    for _ in range(reps):
        ids=[]
        while len(ids)<n:
            s=int(rng.choice(starts)); ids.extend(range(s,min(s+block,n)))
        sample=a[np.asarray(ids[:n])]; u=sample[:,0]>0; d=sample[:,0]<0
        if u.any() and d.any():
            means.append(sample[u,1].mean()-sample[d,1].mean())
            medians.append(np.median(sample[u,1])-np.median(sample[d,1]))
    return {"n":n,"mean_diff_up_minus_down":mean_diff,"mean_ci_low":float(np.quantile(means,.025)),"mean_ci_high":float(np.quantile(means,.975)),"median_diff_up_minus_down":median_diff,"median_ci_low":float(np.quantile(medians,.025)),"median_ci_high":float(np.quantile(medians,.975)),"bootstrap_reps":len(means),"block_sessions":block}

def kruskal_wallis_4(groups):
    """Kruskal-Wallis H with tie correction; chi-square(df=3) survival."""
    chunks=[np.asarray(g,float) for g in groups if len(g)]
    if len(chunks)<2: return None
    allv=np.concatenate(chunks); ranks=pd.Series(allv).rank(method="average").to_numpy()
    n=len(allv); pos=0; h0=0.0
    for g in chunks:
        r=ranks[pos:pos+len(g)];pos+=len(g);h0+=(r.sum()**2)/len(g)
    h=12*h0/(n*(n+1))-3*(n+1)
    counts=pd.Series(allv).value_counts().to_numpy(float)
    tie=1-float(np.sum(counts**3-counts))/(n**3-n)
    h=h/tie if tie>0 else 0.0
    # Exact survival function for chi-square with 3 degrees of freedom.
    p=math.erfc(math.sqrt(h/2))+math.sqrt(2*h/math.pi)*math.exp(-h/2) if h>0 else 1.0
    return h,p

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=[PANEL_PATH,ROUTE_PATH,GAPS_PATH,*CONTRACT_FILES]
    hashes0={str(p):sha(p) for p in inputs}
    d=pd.read_csv(PANEL_PATH,parse_dates=["date"]).set_index("date").sort_index().loc[START:END].copy()
    route=pd.read_csv(ROUTE_PATH,parse_dates=["date"]).set_index("date").sort_index()
    d["contract"] = d.contract.astype(str)
    route_contract=route.contract.astype(str)
    if not route_contract.reindex(d.index).equals(d.contract):
        raise ValueError("Contract mapping differs between existing panel and audited route")
    # Read raw actual contract histories, calculate within-contract close and OI changes.
    raw={}
    for path in CONTRACT_FILES:
        sym=path.name[:5]
        f=pd.read_csv(path,parse_dates=["date"]).sort_values("date").set_index("date")
        f=f.loc[~f.index.duplicated(keep=False)]
        raw[sym]=f.reindex(route.index)
    same_ret=[]; same_oi=[]; same_vol=[]; prev_contract=[]; field_missing=[]
    dates=route.index
    for i,dt in enumerate(dates):
        sym=route_contract.iloc[i]; prev=route_contract.iloc[i-1] if i>0 else None
        prev_contract.append(prev)
        f=raw.get(sym)
        if f is None:
            same_ret.append(np.nan);same_oi.append(np.nan);same_vol.append(np.nan);field_missing.append(True);continue
        if i==0:
            same_ret.append(np.nan);same_oi.append(np.nan);same_vol.append(np.nan);field_missing.append(True);continue
        pdt=dates[i-1]
        vals=[f.at[dt,c] if c in f else np.nan for c in ["close","volume","hold"]]
        prevs=[f.at[pdt,c] if c in f else np.nan for c in ["close","volume","hold"]]
        c,v,oi=vals; pc,pv,poi=prevs
        same_ret.append(c/pc-1 if pd.notna(c) and pd.notna(pc) and pc>0 else np.nan)
        same_oi.append(oi/poi-1 if pd.notna(oi) and pd.notna(poi) and poi>0 and oi>0 else np.nan)
        same_vol.append(v/pv-1 if pd.notna(v) and pd.notna(pv) and pv>0 and v>=0 else np.nan)
        field_missing.append(any(pd.isna(x) for x in [c,v,oi]))
    diag=pd.DataFrame({"actual_contract_return_1d":same_ret,"oi_change_recomputed_same_contract":same_oi,"volume_change_recomputed_same_contract":same_vol,"previous_selected_contract":prev_contract},index=dates)
    diag["roll"]=route_contract.ne(route_contract.shift(1))
    diag.iloc[0,diag.columns.get_loc("roll")]=False
    d=d.join(diag.reindex(d.index))
    d["roll_proximity_5"] = d.roll.rolling(11,center=True,min_periods=1).max().astype(bool)
    d["oi_up"] = d.oi_change_recomputed_same_contract>0
    d["oi_down"] = d.oi_change_recomputed_same_contract<0
    d["oi_change_agreement"] = np.isclose(d.oi_change_1d,d.oi_change_recomputed_same_contract,equal_nan=True,atol=1e-10,rtol=1e-8)
    if not d.oi_change_agreement.fillna(False).all():
        raise ValueError(f"Existing OI factor differs from independent same-contract recomputation on {(~d.oi_change_agreement.fillna(False)).sum()} rows")
    # Primary matched sample requires valid actual-contract daily OI, target, and mapped data.
    d["future10_target_end"] = pd.Series(d.index,index=d.index).shift(-10)
    d["year"] = d.index.year
    d["quadrant"]="unavailable"
    qvalid=d.actual_contract_return_1d.notna() & d.oi_change_recomputed_same_contract.notna()
    cond=[qvalid&(d.actual_contract_return_1d>0)&(d.oi_change_recomputed_same_contract>0),qvalid&(d.actual_contract_return_1d<0)&(d.oi_change_recomputed_same_contract>0),qvalid&(d.actual_contract_return_1d>0)&(d.oi_change_recomputed_same_contract<0),qvalid&(d.actual_contract_return_1d<0)&(d.oi_change_recomputed_same_contract<0)]
    labs=["price_up_oi_up","price_down_oi_up","price_up_oi_down","price_down_oi_down"]
    d["quadrant"]=np.select(cond,labs,default="flat_or_unavailable")
    d.to_csv(OUT/"daily_oi_audit_panel.csv",index_label="date",encoding="utf-8-sig",float_format="%.10g")

    # OI increases vs decreases: non-roll primary, with moving-block bootstrap uncertainty.
    dist_rows=[]
    for roll_group,mask in [("all_dates",~d.future_return_10d.isna()),("exclude_roll_day",~d.roll), ("exclude_roll_plus_minus_5",~d.roll_proximity_5)]:
        z=d[mask & d.oi_change_recomputed_same_contract.notna() & d.future_return_10d.notna()]
        for direction,g in [("oi_increase",z[z.oi_change_recomputed_same_contract>0]),("oi_decrease",z[z.oi_change_recomputed_same_contract<0])]:
            y=g.future_return_10d
            dist_rows.append({"sample":roll_group,"oi_direction":direction,"n":len(g),"mean_future_return_10d":y.mean(),"median_future_return_10d":y.median(),"q25":y.quantile(.25),"q75":y.quantile(.75),"positive_return_share":(y>0).mean()})
    dist=pd.DataFrame(dist_rows);dist.to_csv(OUT/"oi_increase_decrease_distribution.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    boot=[]
    for name,mask in [("exclude_roll_day",~d.roll),("exclude_roll_plus_minus_5",~d.roll_proximity_5)]:
        z=d[mask & d.oi_change_recomputed_same_contract.notna() & d.future_return_10d.notna()]
        boot.append({"sample":name,**block_boot_diff(z)})
    pd.DataFrame(boot).to_csv(OUT/"oi_direction_block_bootstrap.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Four price/OI states use actual same-contract returns, including roll date if both rows exist.
    quad_rows=[]
    for label,g in d[d.quadrant.isin(labs)].groupby("quadrant"):
        y=g.future_return_10d.dropna()
        quad_rows.append({"quadrant":label,"n_factor_days":len(g),"n_future10":len(y),"mean_future10":y.mean(),"median_future10":y.median(),"future10_positive_share":(y>0).mean(),"roll_days":int(g.roll.sum())})
    quad=pd.DataFrame(quad_rows);quad.to_csv(OUT/"price_oi_quadrants.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    # One-way rank ANOVA Kruskal-Wallis descriptive omnibus; bootstrap block null is outside scope.
    groups=[d.loc[(d.quadrant==lab)&d.future_return_10d.notna(),"future_return_10d"].to_numpy() for lab in labs]
    kw=kruskal_wallis_4(groups) if all(len(x) for x in groups) else None
    # Time-series robust group means / pairwise contrasts on the full ordered sample.
    qsample=d[d.quadrant.isin(labs)&d.future_return_10d.notna()].copy()
    qsample["future_y"]=qsample.future_return_10d
    # Baseline is price-down/OI-down; each coefficient is a group-mean difference.
    ref="price_down_oi_down"; qcols=[x for x in labs if x!=ref]
    qX=np.column_stack([(qsample.quadrant==x).astype(float) for x in qcols])
    qfit=hac_ols(qsample.future_y.to_numpy(),qX,HAC_LAGS)
    qmeans=[]
    for lab in labs:
        vec=np.zeros(1+len(qcols));vec[0]=1
        if lab!=ref:vec[1+qcols.index(lab)]=1
        mu=float(vec@qfit["beta"]);se=float(np.sqrt(max(0,vec@qfit["cov"]@vec)))
        qmeans.append({"quadrant":lab,"n":int((qsample.quadrant==lab).sum()),"hac_mean_future10":mu,"se_hac":se,"ci95_low":mu-Z95*se,"ci95_high":mu+Z95*se,"p_vs_zero_normal":normal_p(mu/se if se>0 else np.nan)})
    pd.DataFrame(qmeans).to_csv(OUT/"quadrant_hac_means.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    contrasts=[]
    for i in range(len(labs)):
        for j in range(i+1,len(labs)):
            a,b=labs[i],labs[j];va=np.zeros(1+len(qcols));vb=va.copy();va[0]=vb[0]=1
            if a!=ref:va[1+qcols.index(a)]=1
            if b!=ref:vb[1+qcols.index(b)]=1
            v=va-vb;est=float(v@qfit["beta"]);se=float(np.sqrt(max(0,v@qfit["cov"]@v)));t=est/se if se>0 else np.nan
            contrasts.append({"contrast":f"{a} minus {b}","estimate":est,"se_hac":se,"ci95_low":est-Z95*se,"ci95_high":est+Z95*se,"p_hac_normal":normal_p(t)})
    contrasts=pd.DataFrame(contrasts);contrasts["bh_q_6_quadrant_contrasts"]=bh(contrasts.p_hac_normal);contrasts.to_csv(OUT/"quadrant_hac_pairwise.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    qbeta=qfit["beta"][1:];qcov=qfit["cov"][1:,1:];wald=float(qbeta@np.linalg.pinv(qcov)@qbeta)
    q_omnibus_p=math.erfc(math.sqrt(wald/2))+math.sqrt(2*wald/math.pi)*math.exp(-wald/2) if wald>0 else 1.0

    # Roll impact: compare roll-day and non-roll OI factor/return/volume distributions, plus event windows.
    roll_rows=[]
    for label,mask in [("roll_day",d.roll),("non_roll_day",~d.roll),("within_5_sessions_of_roll",d.roll_proximity_5),("outside_5_sessions_of_roll",~d.roll_proximity_5)]:
        g=d[mask]
        for variable in ["oi_change_recomputed_same_contract","actual_contract_return_1d","volume_change_recomputed_same_contract"]:
            x=g[variable].dropna()
            roll_rows.append({"group":label,"variable":variable,"n":len(x),"mean":x.mean(),"median":x.median(),"std":x.std(),"q05":x.quantile(.05),"q95":x.quantile(.95)})
    pd.DataFrame(roll_rows).to_csv(OUT/"roll_impact.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # IC robustness: all dates, excluding routing roll date, and excluding +/-5 sessions around roll.
    ic_rows=[]
    for sample,mask in [("all_dates",pd.Series(True,index=d.index)),("exclude_roll_day",~d.roll),("exclude_roll_plus_minus_5",~d.roll_proximity_5)]:
        g=d[mask]
        stat=rank_ic(g.oi_change_recomputed_same_contract,g.future_return_10d)
        ic_rows.append({"sample":sample,**stat})
    icdf=pd.DataFrame(ic_rows);icdf["bh_q_3_roll_sensitivity_tests"]=bh(icdf.p);icdf.to_csv(OUT/"rank_ic_roll_sensitivity.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Yearly and regime stability, primary excludes routing roll days and keeps horizon within year.
    annual=[]
    for year in range(2022,2027):
        g=d[(d.year==year)&(~d.roll)&d.future10_target_end.notna()&(d.future10_target_end.dt.year==year)]
        st=rank_ic(g.oi_change_recomputed_same_contract,g.future_return_10d)
        annual.append({"year":year,**st,"sample_description":"partial_year_to_2026-10-12" if year==2026 else "calendar_year","roll_days_excluded":int(d[(d.year==year)&d.roll].shape[0])})
    annual=pd.DataFrame(annual);annual["bh_q_5_annual_tests"]=bh(annual.p);annual.to_csv(OUT/"annual_oi_rank_ic.csv",index=False,encoding="utf-8-sig",float_format="%.10g")
    regimes=[]
    for dim,col,values in [("trend_state","state",["range","transition","uptrend","downtrend"]),("volatility_state","vol_state",["low","high"])]:
        for value in values:
            g=d[(d[col].astype(str)==value)&(~d.roll)]
            st=rank_ic(g.oi_change_recomputed_same_contract,g.future_return_10d)
            regimes.append({"dimension":dim,"state":value,**st})
    regimes=pd.DataFrame(regimes);regimes["bh_q_6_state_tests"]=bh(regimes.p);regimes.to_csv(OUT/"state_oi_rank_ic.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Rolling 252-session Rank IC on non-roll observations only.
    roll_ic=[]
    for end_i in range(ROLL_WINDOW-1,len(d),ROLL_STEP):
        w=d.iloc[end_i-ROLL_WINDOW+1:end_i+1];w=w[~w.roll]
        st=rank_ic(w.oi_change_recomputed_same_contract,w.future_return_10d)
        roll_ic.append({"window_start":d.index[end_i-ROLL_WINDOW+1],"window_end":d.index[end_i],"window_sessions":ROLL_WINDOW,"valid_n":st["n"],**{k:v for k,v in st.items() if k!="n"}})
    rolling=pd.DataFrame(roll_ic);rolling.to_csv(OUT/"rolling_oi_rank_ic.csv",index=False,encoding="utf-8-sig",float_format="%.10g",date_format="%Y-%m-%d")

    # Incremental information conditional on 20d momentum and 5d same-contract volume change.
    ctrl_rows=[]
    for sample,mask in [("all_dates",pd.Series(True,index=d.index)),("exclude_roll_day",~d.roll),("exclude_roll_plus_minus_5",~d.roll_proximity_5)]:
        x=d.loc[mask,["oi_change_recomputed_same_contract","momentum_20d","volume_change_5d","future_return_10d"]].dropna()
        ranks=x.rank(method="average",pct=False)
        yr=ranks.future_return_10d.to_numpy(float);yr=(yr-yr.mean())/yr.std()
        xr=ranks[["oi_change_recomputed_same_contract","momentum_20d","volume_change_5d"]].to_numpy(float)
        xr=(xr-xr.mean(axis=0))/xr.std(axis=0)
        base=hac_ols(yr,xr[:,1:],HAC_LAGS);full=hac_ols(yr,xr,HAC_LAGS)
        # OI coefficient corresponds to column 1 including intercept. Incremental R2 is descriptive.
        b,se,t=full["beta"][1],full["se"][1],full["t"][1]
        ctrl_rows.append({"sample":sample,"n":len(x),"oi_partial_rank_beta":b,"se_hac":se,"ci95_low":b-Z95*se,"ci95_high":b+Z95*se,"p_hac_normal":normal_p(t),"base_r2_rank_ols":base["r2"],"full_r2_rank_ols":full["r2"],"incremental_r2_descriptive":full["r2"]-base["r2"],"volume_partial_beta":full["beta"][3],"momentum_partial_beta":full["beta"][2],"hac_lags":HAC_LAGS})
    controls=pd.DataFrame(ctrl_rows);controls["bh_q_3_control_sensitivity_tests"]=bh(controls.p_hac_normal);controls.to_csv(OUT/"oi_incremental_information.csv",index=False,encoding="utf-8-sig",float_format="%.10g")

    # Figures: distribution, four quadrants, rolling IC, and partial coefficient sensitivity.
    fig,axes=plt.subplots(2,2,figsize=(14,10))
    z=d[(~d.roll)&d.future_return_10d.notna()&d.oi_change_recomputed_same_contract.notna()]
    axes[0,0].boxplot([100*z.loc[z.oi_change_recomputed_same_contract>0,"future_return_10d"],100*z.loc[z.oi_change_recomputed_same_contract<0,"future_return_10d"]],tick_labels=["OI increase","OI decrease"],showfliers=False)
    axes[0,0].axhline(0,color="black",lw=.7);axes[0,0].set_ylabel("Future 10-session return (%)");axes[0,0].set_title("Same-contract OI direction (roll days excluded)")
    quad_order=["price_up_oi_up","price_down_oi_up","price_up_oi_down","price_down_oi_down"]
    qdata=[100*d.loc[(d.quadrant==q)&d.future_return_10d.notna(),"future_return_10d"] for q in quad_order]
    axes[0,1].boxplot(qdata,tick_labels=["Price↑ OI↑","Price↓ OI↑","Price↑ OI↓","Price↓ OI↓"],showfliers=False);axes[0,1].axhline(0,color="black",lw=.7);axes[0,1].set_ylabel("Future 10-session return (%)");axes[0,1].set_title("Four same-contract price/OI states")
    axes[1,0].plot(pd.to_datetime(rolling.window_end),rolling.ic,color="#4c78a8",lw=1.3);axes[1,0].fill_between(pd.to_datetime(rolling.window_end),rolling.low,rolling.high,color="#4c78a8",alpha=.17);axes[1,0].axhline(0,color="black",lw=.7);axes[1,0].set_ylabel("10-session Rank IC");axes[1,0].set_title("252-session rolling IC; roll days excluded");axes[1,0].grid(alpha=.2)
    cc=controls;axes[1,1].errorbar(np.arange(len(cc)),cc.oi_partial_rank_beta,yerr=[cc.oi_partial_rank_beta-cc.ci95_low,cc.ci95_high-cc.oi_partial_rank_beta],fmt="o",capsize=4,color="#e45756");axes[1,1].axhline(0,color="black",lw=.7);axes[1,1].set_xticks(np.arange(len(cc)),["All","Exclude roll","Exclude ±5d"]);axes[1,1].set_ylabel("Partial rank-regression coefficient");axes[1,1].set_title("OI information controlling momentum & volume")
    fig.suptitle("Open Interest Factor Deep Dive — Exploratory, No Strategy")
    fig.tight_layout();fig.savefig(OUT/"oi_deep_dive.png",dpi=170);plt.close(fig)

    # Hash audit and quality checks.
    hashes1={str(p):sha(p) for p in inputs}
    if hashes0!=hashes1: raise RuntimeError("Protected source changed during analysis")
    gap=pd.read_csv(GAPS_PATH)
    summary={"research":"Open Interest Factor Deep Dive","period_start":str(START.date()),"period_end":str(END.date()),"factor":"same-contract one-session open-interest percentage change","target":"causal additive real-contract index close-to-close future 10-session return","roll_days":int(d.roll.sum()),"route_rows":len(d),"same_contract_oi_factor_match_rate":float(d.oi_change_agreement.mean()),"selected_route_rows_missing_raw_required_fields":int(d[["actual_contract_return_1d","oi_change_recomputed_same_contract"]].isna().any(axis=1).sum()),"all_raw_contract_gap_records":len(gap),"all_gap_records_automatically_filled":False,"quadrant_kruskal_wallis_h_unadjusted_iid_diagnostic":float(kw[0]) if kw else None,"quadrant_kruskal_wallis_p_unadjusted_iid_diagnostic":float(kw[1]) if kw else None,"quadrant_hac_omnibus_wald_chi2_3":wald,"quadrant_hac_omnibus_p_chi2_3":q_omnibus_p,"hac_lags":HAC_LAGS,"moving_block_bootstrap_reps":BOOTSTRAPS,"moving_block_length_sessions":BLOCK,"multiple_test_families":{"roll_sensitivity":3,"annual_oi_tests":5,"state_slices":6,"control_sensitivity":3,"quadrant_pairwise_contrasts":6},"exploratory_only":True,"genuine_independent_oos":False,"prior_strategy_backtests_data_modified":False,"strategy_version":"corn-adx20-atr3-forward-v1.0.0 (left unchanged; lock check in report generation)","trading_executed":False,"forward_paper_started":False,"source_hashes_unchanged":hashes0==hashes1,"source_hashes":hashes0}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"route_rows":len(d),"roll_days":int(d.roll.sum()),"annual":annual[["year","n","ic","low","high","bh_q_5_annual_tests"]].to_dict("records"),"controls":controls[["sample","n","oi_partial_rank_beta","ci95_low","ci95_high","bh_q_3_control_sensitivity_tests"]].to_dict("records"),"inputs_unchanged":hashes0==hashes1},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
