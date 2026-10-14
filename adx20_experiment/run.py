from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from akshare.futures import cons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
P=Path(__file__).parent;R=P.parent
protected=[f for f in R.rglob('*') if f.is_file() and '.venv' not in f.parts and P not in f.parents]
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
calendar=pd.DatetimeIndex(pd.to_datetime(cons.get_calendar()))
data={f.name[:5]:pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index() for f in (R/'long_history_audit').glob('C????_daily.csv')}
full=pd.read_csv(R/'long_history_audit/C0_daily.csv',parse_dates=['date']).set_index('date').sort_index()
first=full.index.get_loc(pd.Timestamp('2021-10-12'))
dates=full.index[first-60:full.index.get_loc(pd.Timestamp('2026-10-12'))+1]
assert len(dates)==1272
assert len(calendar[(calendar>=dates.min())&(calendar<=dates.max())].difference(dates))==0
def wilder(a,n=14):
    out=np.full(len(a),np.nan);seed=[];last=np.nan
    for i,v in enumerate(np.asarray(a,float)):
        if not np.isfinite(v):seed=[];last=np.nan;continue
        if np.isnan(last):
            seed.append(v)
            if len(seed)==n:last=np.mean(seed);out[i]=last
        else:last=(last*(n-1)+v)/n;out[i]=last
    return out
def adx(frame):
    z=frame.reindex(calendar[(calendar>=frame.index.min())&(calendar<=frame.index.max())])
    up=z.high.diff();dn=-z.low.diff();valid=z[['high','low','close']].notna().all(axis=1)&z[['high','low','close']].shift().notna().all(axis=1)
    plus=pd.Series(np.where((up>dn)&(up>0),up,0),index=z.index).where(valid)
    minus=pd.Series(np.where((dn>up)&(dn>0),dn,0),index=z.index).where(valid)
    tr=pd.concat([z.high-z.low,(z.high-z.close.shift()).abs(),(z.low-z.close.shift()).abs()],axis=1).max(axis=1).where(valid)
    atr=wilder(tr);p=100*wilder(plus)/atr;m=100*wilder(minus)/atr;den=p+m
    dx=np.divide(100*abs(p-m),den,out=np.zeros(len(z)),where=den!=0);dx[~np.isfinite(den)]=np.nan
    return pd.Series(wilder(dx),index=z.index)
ind={s:adx(d) for s,d in data.items()}
def choose(prev,today,current,frames):
    pool=[]
    for s,d in frames.items():
        if s[1:5]<=today.strftime('%y%m') or (current and s[1:5]<current[1:5]):continue
        if d.index.min()<=prev<=d.index.max() and prev not in d.index:raise ValueError(f'Missing prior OI: {s} {prev}')
        if prev in d.index:pool.append((float(d.loc[prev,'hold']),s))
    top=max(v[0] for v in pool);tied=sorted(s for v,s in pool if v==top)
    return current if current in tied else tied[0]
def signal_path(days,frames):
    start=days[0];current=max((s for s,d in frames.items() if start in d.index and s[1:5]>start.strftime('%y%m')),key=lambda s:float(frames[s].loc[start,'hold']))
    level=float(frames[current].loc[start,'close']);rows=[dict(date=start,contract=current,old_contract=current,index_close=level,roll=False)]
    for i in range(1,len(days)):
        dt,prev=days[i],days[i-1];new=choose(prev,dt,current,frames)
        for s,t in [(current,prev),(current,dt),(new,dt)]:
            if t not in frames[s].index:raise ValueError(f'Missing required price: {s} {t}')
        level+=frames[current].loc[dt,'open']-frames[current].loc[prev,'close']+frames[new].loc[dt,'close']-frames[new].loc[dt,'open']
        rows.append(dict(date=dt,contract=new,old_contract=current,index_close=float(level),roll=new!=current));current=new
    a=pd.DataFrame(rows).set_index('date');a['ma20']=a.index_close.rolling(20).mean();a['ma60']=a.index_close.rolling(60).mean();a['bullish']=((a.ma20>a.ma60)&a.ma60.notna()).astype(int)
    return a
s=signal_path(dates,data)
s.to_csv(P/'independent_signal_path.csv',encoding='utf-8-sig')
def simulate(s,frames,indicators,filtered,fee_cost=True,liquidate=True):
    capital=22250.;equity=capital;held=None;q=0;rows=[];trades=[];decisions=[];cycles=[];cycle=None;rolls=0
    for i in range(60,len(s)):
        dt,prev=s.index[i],s.index[i-1];new=s.contract.iloc[i]
        bullish=bool(s.bullish.iloc[i-1]);cross=bullish and (i==60 or not bool(s.bullish.iloc[i-2]))
        value=float(indicators[new].loc[prev])
        before_q=q
        if not bullish:q=0
        elif cross and q==0:q=int(not filtered or (np.isfinite(value) and value>=20))
        if cross:
            decisions.append(dict(signal_date=prev,execution_date=dt,contract=new,adx14_asof_signal=value,accepted=bool(q),reason='cross_accepted' if q else 'ADX_below20_or_missing'))
        wanted=new if q else None;pnl=0.;legs=[]
        if held:
            if dt not in frames[held].index:raise ValueError(f'held price missing {held} {dt}')
            pnl+=10*(frames[held].loc[dt,'open']-frames[held].loc[prev,'close'])
        if held!=wanted:
            roll=held is not None and wanted is not None
            if roll:rolls+=1
            if held:legs.append((held,'sell','roll' if roll else 'signal_exit',float(frames[held].loc[dt,'open'])))
            if wanted:legs.append((wanted,'buy','roll' if roll else 'signal_entry',float(frames[wanted].loc[dt,'open'])))
        if wanted:pnl+=10*(frames[wanted].loc[dt,'close']-frames[wanted].loc[dt,'open'])
        if i==len(s)-1 and wanted and liquidate:legs.append((wanted,'sell','sample_end',float(frames[wanted].loc[dt,'close'])))
        fee=len(legs)*2 if fee_cost else 0;slip=len(legs)*10 if fee_cost else 0
        prior=equity;equity+=pnl-fee-slip
        for sym,side,reason,px in legs:
            z=dict(signal_date=prev,execution_date=dt,contract=sym,side=side,reason=reason,reference_price=px,execution_price=px+(1 if side=='buy' else -1)*(1 if fee_cost else 0),fee=2 if fee_cost else 0,slippage=10 if fee_cost else 0);trades.append(z)
            if reason=='signal_entry':cycle=dict(entry_date=dt,exit_date=None,gross_pnl=0.,net_pnl=0.,cost=0.,roll_cost=0.,contracts=[])
            assert cycle is not None
            cycle['gross_pnl']+=(1 if side=='sell' else -1)*px*10
            cycle['net_pnl']+=(1 if side=='sell' else -1)*z['execution_price']*10-z['fee']
            cycle['cost']+=z['fee']+z['slippage']
            if reason=='roll':cycle['roll_cost']+=z['fee']+z['slippage']
            if side=='buy':cycle['contracts'].append(sym)
            if reason in ['signal_exit','sample_end']:
                cycle['exit_date']=dt;cycle['contracts']=' → '.join(cycle['contracts']);cycles.append(cycle);cycle=None
        rows.append(dict(date=dt,contract=wanted,position=q,gross_pnl=pnl,fee=fee,slippage=slip,equity=equity,nav=equity/capital,daily_return=equity/prior-1))
        held=wanted
    a=pd.DataFrame(rows).set_index('date');t=pd.DataFrame(trades);c=pd.DataFrame(cycles);decision=pd.DataFrame(decisions)
    if liquidate:
        assert cycle is None
        assert np.isclose(c.net_pnl.sum(),equity-capital)
        cash=sum((1 if z['side']=='sell' else -1)*z['execution_price']*10-z['fee'] for z in trades)
        assert np.isclose(cash,equity-capital)
    nav=np.r_[1,a.nav];ret=a.daily_return
    m=dict(cumulative_return=float(nav[-1]-1),annualized_return=float(nav[-1]**(252/len(a))-1),maximum_drawdown=float(-(nav/np.maximum.accumulate(nav)-1).min()),sharpe=float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)),completed_trades=len(c),wins=int(c.net_pnl.gt(0).sum()),win_rate=float(c.net_pnl.gt(0).mean()),executions=len(t),rolls=rolls,fees=float(a.fee.sum()),slippage=float(a.slippage.sum()),total_cost=float((a.fee+a.slippage).sum()),roll_cost_subset=float(c.roll_cost.sum()),ending_equity=float(equity))
    return a,t,c,decision,m
results={};arrays={};cyc={};decs={}
for name,fil,cost in [('A_net',False,True),('B_net',True,True),('A_gross',False,False),('B_gross',True,False)]:
    a,t,c,de,m=simulate(s,data,ind,fil,cost);results[name]=m;arrays[name]=a;cyc[name]=c;decs[name]=de
    a.to_csv(P/f'{name}_daily.csv',encoding='utf-8-sig');t.to_csv(P/f'{name}_trades.csv',index=False,encoding='utf-8-sig');c.to_csv(P/f'{name}_cycles.csv',index=False,encoding='utf-8-sig');de.to_csv(P/f'{name}_decisions.csv',index=False,encoding='utf-8-sig')
for name in ['A','B']:
    assert np.allclose(arrays[name+'_gross'].equity-arrays[name+'_net'].equity,(arrays[name+'_net'].fee+arrays[name+'_net'].slippage).cumsum())
original=pd.read_csv(R/'five_year_backtest/corrected_net_daily.csv',parse_dates=['date']).set_index('date')
for key in ['position','gross_pnl','fee','slippage','equity']:assert np.allclose(arrays['A_net'][key],original[key])
# Prefix tests rebuild selections, ADX, signals and both account simulations from truncated raw data.
checks=[]
for cutoff in ['2022-12-30','2024-12-31','2025-12-31']:
    fs={sym:d.loc[:cutoff] for sym,d in data.items() if not d.loc[:cutoff].empty};ii={sym:adx(d) for sym,d in fs.items()};ss=signal_path(dates[dates<=pd.Timestamp(cutoff)],fs)
    for name,fil in [('A_net',False),('B_net',True)]:
        aa,*_=simulate(ss,fs,ii,fil,True,False)
        for col in ['position','gross_pnl','fee','slippage','equity']:assert np.allclose(aa[col],arrays[name].loc[aa.index,col])
    checks.append(dict(cutoff=cutoff,both_strategies_prefix_invariant=True))
impact=[]
priorfeatures=pd.read_csv(R/'five_year_trade_research/twelve_trade_features.csv',parse_dates=['entry_date','exit_date'])
for i,q in cyc['A_net'].iterrows():
    dt=q.entry_date;decision=decs['B_net'][decs['B_net'].execution_date.eq(dt)].iloc[0]
    match=cyc['B_net'][cyc['B_net'].entry_date.eq(dt)]
    selected=s.loc[dt:q.exit_date,'contract']
    later=[]
    for dd in selected.index[1:]:
        prev=s.index[s.index.get_loc(dd)-1]
        val=ind[s.loc[dd,'contract']].loc[prev]
        if np.isfinite(val) and val>=20:later.append(dd)
    impact.append(dict(original_trade=i+1,signal_date=decision.signal_date,execution_date=dt,adx=decision.adx14_asof_signal,accepted=bool(decision.accepted),original_net_pnl=q.net_pnl,original_peak_gross_profit=float(priorfeatures.iloc[i].max_gross_profit),B_entry=None if match.empty else match.iloc[0].entry_date,first_later_ADX_pass_date=None if not later else later[0],delayed_entry=False))
pd.DataFrame(impact).to_csv(P/'entry_filter_impact.csv',index=False,encoding='utf-8-sig')
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
summary=dict(start='2021-10-12',end='2026-10-12',days=1212,capital=22250,adx_threshold=20,entry_policy='Golden-cross day only; rejected cross waits for next golden cross, no delayed entry.',source_unchanged=True,A_matches_previous_daily_results=True,prefix_tests=checks,results=results)
(P/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
fig,(ax,ay)=plt.subplots(2,1,figsize=(13,8),sharex=True);fig.subplots_adjust(bottom=.13,hspace=.16)
for name,color in [('A_net','#467b9c'),('B_net','#d58b32')]:
    a=arrays[name];ax.plot(a.index,a.nav,label=name,color=color);nav=np.r_[1,a.nav];dd=nav/np.maximum.accumulate(nav)-1;ay.plot(a.index,dd[1:]*100,color=color,label=name)
ax.set(title='Corn 20/60 SMA — Independent A/B Experiment, ADX14 >= 20',ylabel='NAV');ax.legend(frameon=False);ay.set(ylabel='Drawdown %',xlabel='Date')
for a in [ax,ay]:a.grid(alpha=.2)
ay.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ay.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.text(.01,.025,'B tests ADX only on golden-cross signal days; executes next open; no late entry. Same size, costs and causal roll rules.',fontsize=9)
fig.savefig(P/'AB_nav_drawdown.png',dpi=170)
print(json.dumps(summary,ensure_ascii=False,indent=2));print(pd.DataFrame(impact).to_string(index=False))
