from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from akshare.futures import cons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
P=Path(__file__).parent; R=P.parent
protected=[f for f in R.rglob('*') if '.venv' not in f.parts and P not in f.parents]
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected if f.is_file()}
full=pd.read_csv(R/'long_history_audit/C0_daily.csv',parse_dates=['date']).set_index('date').sort_index()
first=full.index.get_loc(pd.Timestamp('2021-10-12'))
c=full.iloc[first-60:full.index.get_loc(pd.Timestamp('2026-10-12'))+1].copy()
assert len(c)==1272
mapping=pd.read_csv(R/'long_history_audit/ten_year_contract_mapping.csv',parse_dates=['date']).set_index('date')
assert mapping.loc['2021-10-12':'2026-10-12','count'].eq(1).all()
data={}
for f in list((R/'long_history_audit').glob('C????_daily.csv')):
    d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
    assert not d.index.duplicated().any() and not d[['open','high','low','close','volume','hold']].isna().any().any()
    data[f.name.split('_')[0]]=d
calendar=pd.DatetimeIndex(pd.to_datetime(cons.get_calendar()))
assert len(calendar[(calendar>=c.index.min())&(calendar<=c.index.max())].difference(c.index))==0
missing=[]
for s,d in data.items():
    for date in calendar[(calendar>=d.index.min())&(calendar<=d.index.max())].difference(d.index):
        if c.index.min()<=date<=c.index.max():
            missing.append(dict(contract=s,date=date,delivery_month=s[1:5],excluded_by_delivery_month=date.strftime('%y%m')>=s[1:5]))
pd.DataFrame(missing).to_csv(P/'missing_dates.csv',index=False,encoding='utf-8-sig')
def eligible(s,date): return s[1:5]>date.strftime('%y%m')
def select(prev,today,current):
    pool=[]
    for s,d in data.items():
        if not eligible(s,today) or (current and s[1:5]<current[1:5]): continue
        if d.index.min()<=prev<=d.index.max() and prev not in d.index:
            raise RuntimeError(f'Missing eligible selection observation: {s} {prev}')
        if prev in d.index: pool.append((float(d.loc[prev,'hold']),s))
    if not pool: raise RuntimeError(f'No candidates: {today}')
    maximum=max(q for q,s in pool)
    tied=sorted(s for q,s in pool if q==maximum)
    return current if current in tied else tied[0]
# Build a causal additive price index, stripping contract-level differences.
dates=list(c.index); initial_candidates=[s for s,d in data.items() if dates[0] in d.index and eligible(s,dates[0])]
current=max(initial_candidates,key=lambda s:float(data[s].loc[dates[0],'hold']))
level=float(data[current].loc[dates[0],'close'])
records=[dict(date=dates[0],contract=current,old_contract=current,index_close=level,open=float(data[current].loc[dates[0],'open']),close=level,roll=False)]
for i in range(1,len(dates)):
    today,prev=dates[i],dates[i-1]
    new=select(prev,today,current)
    olddata,newdata=data[current],data[new]
    for s,t in [(current,prev),(current,today),(new,today)]:
        if t not in data[s].index: raise RuntimeError(f'Missing required price: {s} {t}')
    change=float(olddata.loc[today,'open']-olddata.loc[prev,'close']+newdata.loc[today,'close']-newdata.loc[today,'open'])
    level+=change
    records.append(dict(date=today,contract=new,old_contract=current,index_close=level,open=float(newdata.loc[today,'open']),close=float(newdata.loc[today,'close']),roll=new!=current))
    current=new
signal=pd.DataFrame(records).set_index('date')
signal['ma20']=signal.index_close.rolling(20).mean();signal['ma60']=signal.index_close.rolling(60).mean()
signal['signal']=((signal.ma20>signal.ma60)&signal.ma60.notna()).astype(int)
signal['target']=signal.signal.shift(1).fillna(0).astype(int)
signal.to_csv(P/'causal_signal_and_contracts.csv',encoding='utf-8-sig')
x=signal.iloc[60:];capital=22250.
def run(benchmark=False,cost=False):
    equity=capital;held=None;rows=[];trades=[];roll_count=0
    for j,(date,row) in enumerate(x.iterrows()):
        i=dates.index(date);prev=dates[i-1]
        q=1 if benchmark else int(row.target)
        wanted=row.contract if q else None
        pnl=0.;legs=[]
        if held:
            if date not in data[held].index: raise RuntimeError(f'Missing held price {held} {date}')
            pnl+=10*(data[held].loc[date,'open']-data[held].loc[prev,'close'])
        if held!=wanted:
            isroll=held is not None and wanted is not None
            if isroll: roll_count+=1
            if held: legs.append((held,'sell','roll' if isroll else 'signal_exit',float(data[held].loc[date,'open'])))
            if wanted: legs.append((wanted,'buy','roll' if isroll else 'signal_entry',float(data[wanted].loc[date,'open'])))
        if wanted: pnl+=10*(data[wanted].loc[date,'close']-data[wanted].loc[date,'open'])
        if j==len(x)-1 and wanted:
            legs.append((wanted,'sell','sample_end',float(data[wanted].loc[date,'close'])))
        fee=len(legs)*2 if cost else 0.;slip=len(legs)*10 if cost else 0.
        before=equity;equity+=pnl-fee-slip
        for s,side,reason,px in legs:
            trades.append(dict(date=date,decision_date=prev,contract=s,side=side,reason=reason,reference_price=px,execution_price=px+(1 if side=='buy' else -1)*(1 if cost else 0),fee=2 if cost else 0,slippage_cost=10 if cost else 0))
        rows.append(dict(date=date,contract=wanted,position=int(q),gross_pnl=pnl,fee=fee,slippage=slip,equity=equity,nav=equity/capital,daily_return=equity/before-1))
        held=wanted
    a=pd.DataFrame(rows);t=pd.DataFrame(trades);nav=np.r_[1,a.nav];ret=a.daily_return
    assert (a.equity>0).all()
    # Independent cash-flow reconciliation after final liquidation.
    cashflow=sum((1 if z.side=='sell' else -1)*z.execution_price*10-z.fee for z in t.itertuples())
    assert np.isclose(cashflow,equity-capital)
    assert (t[t.reason!='sample_end'].decision_date<t[t.reason!='sample_end'].date).all()
    assert (t.side=='buy').sum()==(t.side=='sell').sum()
    m=dict(cumulative_return=float(nav[-1]-1),annualized_return=float(nav[-1]**(252/len(a))-1),maximum_drawdown=float(-(nav/np.maximum.accumulate(nav)-1).min()),sharpe=float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)),executions=len(t),rolls=roll_count,signal_entries=int((t.reason=='signal_entry').sum()),total_fees=float(a.fee.sum()),total_slippage=float(a.slippage.sum()),ending_equity=equity)
    cycles=[]; running=None
    for z in t.itertuples():
        if z.reason=='signal_entry':
            assert running is None
            running={'entry_date':z.date,'exit_date':None,'net_pnl':0.,'gross_pnl':0.,'fees':0.,'slippage':0.,'roll_cost':0.,'contracts':[]}
        assert running is not None
        running['gross_pnl']+=(1 if z.side=='sell' else -1)*z.reference_price*10
        running['net_pnl']+=(1 if z.side=='sell' else -1)*z.execution_price*10-z.fee
        running['fees']+=z.fee;running['slippage']+=z.slippage_cost
        if z.reason=='roll':running['roll_cost']+=z.fee+z.slippage_cost
        if z.side=='buy':running['contracts'].append(z.contract)
        if z.reason in ['signal_exit','sample_end']:
            running['exit_date']=z.date;running['contracts']=' → '.join(running['contracts']);running['sample_end_exit']=z.reason=='sample_end'
            cycles.append(running);running=None
    assert running is None and np.isclose(sum(z['net_pnl'] for z in cycles),equity-capital)
    m['completed_trades']=len(cycles)
    m['winning_trades']=sum(z['net_pnl']>0 for z in cycles)
    m['win_rate']=m['winning_trades']/len(cycles) if cycles else None
    m['sample_end_exits']=sum(z['sample_end_exit'] for z in cycles)
    m['roll_cost_subset']=sum(z['roll_cost'] for z in cycles)
    return a,t,m,pd.DataFrame(cycles)
old=json.loads((R/'corrected_backtest/comparison.json').read_text())
results={};arrays={}
for name,base,cost in [('corrected_gross',False,False),('corrected_net',False,True),('benchmark_gross',True,False),('benchmark_net',True,True)]:
    a,t,m,cycles=run(base,cost);arrays[name]=a;results[name]=m
    cycles.to_csv(P/f'{name}_cycles.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    a.to_csv(P/f'{name}_daily.csv',index=False,encoding='utf-8-sig');t.to_csv(P/f'{name}_trades.csv',index=False,encoding='utf-8-sig')
for name in ['corrected','benchmark']:
    assert np.allclose(arrays[name+'_gross'].equity-arrays[name+'_net'].equity,(arrays[name+'_net'].fee+arrays[name+'_net'].slippage).cumsum())
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==v for f,v in hashes.items())
summary=dict(period_start=str(x.index.min().date()),period_end=str(x.index.max().date()),days=len(x),initial_capital=capital,missing_records=len(missing),all_missing_excluded_by_delivery_month=all(m['excluded_by_delivery_month'] for m in missing),required_prices_missing=0,protected_files_unchanged=True,old_two_year=old['corrected'],corrected=results)
(P/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
fig,ax=plt.subplots(figsize=(13,6));fig.subplots_adjust(bottom=.16)
for name,label in [('corrected_gross','Corrected strategy gross'),('corrected_net','Corrected strategy net'),('benchmark_net','Actual-contract benchmark net')]: ax.plot(x.index,arrays[name].nav,label=label)

ax.set(title='Corn 20/60 SMA — Five-year Actual-contract Backtest',ylabel='NAV',xlabel='Date');ax.legend(frameon=False);ax.grid(alpha=.2)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.text(.01,.02,'Prior-day OI selection; next-open execution; causal additive signal index; fees/slippage on every entry, exit and roll leg',fontsize=9)
fig.savefig(P/'five_year_nav.png',dpi=180)
(P/'protected_input_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
