from pathlib import Path
import ast,json,hashlib,math
import numpy as np
import pandas as pd
from akshare.futures import cons
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
P=Path(__file__).parent;R=P.parent;E=R/'adx20_experiment'
files=[f for f in R.rglob('*') if f.is_file() and '.venv' not in f.parts and P not in f.parents]
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
calendar=pd.DatetimeIndex(pd.to_datetime(cons.get_calendar()))
data={f.name[:5]:pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index() for f in (R/'long_history_audit').glob('C????_daily.csv')}
full=pd.read_csv(R/'long_history_audit/C0_daily.csv',parse_dates=['date']).set_index('date').sort_index()
first=full.index.get_loc(pd.Timestamp('2021-10-12'));dates=full.index[first-60:full.index.get_loc(pd.Timestamp('2026-10-12'))+1]
# Reuse pure, frozen indicator and causal selection functions; do not execute old script.
tree=ast.parse((E/'run.py').read_text());names={'wilder','adx','choose','signal_path'}
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
exec(compile(functions,'frozen-pure-functions','exec'),globals())
def atr(frame):
    z=frame.reindex(calendar[(calendar>=frame.index.min())&(calendar<=frame.index.max())])
    valid=z[['high','low','close']].notna().all(axis=1)&z.close.shift().notna()
    tr=pd.concat([z.high-z.low,(z.high-z.close.shift()).abs(),(z.low-z.close.shift()).abs()],axis=1).max(axis=1).where(valid)
    return pd.Series(wilder(tr,14),index=z.index)
ind={sym:adx(d) for sym,d in data.items()};atrs={sym:atr(d) for sym,d in data.items()}
s=signal_path(dates,data)
def simulate(path,frames,adxs,ats,use_stop=True,cost=True,liquidate=True):
    equity=22250.;held=None;previous_close=None;stop=None;highwater=None;cycle=None
    rows=[];trades=[];cycles=[];logs=[];decisions=[];rolls=0
    def leg(date,prev,sym,side,reason,price):
        nonlocal cycle
        fee=2 if cost else 0;slip=10 if cost else 0
        is_atr=reason.startswith('atr_')
        z=dict(signal_date=date if is_atr else prev,stop_level_asof_date=prev if is_atr else None,trigger_date=date if is_atr else None,execution_date=date,contract=sym,side=side,reason=reason,reference_price=float(price),execution_price=float(price+(1 if side=='buy' else -1)*(1 if cost else 0)),fee=fee,slippage=slip)
        trades.append(z)
        if reason=='signal_entry':cycle=dict(entry_date=date,exit_date=None,exit_reason=None,gross_pnl=0.,net_pnl=0.,cost=0.,roll_cost=0.,contracts=[])
        assert cycle is not None
        sign=1 if side=='sell' else -1
        cycle['gross_pnl']+=sign*price*10;cycle['net_pnl']+=sign*z['execution_price']*10-fee;cycle['cost']+=fee+slip
        if reason=='roll':cycle['roll_cost']+=fee+slip
        if side=='buy':cycle['contracts'].append(sym)
        if side=='sell' and reason!='roll':
            cycle['exit_date']=date;cycle['exit_reason']=reason;cycle['contracts']=' → '.join(cycle['contracts']);cycles.append(cycle);cycle=None
    for i in range(60,len(path)):
        date,prev=path.index[i],path.index[i-1];selected=path.contract.iloc[i]
        bullish=bool(path.bullish.iloc[i-1]);cross=bullish and (i==60 or not bool(path.bullish.iloc[i-2]))
        ad=float(adxs[selected].loc[prev]);atrprev=float(ats[selected].loc[prev])
        before=equity;nlegs=len(trades);pnl=0.;stopped_open=False;roll_event=None;intraday_contract=None
        log=dict(date=date,prior_date=prev,old_contract=held,selected_contract=selected,stop_old_at_open=stop,highwater_old=highwater,roll_spread=None,stop_after_roll=None,initial_atr_asof=None,stop_active_intraday=None,stop_next_session=None,exit_reason=None)
        if held:
            bar=frames[held].loc[date];pnl+=(bar.open-previous_close)*10
            if use_stop and bar.open<=stop:
                leg(date,prev,held,'sell','atr_gap_open',bar.open);log['exit_reason']='atr_gap_open';stopped_open=True;held=None;stop=None;highwater=None
            elif not bullish:
                leg(date,prev,held,'sell','ma_exit',bar.open);log['exit_reason']='ma_exit';held=None;stop=None;highwater=None
            elif held!=selected:
                old=held;oldopen=float(bar.open);newopen=float(frames[selected].loc[date,'open']);spread=newopen-oldopen
                leg(date,prev,old,'sell','roll',oldopen);leg(date,prev,selected,'buy','roll',newopen);rolls+=1
                if use_stop:
                    oldstop=stop;oldwater=highwater;stop=oldstop+spread;highwater=oldwater+spread
                    assert np.isclose(newopen-stop,oldopen-oldstop)
                    assert np.isclose(highwater-newopen,oldwater-oldopen)
                    log['roll_spread']=spread;log['stop_after_roll']=stop
                held=selected
        if cross:
            accepted=not held and not stopped_open and np.isfinite(ad) and ad>=20
            decisions.append(dict(signal_date=prev,execution_date=date,contract=selected,adx14=ad,accepted=accepted,atr14_prior_close=atrprev))
            if accepted:
                if use_stop and (not np.isfinite(atrprev) or atrprev<=0):raise ValueError('Required entry ATR missing')
                entryopen=float(frames[selected].loc[date,'open']);leg(date,prev,selected,'buy','signal_entry',entryopen);held=selected
                if use_stop:highwater=entryopen;stop=math.floor(entryopen-3*atrprev);log['initial_atr_asof']=str(prev.date())
        if held:
            bar=frames[held].loc[date];intraday_contract=held
            if use_stop:
                log['stop_active_intraday']=stop
                # The line was known before the day's high/low: no same-day high ratchet.
                if bar.open<=stop:
                    # Can only occur at initial entry if malformed data/ATR; roll preserves distance.
                    leg(date,prev,held,'sell','atr_gap_open',bar.open);log['exit_reason']='atr_gap_open';held=None;stop=None;highwater=None
                elif bar.low<=stop:
                    pnl+=(stop-bar.open)*10;leg(date,prev,held,'sell','atr_intraday',stop);log['exit_reason']='atr_intraday';held=None;stop=None;highwater=None
                else:
                    pnl+=(bar.close-bar.open)*10
                    nowatr=float(ats[held].loc[date])
                    if not np.isfinite(nowatr) or nowatr<=0:raise ValueError('Required end-of-day ATR missing')
                    highwater=max(highwater,float(bar.high));newstop=max(stop,math.floor(highwater-3*nowatr));assert newstop>=stop
                    stop=newstop;log['stop_next_session']=stop
            else:pnl+=(bar.close-bar.open)*10
        if i==len(path)-1 and held and liquidate:
            leg(date,prev,held,'sell','sample_end',float(frames[held].loc[date,'close']));log['exit_reason']='sample_end';held=None;stop=None;highwater=None
        newlegs=trades[nlegs:];fees=sum(z['fee'] for z in newlegs);slip=sum(z['slippage'] for z in newlegs)
        for z in newlegs:
            assert frames[z['contract']].loc[date,'volume']>0
        equity+=pnl-fees-slip
        rows.append(dict(date=date,contract_close=held,intraday_contract=intraday_contract,position_close=int(held is not None),gross_pnl=float(pnl),fee=fees,slippage=slip,equity=equity,nav=equity/22250,daily_return=equity/before-1))
        logs.append(log)
        previous_close=float(frames[held].loc[date,'close']) if held else None
    a=pd.DataFrame(rows).set_index('date');t=pd.DataFrame(trades);c=pd.DataFrame(cycles);lg=pd.DataFrame(logs);de=pd.DataFrame(decisions)
    if liquidate:
        assert cycle is None
        assert np.isclose(c.net_pnl.sum(),equity-22250)
        cash=sum((1 if z.side=='sell' else -1)*z.execution_price*10-z.fee for z in t.itertuples());assert np.isclose(cash,equity-22250)
    assert (a.equity>0).all()
    nav=np.r_[1,a.nav];ret=a.daily_return
    metrics=dict(cumulative_return=float(nav[-1]-1),annualized_return=float(nav[-1]**(252/len(a))-1),maximum_drawdown=float(-(nav/np.maximum.accumulate(nav)-1).min()),sharpe=float(ret.mean()/ret.std(ddof=1)*np.sqrt(252)),trades=len(c),wins=int(c.net_pnl.gt(0).sum()),win_rate=float(c.net_pnl.gt(0).mean()),executions=len(t),rolls=rolls,fees=float(a.fee.sum()),slippage=float(a.slippage.sum()),total_cost=float((a.fee+a.slippage).sum()),roll_cost_subset=float(c.roll_cost.sum()),ending_equity=float(equity),atr_gap_exits=int(c.exit_reason.eq('atr_gap_open').sum()),atr_intraday_exits=int(c.exit_reason.eq('atr_intraday').sum()),ma_exits=int(c.exit_reason.eq('ma_exit').sum()))
    return a,t,c,lg,de,metrics
results={};arrays={};cycle_tables={};logs={}
for name,st,cost in [('baseline_net',False,True),('ATR3_net',True,True),('baseline_gross',False,False),('ATR3_gross',True,False)]:
    a,t,c,lg,de,m=simulate(s,data,ind,atrs,st,cost);results[name]=m;arrays[name]=a;cycle_tables[name]=c;logs[name]=lg
    a.to_csv(P/f'{name}_daily.csv',encoding='utf-8-sig');t.to_csv(P/f'{name}_trades.csv',index=False,encoding='utf-8-sig');c.to_csv(P/f'{name}_cycles.csv',index=False,encoding='utf-8-sig');lg.to_csv(P/f'{name}_stop_log.csv',index=False,encoding='utf-8-sig');de.to_csv(P/f'{name}_decisions.csv',index=False,encoding='utf-8-sig')
old=pd.read_csv(E/'B_net_daily.csv',parse_dates=['date']).set_index('date')
for key in ['gross_pnl','fee','slippage','equity']:assert np.allclose(arrays['baseline_net'][key],old[key])
for name in ['baseline','ATR3']:
    assert np.allclose(arrays[name+'_gross'].equity-arrays[name+'_net'].equity,(arrays[name+'_net'].fee+arrays[name+'_net'].slippage).cumsum())
checks=[]
for cutoff in ['2022-12-30','2024-12-31','2025-12-31']:
    fs={sym:d.loc[:cutoff] for sym,d in data.items() if not d.loc[:cutoff].empty};ii={sym:adx(d) for sym,d in fs.items()};aa={sym:atr(d) for sym,d in fs.items()};ss=signal_path(dates[dates<=pd.Timestamp(cutoff)],fs)
    part,*_=simulate(ss,fs,ii,aa,True,True,False)
    for key in ['position_close','gross_pnl','fee','slippage','equity']:assert np.allclose(part[key],arrays['ATR3_net'].loc[part.index,key])
    checks.append(dict(cutoff=cutoff,prefix_invariant=True))
base=cycle_tables['baseline_net'];test=cycle_tables['ATR3_net'];comparison=base.merge(test,on='entry_date',suffixes=('_baseline','_ATR'),how='outer',indicator=True)
assert comparison['_merge'].eq('both').all()
comparison['net_improvement']=comparison.net_pnl_ATR-comparison.net_pnl_baseline
features=pd.read_csv(R/'five_year_trade_research/twelve_trade_features.csv',parse_dates=['entry_date'])
comparison=comparison.merge(features[['entry_date','trade_id']],on='entry_date',how='left')
comparison.to_csv(P/'paired_trade_comparison.csv',index=False,encoding='utf-8-sig')
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
summary=dict(start='2021-10-12',end='2026-10-12',days=1212,initial_capital=22250,ATR_period=14,multiplier=3,entry_ADX_threshold=20,stop_price_tick=1,baseline_matches_saved=True,source_unchanged=True,prefix_checks=checks,results=results,roll_transfers=int(logs['ATR3_net'].roll_spread.notna().sum()),same_day_high_used_to_raise_active_stop=False)
(P/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
fig,(ax,ay)=plt.subplots(2,1,figsize=(13,8),sharex=True);fig.subplots_adjust(bottom=.13,hspace=.14)
for name,label,color in [('baseline_net','ADX baseline / MA exit','#467b9c'),('ATR3_net','ADX + ATR14 x 3 stop','#d08b35')]:
    a=arrays[name];ax.plot(a.index,a.nav,label=label,color=color);v=np.r_[1,a.nav];dd=v/np.maximum.accumulate(v)-1;ay.plot(a.index,dd[1:]*100,color=color)
ax.set(title='Corn — Fixed ATR14 x 3 Exit Experiment (In Sample)',ylabel='NAV');ax.legend(frameon=False);ay.set(ylabel='Drawdown %',xlabel='Trading date')
for axis in [ax,ay]:axis.grid(alpha=.2)
ay.xaxis.set_major_locator(mdates.MonthLocator(interval=6));ay.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.text(.01,.025,'Stops use prior information; gap exits at open; no same-day ratchet; roll levels translated at observed opens. Exploratory only.',fontsize=9)
fig.savefig(P/'ATR3_nav_drawdown.png',dpi=170);plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2));print(comparison[['trade_id','entry_date','exit_date_baseline','exit_date_ATR','exit_reason_ATR','net_pnl_baseline','net_pnl_ATR','net_improvement']].to_string(index=False))
