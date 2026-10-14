from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).parent
full=pd.read_csv(P/'causal_signal_and_contracts.csv',parse_dates=['date']).set_index('date')
source=(P/'run.py').read_text().split("signal.to_csv")[0]
source=source.replace("pd.DataFrame(missing).to_csv(P/'missing_dates.csv',index=False,encoding='utf-8-sig')",'pass')
tests=[]
for cutoff in ['2022-12-30','2024-12-31','2025-12-31']:
    part=source.replace('calendar=pd.DatetimeIndex',f"c=c.loc[:'{cutoff}'];data={{s:d.loc[:'{cutoff}'] for s,d in data.items() if not d.loc[:'{cutoff}'].empty}}\ncalendar=pd.DatetimeIndex")
    ns={'__file__':str(P/'run.py')}
    exec(compile(part,'prefix-check','exec'),ns)
    a=ns['signal'];b=full.loc[a.index]
    assert (a.contract==b.contract).all()
    for field in ['index_close','ma20','ma60','signal','target']:
        assert np.allclose(a[field],b[field],equal_nan=True)
    tests.append(dict(cutoff=cutoff,rows=len(a),prefix_invariant=True))
# Overlapping strategy trades match earlier two-year corrected trades.
new=pd.read_csv(P/'corrected_net_cycles.csv')
old=pd.read_csv(P.parent/'attribution/trade_attribution.csv')
recent=new[new.entry_date>='2025-01-07'].reset_index(drop=True)
assert len(recent)==len(old)==4
for field in ['entry_date','exit_date','contracts','net_pnl','gross_pnl']:
    assert recent[field].tolist()==old[field].tolist()
rolls=full[full.roll].copy();rolls['decision_date']=full.index.to_series().shift().loc[rolls.index]
rolls.to_csv(P/'all_contract_switches.csv',encoding='utf-8-sig')
result=dict(prefix_tests=tests,two_year_overlapping_trades_identical=True,total_routing_switches_including_warmup=len(rolls),evaluation_switches=int((rolls.index>=pd.Timestamp('2021-10-12')).sum()))
(P/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
