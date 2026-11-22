"""Collects the numbers compared across P45B, NCR and LFP -> packs/cross_pack.json."""
import pandas as pd, numpy as np, json
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import make_pack_tables as t
from make_pack_tables import pack_dir, ROOT
P={'P45B':'P45B','LFP':'LFP','NCR':'NCR'}
M=['Eco','Normal','Sport']
out={}
for p in P:
    t3=t.table3(p); out[p+'_t3']={f"{r.Ageing}_{r.Case}":{m:round(r[m+'_mean'],3) for m in M} for _,r in t3.iterrows()}
    out[p+'_t3sd']={f"{r.Ageing}_{r.Case}":{m:r[m] for m in M} for _,r in t3.iterrows()}
    t7,fr=t.table7(p); out[p+'_t7']=t7[['Ageing','Mode','Static','Ungated','Gated_mean','Q_est_Ah','Q_true_Ah','SOC_end_pct','T_end']].round(3).to_dict('records'); out[p+'_fresh_tp']=fr.round(3).to_dict()
    t6=t.table6(p); out[p+'_t6']=t6.round(2).to_dict('records')
    t5=t.table5(p); out[p+'_t5']=t5.round(1).to_dict('records')
    out[p+'_t9']=t.table9(p).round(3).to_dict('records')
    out[p+'_val']=t.validation(p).round(4).to_dict('records')
    im=t.imputation(p); out[p+'_imp']=im.round(3).to_dict('records')
    pm=pd.read_csv(os.path.join(pack_dir(p),'bms','results','revision2','detector_per_mode.csv')); out[p+'_det_mode']=pm[(pm.Rule=='2of3')&(pm.Tau==3.5)].to_dict('records')
    pt=t.table4(p); out[p+'_t4']=pt.round(4).to_dict('records')
json.dump(out,open(os.path.join(ROOT,'packs','cross_pack.json'),'w'),indent=0,default=float)
for p in P:
    print('=====',p)
    for k in ['_t3','_fresh_tp']: print(k,out[p+k])
    print(pd.DataFrame(out[p+'_t7']).to_string(index=False))
    print(pd.DataFrame(out[p+'_det_mode'])[['Mode','FalseAlarmRate','DetectionRate','MedianDelay_s']].to_string(index=False))
    print(pd.DataFrame(out[p+'_t6']).to_string(index=False))
    print(pd.DataFrame(out[p+'_t9']).to_string(index=False))
    print(pd.DataFrame(out[p+'_imp']).to_string(index=False))
    print(pd.DataFrame(out[p+'_val']).to_string(index=False))
    t5=pd.DataFrame(out[p+'_t5']); print(t5.to_string(index=False))
