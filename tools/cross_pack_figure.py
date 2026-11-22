"""Three-pack summary chart -> packs/cross_pack_summary.png (run tools/cross_pack.py first)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
J=json.load(open(os.path.join(ROOT,'packs','cross_pack.json')))
P=['P45B','NCR','LFP']; LAB={'P45B':'Molicel P45B','NCR':'Panasonic NCR18650BD','LFP':'A123 ANR26650 (LFP)'}
C={'P45B':'#2a78d6','NCR':'#eb6834','LFP':'#1baf7a'}; M=['Eco','Normal','Sport']
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#888","axes.labelcolor":"#222"})
fig,ax=plt.subplots(1,3,figsize=(15,4.3),gridspec_kw=dict(width_ratios=[1.25,1.25,0.9]))
def grouped(a,get,title):
    w=0.13; x=np.arange(3)
    for i,p in enumerate(P):
        s,g=zip(*[get(p,m) for m in M])
        off=(i-1)*2*w*1.08
        a.bar(x+off-w/2,s,w,color='white',edgecolor=C[p],hatch='///',lw=1.2)
        b=a.bar(x+off+w/2,g,w,color=C[p],edgecolor='white',lw=1)
        for xx,v in zip(x+off+w/2,g): a.text(xx,v*1.12,f"{v:.2f}" if v<10 else f"{v:.1f}",ha='center',va='bottom',fontsize=7.5,rotation=90,color='#333')
    a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels(M); a.set_ylabel('SOC RMSE (%)'); a.set_title(title,fontsize=11.5)
    a.grid(axis='y',alpha=.25,which='both'); a.set_ylim(0.015,150)
grouped(ax[0],lambda p,m:(J[p+'_t3']['step_Static'][m],J[p+'_t3']['step_Gated'][m]),'(a) Matched plant, step ageing')
tp=lambda p,m:[(r['Static'],r['Gated_mean']) for r in J[p+'_t7'] if r['Ageing']=='step' and r['Mode']==m][0]
grouped(ax[1],tp,'(b) Table plant, 4 cycles, step ageing')
a=ax[2]; x=np.arange(2); w=0.25
for i,p in enumerate(P):
    r=[d for d in J[p+'_t6'] if d['Rule']=='2of3' and d['Tau']==3.5][0]
    v=[r['FalseAlarm_pct'],r['Detection_pct']]
    a.bar(x+(i-1)*w*1.08,v,w,color=C[p],edgecolor='white')
    for xx,vv in zip(x+(i-1)*w*1.08,v): a.text(xx,vv+1.5,f"{vv:.1f}",ha='center',fontsize=8,color='#333')
a.set_xticks(x); a.set_xticklabels(['False alarm,\nhealthy pack','Detection,\naged pack']); a.set_ylabel('Rate (%)'); a.set_ylim(0,112)
a.set_title('(c) Detector, 2 of 3, $\\tau$ = 3.5',fontsize=11.5); a.grid(axis='y',alpha=.25)
h=[Patch(color=C[p],label=LAB[p]) for p in P]+[Patch(facecolor='white',edgecolor='#555',hatch='///',label='Static BMS'),Patch(color='#555',label='Gated BMS')]
fig.legend(handles=h,loc='upper center',ncol=5,frameon=False,bbox_to_anchor=(0.5,1.02))
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(os.path.join(ROOT,'packs','cross_pack_summary.png'),dpi=300)
