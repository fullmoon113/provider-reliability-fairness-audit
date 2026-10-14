from pathlib import Path
HERE = Path(__file__).resolve().parent      # src/
ROOT = HERE.parent                          # repository root
DATA = ROOT / 'data'
OUT = ROOT / 'output'; OUT.mkdir(exist_ok=True)

import warnings; warnings.filterwarnings("ignore")
exec(open(HERE/'fpr_class_weighting.py').read().split("oof0 = oof_smote(ext0)")[0])
neg=y==0
oof0=oof_smote(ext0); oofc=oof_cw()
for nm,o in (('SMOTE',oof0),('CW',oofc)):
    print(nm,'fraud rate ext/mid',o[fm&(ext0==1)].mean(),o[fm&(ext0==0)].mean(),'| neg flag ext/mid',o[neg&(ext0==1)].mean(),o[neg&(ext0==0)].mean())
    print('  fraud',fmt(fit(o,fm,ext0)),'|',fmt(fit(o,fm,ext0,True)))
    print('  neg  ',fmt(fit(o,neg,ext0)),'|',fmt(fit(o,neg,ext0,True)))
print(neg.sum(), (neg&(ext0==1)).sum(), (neg&(ext0==0)).sum())
