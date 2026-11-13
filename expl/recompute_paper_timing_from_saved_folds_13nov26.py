"""Recompute paper timing from saved totals; no refitting or changes to raw outputs.
Prediction covers training and test rows in each outer fold. All results must
have complete five-fold timing and positive used-sample counts. Writes derived
CSV files alongside this script; leaves manuscript edits to a separate step.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
root=Path(__file__).resolve().parents[1]
rows=[]
for i,p in enumerate(sorted((root/'results/revision-A').glob('*.joblib'))):
    r=joblib.load(p); fs=r['fold_results']; n=r['n_samples_used']
    assert len(fs)==5 and n>0 and r['config']['n_jobs']==1,p
    row={'model':r['model_name'],'dataset':r['dataset_name'],'n_used':n,'folds':len(fs),'n_prediction_rows':n*len(fs)}
    for key in ['tune_time','train_time','predict_time']:
        assert np.isclose(r[key],sum(f[key] for f in fs)),(p,key)
        assert np.isfinite(r[key]) and r[key]>=0,(p,key)
        row[key]=r[key]
    assert all(np.isfinite(f['test_r2']) and np.isfinite(f['train_r2']) for f in fs),p
    row['predict_ms_1k']=row['predict_time']/row['n_prediction_rows']*1e6
    rows.append(row); del r
    if (i+1)%100==0: print('Checked',i+1,flush=True)
d=pd.DataFrame(rows)
assert len(d)==494 and len(d[d.model=='tabpfn'])==54
out=root/'expl'
d.to_csv(out/'corrected_timing_per_dataset_13nov26.csv',index=False)
a=d.groupby('model').agg(n_datasets=('dataset','size'),tune_s=('tune_time','mean'),train_s=('train_time','mean'),predict_ms_1k=('predict_ms_1k','mean'),predict_median_ms_1k=('predict_ms_1k','median')).sort_values('tune_s')
a.to_csv(out/'corrected_timing_summary_13nov26.csv')
print(a.round(4).to_string(),flush=True)
print('EXAMPLES\n',d[d.dataset.isin(['concrete_strength','california_housing','superconduct'])][['dataset','model','tune_time','train_time']].round(3).to_string(index=False),flush=True)
print('COMPLETE',flush=True)
