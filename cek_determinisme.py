# ============================================================
# cek_determinisme.py — Uji determinisme LLM (repeat 1 vs repeat 2).
# Baca LLM_predictions_repeat.xlsx (hasil llm_harness_repeat.py, REPEAT=2).
# Bandingkan prediksi label antar-repeat per (ID_unik, model).
# TIDAK menyentuh ground truth — ini murni stabilitas output model.
# Output: konsol + Cek_determinisme.xlsx (ringkasan + daftar sel berubah).
# ============================================================
import pandas as pd
import numpy as np
from config import DATA_DIR

INPUT = DATA_DIR / 'LLM_predictions_repeat.xlsx'
OUT   = DATA_DIR / 'Cek_determinisme.xlsx'

LABELS = ['is_radiologi','imp_tb','infiltrat','nodul','kalsifikasi','fibrotik','kavitas','kardiomegali']

def _norm(v):
    return np.nan if pd.isna(v) else str(v).strip().lower()

df = pd.read_excel(INPUT, sheet_name='Predictions')
for l in LABELS:
    if l in df.columns: df[l] = df[l].map(_norm)

reps = sorted(df['repeat'].unique())
assert len(reps) >= 2, f'Butuh >=2 repeat, ditemukan: {reps}. Jalankan llm_harness_repeat.py (REPEAT=2).'
r1, r2 = reps[0], reps[1]

ringkas, berubah = [], []
for model in df['model'].unique():
    d1 = df[(df['model']==model) & (df['repeat']==r1)].set_index('ID_unik')
    d2 = df[(df['model']==model) & (df['repeat']==r2)].set_index('ID_unik')
    ids = d1.index.intersection(d2.index)
    total = same = 0
    # konsistensi JSON valid
    jv1 = d1.loc[ids,'json_valid'] if 'json_valid' in d1.columns else pd.Series(True,index=ids)
    jv2 = d2.loc[ids,'json_valid'] if 'json_valid' in d2.columns else pd.Series(True,index=ids)
    for l in LABELS:
        if l not in d1.columns or l not in d2.columns: continue
        a = d1.loc[ids,l]; b = d2.loc[ids,l]
        eq = (a.values == b.values) | (a.isna().values & b.isna().values)
        total += len(ids); same += int(eq.sum())
        for i, e in zip(ids, eq):
            if not e:
                berubah.append({'model':model,'ID_unik':i,'label':l,
                                f'repeat{r1}':d1.loc[i,l], f'repeat{r2}':d2.loc[i,l]})
    pct = 100*same/total if total else np.nan
    ringkas.append({'model':model,'n_teks':len(ids),'n_sel_dibanding':total,
                    'n_identik':same,'n_berubah':total-same,
                    'persen_identik':round(pct,2),
                    'json_valid_sama': int((jv1.values==jv2.values).sum())})

df_ring = pd.DataFrame(ringkas)
df_ber  = pd.DataFrame(berubah)

# ringkas keseluruhan
tot = df_ring['n_sel_dibanding'].sum(); sm = df_ring['n_identik'].sum()
overall = round(100*sm/tot,2) if tot else np.nan

print('='*60)
print(f'CEK DETERMINISME — repeat {r1} vs {r2}')
print('='*60)
print(df_ring.to_string(index=False))
print(f'\nKESELURUHAN: {sm}/{tot} sel identik = {overall}%')
if len(df_ber):
    print(f'\nSel berubah ({len(df_ber)}):')
    print(df_ber.to_string(index=False))
else:
    print('\nTidak ada sel berubah — output identik 100% antar-repeat.')

with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    df_ring.to_excel(w, sheet_name='Ringkasan', index=False)
    (df_ber if len(df_ber) else pd.DataFrame(columns=['model','ID_unik','label',f'repeat{r1}',f'repeat{r2}'])
     ).to_excel(w, sheet_name='SelBerubah', index=False)
    pd.DataFrame([{'metrik':'persen_identik_keseluruhan','nilai':overall},
                  {'metrik':'n_sel_dibanding','nilai':int(tot)},
                  {'metrik':'repeat_dibanding','nilai':f'{r1} vs {r2}'}]
                 ).to_excel(w, sheet_name='Overall', index=False)
print(f'\nTersimpan: {OUT}')
