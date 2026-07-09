# ============================================================
# evaluasi.py — Evaluasi komparatif 3 sistem vs ground truth radiolog.
# JUJUR by design: cell tipis dilaporkan deskriptif (bukan F1 ber-CI),
# IAA dijabarkan penuh, prediksi invalid dihitung (tidak disembunyikan),
# tanpa uji signifikansi (deskriptif + CI, sesuai proposal).
#
# Satuan analisis: TEKS UNIK (ID_unik). Bootstrap resample teks unik.
#
# INPUT (semua ber-ID_unik):
#   Form_Anotasi_..._AnotatorA.xlsx  (sheet 'Anotasi')  <- radiolog A
#   Form_Anotasi_..._AnotatorB.xlsx  (sheet 'Anotasi')  <- radiolog B
#   [opsional] Form_Anotasi_Konsensus.xlsx               <- GT final (hasil konsensus)
#   RuleBased_pred.xlsx   (sheet 'PredUnik')
#   LLM_predictions.xlsx  (sheet 'Predictions')
# OUTPUT: Evaluasi_hasil.xlsx (banyak sheet)
#
# Deps: pandas, numpy, openpyxl  (TANPA sklearn — metrik dihitung manual & transparan)
# ============================================================

import os
import numpy as np
import pandas as pd
from config import DATA_DIR

# ── FILE ──
F_ANOT_A = DATA_DIR / 'Annotation A.xlsx'
F_ANOT_B = DATA_DIR / 'Annotation B.xlsx'
F_KONSEN = DATA_DIR / 'Form_anotasi_konsensus.xlsx'   # format LONG (ID_unik, label, Konsensus); baris disagreement saja
F_RULE   = DATA_DIR / 'RuleBased_pred.xlsx'
F_LLM_ZS = DATA_DIR / 'LLM_predictions.xlsx'           # zero-shot
F_LLM_OS = DATA_DIR / 'LLM_predictions_oneshot.xlsx'   # one-shot (OPSIONAL; dilewati bila tak ada)
OUT      = DATA_DIR / 'Evaluasi_hasil.xlsx'

N_BOOT   = 1000
SEED     = 42
POWER_THRESHOLD = 10          # < ini (kelas minoritas) -> DESKRIPTIF, bukan endpoint ber-CI

# ── SKEMA LABEL ──
LABELS = ['is_radiologi','imp_tb','infiltrat','nodul','kalsifikasi','fibrotik','kavitas','kardiomegali']
NOMINAL_CLASSES = {
    'is_radiologi': ['radiologi','artefak'],
    'imp_tb':       ['aktif','suspek','inaktif','tidak'],
}
BINARY_LABELS = ['infiltrat','nodul','kalsifikasi','fibrotik','kavitas','kardiomegali']  # ada/tidak
# Binarisasi klinis (endpoint sekunder): mana yang dianggap "positif".
BIN_POSITIVE = {**{l: 'ada' for l in BINARY_LABELS}, 'imp_tb': 'aktif'}  # imp_tb: TB aktif vs lainnya
DESCRIPTIVE_ONLY = ['is_radiologi']   # n artefak ~9 -> selalu deskriptif

# harmonisasi nama kolom tiap sumber -> nama label kanonik
RULE_MAP = {'find_infiltrat':'infiltrat','find_nodul':'nodul','find_kalsifikasi':'kalsifikasi',
            'find_fibrotik':'fibrotik','find_kavitas':'kavitas','card_kardiomegali':'kardiomegali'}

def _norm(v):
    if pd.isna(v): return np.nan
    return str(v).strip().lower()

def load_annot(path, sheet='Anotasi'):
    xl = pd.ExcelFile(path)
    sh = sheet if sheet in xl.sheet_names else xl.sheet_names[0]   # fallback ke sheet pertama
    d = pd.read_excel(path, sheet_name=sh)
    keep = ['ID_unik'] + [l for l in LABELS if l in d.columns]
    d = d[keep].copy()
    for l in LABELS:
        if l in d.columns: d[l] = d[l].map(_norm)
    return d.set_index('ID_unik')

def load_rule(path):
    d = pd.read_excel(path, sheet_name='PredUnik').rename(columns=RULE_MAP)
    keep = ['ID_unik'] + [l for l in LABELS if l in d.columns]
    d = d[keep].copy()
    for l in LABELS:
        if l in d.columns: d[l] = d[l].map(_norm)
    return d.set_index('ID_unik')

def load_llm(path, model_tag):
    d = pd.read_excel(path, sheet_name='Predictions')
    d = d[d['model'] == model_tag].copy()
    invalid = int((~d['json_valid']).sum()) if 'json_valid' in d.columns else 0
    keep = ['ID_unik'] + [l for l in LABELS if l in d.columns]
    d = d[keep].copy()
    for l in LABELS:
        if l in d.columns: d[l] = d[l].map(_norm)
    return d.set_index('ID_unik'), invalid

# ── METRIK MANUAL (transparan) ──
def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0 or len(cats) < 2: return np.nan
    idx = {c:i for i,c in enumerate(cats)}
    po = np.mean(a == b)
    ea = np.array([np.mean(a==c) for c in cats]); eb = np.array([np.mean(b==c) for c in cats])
    pe = float(np.sum(ea*eb))
    if pe >= 1.0: return np.nan
    return (po - pe) / (1 - pe)

def pct_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.mean(a == b)) if len(a) else np.nan

def f1_per_class(gt, pred, cls):
    tp = np.sum((gt==cls) & (pred==cls)); fp = np.sum((gt!=cls) & (pred==cls))
    fn = np.sum((gt==cls) & (pred!=cls))
    if tp==0 and (fp>0 or fn>0): return 0.0
    if tp==0 and fp==0 and fn==0: return np.nan            # kelas absen
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

def macro_f1(gt, pred, classes):
    fs = [f1_per_class(gt,pred,c) for c in classes]
    fs = [f for f in fs if not np.isnan(f)]
    return float(np.mean(fs)) if fs else np.nan

def binary_stats(gt, pred, pos):
    g = (gt==pos); p = (pred==pos)
    tp=int(np.sum(g&p)); tn=int(np.sum(~g&~p)); fp=int(np.sum(~g&p)); fn=int(np.sum(g&~p))
    sens = tp/(tp+fn) if (tp+fn)>0 else np.nan
    spec = tn/(tn+fp) if (tn+fp)>0 else np.nan
    ppv  = tp/(tp+fp) if (tp+fp)>0 else np.nan
    npv  = tn/(tn+fn) if (tn+fn)>0 else np.nan
    return dict(TP=tp,TN=tn,FP=fp,FN=fn,sens=sens,spec=spec,ppv=ppv,npv=npv)

def bootstrap_ci(gt, pred, fn, n_boot=N_BOOT, seed=SEED):
    gt, pred = np.asarray(gt), np.asarray(pred)
    n = len(gt)
    if n == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                 # resample TEKS UNIK dgn replacement
        v = fn(gt[idx], pred[idx])
        if not np.isnan(v): vals.append(v)
    if not vals: return (np.nan, np.nan)
    return (float(np.percentile(vals,2.5)), float(np.percentile(vals,97.5)))

# ── LOAD ──
A = load_annot(F_ANOT_A); B = load_annot(F_ANOT_B)
rule = load_rule(F_RULE)
SYSTEMS = {'rule_based': rule}
INVALID = {'rule_based': 0}

# zero-shot (wajib ada)
if os.path.exists(F_LLM_ZS):
    g, ig = load_llm(F_LLM_ZS, 'gemma2_generik'); s, isx = load_llm(F_LLM_ZS, 'sahabatai')
    SYSTEMS['gemma_zs'] = g;  SYSTEMS['sahabat_zs'] = s
    INVALID['gemma_zs'] = ig; INVALID['sahabat_zs'] = isx
else:
    print(f'PERINGATAN: {F_LLM_ZS} tidak ada — lengan LLM zero-shot dilewati.')

# one-shot (opsional)
if os.path.exists(F_LLM_OS):
    g, ig = load_llm(F_LLM_OS, 'gemma2_generik'); s, isx = load_llm(F_LLM_OS, 'sahabatai')
    SYSTEMS['gemma_os'] = g;  SYSTEMS['sahabat_os'] = s
    INVALID['gemma_os'] = ig; INVALID['sahabat_os'] = isx
    print('One-shot disertakan (5 sistem dibandingkan).')
else:
    print(f'Catatan: {F_LLM_OS} tidak ada — one-shot dilewati (hanya zero-shot).')

ids = A.index.intersection(B.index)
print(f'ID_unik cocok A&B: {len(ids)} | sistem: {list(SYSTEMS)}')

# ════════════════════════════════════════════════════════════
# 1) INTER-ANNOTATOR AGREEMENT (sebelum konsensus) — DIJABARKAN
# ════════════════════════════════════════════════════════════
iaa_rows = []
for l in LABELS:
    if l not in A.columns or l not in B.columns: continue
    # is_radiologi: IAA atas semua teks; label lain: hanya teks yg KEDUANYA tandai radiologi
    if l == 'is_radiologi':
        sub = ids
    else:
        sub = ids[(A.loc[ids,'is_radiologi']=='radiologi') & (B.loc[ids,'is_radiologi']=='radiologi')]
    a = A.loc[sub, l]; b = B.loc[sub, l]
    m = a.notna() & b.notna()
    a, b = a[m].values, b[m].values
    iaa_rows.append({'label': l, 'n': len(a),
                     'kappa': round(cohen_kappa(a,b),3) if len(a) else np.nan,
                     'persen_setuju': round(pct_agreement(a,b),3) if len(a) else np.nan})
df_iaa = pd.DataFrame(iaa_rows)

# daftar ketidaksepakatan (untuk resolusi konsensus manual)
disagree = []
for l in LABELS:
    if l not in A.columns or l not in B.columns: continue
    for i in ids:
        va, vb = A.loc[i,l], B.loc[i,l]
        if pd.notna(va) and pd.notna(vb) and va != vb:
            disagree.append({'ID_unik': i, 'label': l, 'anotator_A': va, 'anotator_B': vb})
df_disagree = pd.DataFrame(disagree)

# ════════════════════════════════════════════════════════════
# 2) GROUND TRUTH POST-KONSENSUS (GT tunggal, 107 record)
#    Basis = sel yang A&B SEPAKAT. Sel disagreement DITIMPA nilai dari
#    Form_anotasi_konsensus.xlsx (format LONG: ID_unik, label, Konsensus;
#    isinya HANYA baris disagreement).
# ════════════════════════════════════════════════════════════
GT = A.copy()
for l in LABELS:
    if l in A.columns and l in B.columns:
        GT[l] = np.where(A[l] == B[l], A[l], np.nan)   # sepakat -> nilai; beda -> NaN (akan ditimpa konsensus)

if os.path.exists(F_KONSEN):
    kon = pd.read_excel(F_KONSEN)                       # ID_unik, label, ..., Konsensus (long, disagreement saja)
    kon.columns = [str(c).strip() for c in kon.columns]
    kol_kon = 'Konsensus' if 'Konsensus' in kon.columns else kon.columns[-1]
    n_patch = 0
    for _, r in kon.iterrows():
        idu = r['ID_unik']; lab = str(r['label']).strip()
        if idu in GT.index and lab in GT.columns:
            GT.at[idu, lab] = _norm(r[kol_kon]); n_patch += 1
    sisa_nan = int(GT[[l for l in LABELS if l in GT.columns]].isna().sum().sum())
    gt_note = (f'konsensus final: {n_patch} sel disagreement ditimpa dari {F_KONSEN}. '
               f'Sisa sel kosong (belum tercakup konsensus): {sisa_nan}.')
else:
    gt_note = ('PERINGATAN: file konsensus belum ada. GT hanya sel yang DISEPAKATI A&B; '
               'sel disagreement masih KOSONG (dikeluarkan dari metrik). '
               'Sediakan Form_anotasi_konsensus.xlsx utk hasil final.')
print(gt_note)

# ════════════════════════════════════════════════════════════
# 3) METRIK SISTEM vs GT  (primer: macro-F1 + CI ; sekunder klinis: sens/spec/PPV/NPV)
# ════════════════════════════════════════════════════════════
prim_rows, clin_rows = [], []
for l in LABELS:
    if l in DESCRIPTIVE_ONLY: continue
    classes = NOMINAL_CLASSES.get(l, ['ada','tidak'])
    # baris valid utk GT label ini: hanya teks radiologi (buang artefak) & GT tidak kosong
    base = GT.index
    if 'is_radiologi' in GT.columns:
        base = GT.index[GT['is_radiologi']=='radiologi']
    for sname, S in SYSTEMS.items():
        common = base.intersection(S.index)
        g = GT.loc[common, l]; p = S.loc[common, l] if l in S.columns else pd.Series(index=common, dtype=object)
        m = g.notna()                          # GT harus terisi
        n_eval = int(m.sum())
        n_nopred = int(p[m].isna().sum())      # sistem tak memprediksi (mis. LLM invalid)
        gg = g[m].values
        pp = p[m].fillna('__NOPRED__').values  # NOPRED dihitung sbg salah (konservatif, tidak menguntungkan model)
        # power: kelas minoritas di GT
        minority = min([int(np.sum(gg==c)) for c in classes if np.sum(gg==c)>0], default=0)
        powered = minority >= POWER_THRESHOLD
        mf1 = macro_f1(gg, pp, classes)
        acc = float(np.mean(gg==pp)) if n_eval else np.nan
        kap = cohen_kappa(gg, pp)
        if powered:
            lo,hi = bootstrap_ci(gg, pp, lambda a,b: macro_f1(a,b,classes))
            ci = f'[{lo:.2f}, {hi:.2f}]'
        else:
            ci = 'DESKRIPTIF (n<{}, CI tidak dilaporkan)'.format(POWER_THRESHOLD)
        prim_rows.append({'label':l,'sistem':sname,'n_eval':n_eval,'n_minoritas_GT':minority,
                          'powered':powered,'akurasi':round(acc,3),
                          'macro_F1':round(mf1,3) if not np.isnan(mf1) else np.nan,
                          'macroF1_CI95':ci,'kappa_vs_GT':round(kap,3) if not np.isnan(kap) else np.nan,
                          'n_tanpa_prediksi':n_nopred})
        # sekunder klinis: binarisasi
        pos = BIN_POSITIVE.get(l)
        if pos is not None:
            bs = binary_stats(gg, pp, pos)
            clin_rows.append({'label':l,'positif':pos,'sistem':sname,'n_eval':n_eval,
                              'n_positif_GT':int(np.sum(gg==pos)),'powered':powered,
                              **{k:(round(v,3) if isinstance(v,float) and not np.isnan(v) else v)
                                 for k,v in bs.items()}})
df_prim = pd.DataFrame(prim_rows)
df_clin = pd.DataFrame(clin_rows)

# ════════════════════════════════════════════════════════════
# 4) is_radiologi — DESKRIPTIF (n artefak ~9). Tabel hitungan, bukan CI.
# ════════════════════════════════════════════════════════════
rad_rows = []
if 'is_radiologi' in GT.columns:
    base = GT.index[GT['is_radiologi'].notna()]
    n_art_gt = int((GT.loc[base,'is_radiologi']=='artefak').sum())
    for sname, S in SYSTEMS.items():
        common = base.intersection(S.index)
        g = GT.loc[common,'is_radiologi']; p = S.loc[common,'is_radiologi'] if 'is_radiologi' in S.columns else pd.Series(index=common,dtype=object)
        art = g=='artefak'
        benar_art = int(np.sum(art & (p=='artefak')))
        rad_rows.append({'sistem':sname,'artefak_GT':int(art.sum()),
                         'artefak_dikenali_benar':benar_art,
                         'artefak_terlewat':int(art.sum())-benar_art,
                         'radiologi_salah_dicap_artefak':int(np.sum((g=='radiologi')&(p=='artefak')))})
df_rad = pd.DataFrame(rad_rows)

# ════════════════════════════════════════════════════════════
# 5) ERROR ANALYSIS — per teks, di mana sistem menyimpang dari GT (untuk dibaca manual)
# ════════════════════════════════════════════════════════════
err_rows = []
base = GT.index[GT['is_radiologi']=='radiologi'] if 'is_radiologi' in GT.columns else GT.index
eval_labels = [l for l in LABELS if l not in DESCRIPTIVE_ONLY]
for i in base:
    row = {'ID_unik': i}
    nmis = 0; which = []
    for l in eval_labels:
        gv = GT.loc[i,l] if l in GT.columns else np.nan
        row[f'GT_{l}'] = gv
        for sname,S in SYSTEMS.items():
            pv = S.loc[i,l] if (i in S.index and l in S.columns) else np.nan
            row[f'{sname}_{l}'] = pv
            if pd.notna(gv) and pd.notna(pv) and gv!=pv:
                nmis += 1; which.append(f'{sname}:{l}')
    row['n_mismatch'] = nmis; row['mismatch_detail'] = ' | '.join(which)
    err_rows.append(row)
df_err = pd.DataFrame(err_rows).sort_values('n_mismatch', ascending=False)

# ════════════════════════════════════════════════════════════
# 6) RUN INFO / kejujuran
# ════════════════════════════════════════════════════════════
info = [
    ('n_teks_unik_dievaluasi', len(ids)),
    ('sistem_dibandingkan', ', '.join(SYSTEMS.keys())),
    ('ground_truth', gt_note),
    ('bootstrap_iter', N_BOOT), ('seed', SEED),
    ('power_threshold_endpoint', POWER_THRESHOLD),
    ('label_deskriptif', ', '.join(DESCRIPTIVE_ONLY) + ' (n artefak kecil)'),
]
for sname, nv in INVALID.items():
    info.append((f'prediksi_invalid_{sname}', nv))
info += [
    ('kebijakan_no_prediction', 'dihitung sebagai SALAH (konservatif, tidak menguntungkan model)'),
    ('uji_signifikansi', 'TIDAK dilakukan (deskriptif + CI, sesuai proposal)'),
    ('binarisasi_imp_tb', f"positif = {BIN_POSITIVE['imp_tb']} (TB aktif); ubah di BIN_POSITIVE bila perlu"),
    ('CATATAN', 'macro-F1 & CI hanya utk label powered (n_minoritas>=threshold). '
                'Label underpowered dilaporkan apa adanya + ditandai, TANPA CI.'),
]
df_info = pd.DataFrame(info, columns=['item','nilai'])

with pd.ExcelWriter(OUT, engine='openpyxl') as w:
    df_info.to_excel(w, sheet_name='RunInfo', index=False)
    df_iaa.to_excel(w, sheet_name='IAA_antar_anotator', index=False)
    df_disagree.to_excel(w, sheet_name='Disagreements', index=False)
    df_prim.to_excel(w, sheet_name='Metrik_primer_macroF1', index=False)
    df_clin.to_excel(w, sheet_name='Metrik_klinis_sens_spec', index=False)
    df_rad.to_excel(w, sheet_name='is_radiologi_deskriptif', index=False)
    df_err.to_excel(w, sheet_name='ErrorAnalysis', index=False)
print(f'\nTersimpan: {OUT}')
print('\nIAA:'); print(df_iaa.to_string(index=False))
print('\nMetrik primer:'); print(df_prim.to_string(index=False))