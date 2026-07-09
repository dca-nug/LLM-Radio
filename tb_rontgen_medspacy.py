# ============================================================
# tb_rontgen_medspacy.py — v4 (lengan RULE-BASED)
# Sumber tunggal: TB_teks_unik.xlsx (dari ambil_unik.py). Dikunci ke ID_unik.
#
# LABEL yang diprediksi:
#   is_radiologi {radiologi, artefak}   <- gate; artefak = bukan bacaan foto
#   imp_tb       {aktif, suspek, inaktif, tidak}
#   temuan       {ada, tidak} -> infiltrat, nodul, kalsifikasi, fibrotik, kavitas
#   kardiomegali {ada, tidak}
#   dx_category  : turunan (DIHITUNG)
#
# Catatan: is_radiologi=artefak hanya ~9 teks -> DESKRIPTIF, bukan endpoint ber-CI.
#   Rule-based mengenali artefak hanya jika penandanya terdaftar (keterbatasan,
#   harus disebut di paper).
#
# Input : TB_teks_unik.xlsx (ID_unik, teks, n_baris) ; TB_peta_baris.xlsx (opsional)
# Output: RuleBased_pred.xlsx
# Install: pip install medspacy openpyxl pandas rapidfuzz
# ============================================================

import os
import pandas as pd
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule
from rapidfuzz import process, fuzz
from config import DATA_DIR

INPUT_UNIK = DATA_DIR / 'TB_teks_unik.xlsx'
INPUT_PETA = DATA_DIR / 'TB_peta_baris.xlsx'
OUTPUT     = DATA_DIR / 'RuleBased_pred.xlsx'

# ── PENANDA ARTEFAK (non-bacaan). Deterministik, terdaftar eksplisit. ──
ARTIFACT_MARKERS = [
    'hamil', 'menolak', 'tolak', 'tdl', 'tidak dilakukan', 'tdk dilakukan',
    'sudah mendapatkan tpt', 'tpt 6h', 'tpt 3hp', 'sedang dalam pengobatan',
    'sedang pengobatan', 'sedang pengpbatan', 'sudah rogsent', 'sudah rontgen',
    'di parahita', 'hanya di cek tcm', 'cek tcm',
]

# ── LEXICON — entitas hanya menandai dirinya. Temuan TIDAK ber-cap TB. ──
IMPRESSION_RULES = [
    ('tb aktif','IMPRESSION','imp_tb_aktif'),('tbc aktif','IMPRESSION','imp_tb_aktif'),
    ('tuberculosis aktif','IMPRESSION','imp_tb_aktif'),('kp aktif','IMPRESSION','imp_tb_aktif'),
    ('koch pulmonum aktif','IMPRESSION','imp_tb_aktif'),('tb paru aktif','IMPRESSION','imp_tb_aktif'),
    ('gambaran tb aktif','IMPRESSION','imp_tb_aktif'),('proses spesifik aktif','IMPRESSION','imp_tb_aktif'),
    ('tb pulmo','IMPRESSION','imp_tb_aktif'),
    # "lama aktif" = TB kronis yg MASIH aktif -> aktif (tertulis 'aktif'). Cegah 'lama'->inaktif.
    ('tb lama aktif','IMPRESSION','imp_tb_aktif'),('tbc lama aktif','IMPRESSION','imp_tb_aktif'),
    ('tuberculosis lama aktif','IMPRESSION','imp_tb_aktif'),('kp lama aktif','IMPRESSION','imp_tb_aktif'),
    ('tb lama','IMPRESSION','imp_tb_inaktif'),('tbc lama','IMPRESSION','imp_tb_inaktif'),
    ('tuberculosis lama','IMPRESSION','imp_tb_inaktif'),('kp lama','IMPRESSION','imp_tb_inaktif'),
    ('tb inaktif','IMPRESSION','imp_tb_inaktif'),('post tb','IMPRESSION','imp_tb_inaktif'),
    ('bekas tb','IMPRESSION','imp_tb_inaktif'),('scar tb','IMPRESSION','imp_tb_inaktif'),
    ('proses spesifik lama','IMPRESSION','imp_tb_inaktif'),
]
FINDING_RULES = [
    ('infiltrat','FINDING','find_infiltrat'),('nodul','FINDING','find_nodul'),
    ('kalsifikasi','FINDING','find_kalsifikasi'),('fibrotik','FINDING','find_fibrotik'),
    ('fibrosis','FINDING','find_fibrotik'),('kavitas','FINDING','find_kavitas'),
    ('kaviti','FINDING','find_kavitas'),('cavitas','FINDING','find_kavitas'),
]
CARDIAC_RULES = [
    ('kardiomegali','CARDIAC','card_kardiomegali'),('cardiomegaly','CARDIAC','card_kardiomegali'),
    ('jantung membesar','CARDIAC','card_kardiomegali'),('pembesaran jantung','CARDIAC','card_kardiomegali'),
    ('ctr meningkat','CARDIAC','card_kardiomegali'),
]
ALL_RULES    = IMPRESSION_RULES + FINDING_RULES + CARDIAC_RULES
LABEL_TO_CAT = {t.lower(): c for t, _, c in ALL_RULES}

FINDING_COLS = ['find_infiltrat','find_nodul','find_kalsifikasi','find_fibrotik','find_kavitas']
FIND_ALL     = FINDING_COLS + ['card_kardiomegali']
OUT_COLS     = ['is_radiologi','imp_tb'] + FINDING_COLS + ['card_kardiomegali','dx_category','match_detail']

# ── MEDSPACY ──
nlp = medspacy.load(enable=['medspacy_target_matcher','medspacy_context'])
nlp.get_pipe('medspacy_target_matcher').add([TargetRule(t,l) for t,l,_ in ALL_RULES])
nlp.get_pipe('medspacy_context').add([
    ConTextRule('tidak','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tak','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tanpa','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tidak tampak','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tak tampak','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tidak ada','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tidak ditemukan','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('tidak terlihat','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('menyingkirkan','NEGATED_EXISTENCE',direction='FORWARD'),
    ConTextRule('curiga','UNCERTAIN',direction='FORWARD'),
    ConTextRule('kemungkinan','UNCERTAIN',direction='FORWARD'),
    ConTextRule('suspek','UNCERTAIN',direction='FORWARD'),
    ConTextRule('suspect','UNCERTAIN',direction='FORWARD'),
    ConTextRule('sugestif','UNCERTAIN',direction='FORWARD'),
    ConTextRule('mengarah ke','UNCERTAIN',direction='FORWARD'),
    ConTextRule('dd','UNCERTAIN',direction='FORWARD'),
    ConTextRule('riwayat','HISTORICAL',direction='FORWARD'),
    ConTextRule('bekas','HISTORICAL',direction='FORWARD'),
    ConTextRule('post','HISTORICAL',direction='FORWARD'),
    # 'lama' TIDAK dijadikan modifier historis: 'tb lama' sudah rule inaktif langsung,
    # dan 'lama' sbg modifier akan salah menurunkan 'tb lama aktif' jadi inaktif.
])

FUZZY_THRESHOLD = 88
KEYWORD_LIST = [t.lower() for t,_,_ in ALL_RULES]
NEG_KATA  = ['tidak','tak','tanpa','tidak tampak','tak tampak','tidak ada','tidak ditemukan']
UNC_KATA  = ['curiga','kemungkinan','suspek','sugestif','mengarah','dd']
HIST_KATA = ['riwayat','bekas','post']

def _apply(cat, neg, unc, hist, state, out):
    if cat.startswith('imp_'):
        if neg: return
        if unc: state['suspek'] = True
        elif hist and cat == 'imp_tb_aktif': state['inaktif'] = True
        elif cat == 'imp_tb_aktif': state['aktif'] = True
        else: state['inaktif'] = True
    else:
        if not neg: out[cat] = 'ada'

def kategorisasi_teks(teks):
    out = {c:'tidak' for c in FIND_ALL}
    out['is_radiologi']='radiologi'; out['imp_tb']='tidak'
    out['dx_category']=None; out['match_detail']=''
    if pd.isna(teks) or str(teks).strip()=='':
        out['is_radiologi']='artefak'; out['dx_category']='artefak'
        out['match_detail']='kosong'; return out

    t = str(teks).lower().strip()

    # GATE: artefak (non-bacaan) — cek penanda lebih dulu
    if any(m in t for m in ARTIFACT_MARKERS):
        out['is_radiologi']='artefak'; out['dx_category']='artefak'
        hit=[m for m in ARTIFACT_MARKERS if m in t]
        out['match_detail']='artefak:'+','.join(hit)
        return out

    # Bacaan valid -> ekstraksi normal
    state = {'aktif':False,'suspek':False,'inaktif':False}
    detail, matched = [], set()
    for ent in nlp(t).ents:
        cat = LABEL_TO_CAT.get(ent.text.lower())
        if not cat: continue
        neg,unc,hist = ent._.is_negated, ent._.is_uncertain, ent._.is_historical
        _apply(cat,neg,unc,hist,state,out); matched.add(ent.text.lower())
        detail.append(f'{cat}:{"NEG" if neg else "UNC" if unc else "HIST" if hist else "YES"}:exact')
    words = t.split()
    for w in range(1,4):
        for i in range(len(words)-w+1):
            chunk=' '.join(words[i:i+w])
            if any(chunk in m or m in chunk for m in matched): continue
            m=process.extractOne(chunk,KEYWORD_LIST,scorer=fuzz.ratio,score_cutoff=FUZZY_THRESHOLD)
            if not m: continue
            cat=LABEL_TO_CAT[m[0]]; prefix=' '.join(words[max(0,i-3):i])
            neg=any(k in prefix for k in NEG_KATA); unc=any(k in prefix for k in UNC_KATA)
            hist=any(k in prefix for k in HIST_KATA)
            _apply(cat,neg,unc,hist,state,out); matched.add(chunk)
            detail.append(f'{cat}:fuzzy("{chunk}"->"{m[0]}",{m[1]:.0f})')
    out['imp_tb']=('aktif' if state['aktif'] else 'suspek' if state['suspek']
                   else 'inaktif' if state['inaktif'] else 'tidak')
    any_abn=any(out[c]=='ada' for c in FINDING_COLS) or out['card_kardiomegali']=='ada'
    out['dx_category']={'aktif':'tb_aktif','suspek':'tb_suspek','inaktif':'tb_inaktif'}.get(
        out['imp_tb'],'non_tb_abnormal' if any_abn else 'normal')
    out['match_detail']=' | '.join(detail)
    return out

# ── LOAD master ──
u = pd.read_excel(INPUT_UNIK)               # ID_unik, teks, n_baris
print(f'Teks unik dimuat: {len(u)} (dari {INPUT_UNIK})')

res = u['teks'].apply(kategorisasi_teks)
ru = u.copy()
for c in OUT_COLS:
    ru[c] = res.apply(lambda x: x[c])

# ── COUNT + flag endpoint (>=10) ──
count_rows = [('is_radiologi=radiologi', int((ru['is_radiologi']=='radiologi').sum()), ''),
              ('is_radiologi=artefak',   int((ru['is_radiologi']=='artefak').sum()),   '<10 deskriptif')]
rad = ru[ru['is_radiologi']=='radiologi']        # metrik label lain hanya atas bacaan valid
count_rows.append(('--- imp_tb (atas bacaan valid) ---', len(rad), ''))
for v in ['aktif','suspek','inaktif','tidak']:
    n=int((rad['imp_tb']==v).sum())
    count_rows.append((f'  imp_tb={v}', n, '>=10 endpoint' if n>=10 else '<10 deskriptif'))
for c in FIND_ALL:
    n=int((rad[c]=='ada').sum())
    count_rows.append((c, n, '>=10 endpoint' if n>=10 else '<10 deskriptif'))
df_count=pd.DataFrame(count_rows,columns=['label','n_teks_unik','status'])

def crosstab_by(key):
    xt=pd.DataFrame({c: rad.groupby(key).apply(lambda g:int((g[c]=='ada').sum())) for c in FIND_ALL})
    xt['n_teks']=rad[key].value_counts()
    return xt.fillna(0).astype(int).reset_index()
xt_imp=crosstab_by('imp_tb'); xt_dx=crosstab_by('dx_category')

print(f'\n=== COUNT (total unik={len(ru)}) ===');  print(df_count.to_string(index=False))
print('\n=== CROSS-TAB imp_tb x temuan (bacaan valid) ==='); print(xt_imp.to_string(index=False))
print('\n=== CROSS-TAB dx_category x temuan ==='); print(xt_dx.to_string(index=False))

sheets={'PredUnik':ru,'CountLabel':df_count,'Crosstab_impTB':xt_imp,'Crosstab_dxCat':xt_dx}
if os.path.exists(INPUT_PETA):
    peta=pd.read_excel(INPUT_PETA)
    per_baris=peta.merge(ru.drop(columns=['teks','n_baris'],errors='ignore'),on='ID_unik',how='left')
    sheets['PerBaris_sensitivitas']=per_baris
    print(f'\nPropagasi ke {len(per_baris):,} baris (sensitivitas).')

with pd.ExcelWriter(OUTPUT,engine='openpyxl') as wtr:
    for name,d in sheets.items(): d.to_excel(wtr,sheet_name=name,index=False)
print(f'\nTersimpan: {OUTPUT}. Kunci join: ID_unik')