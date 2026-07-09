# ============================================================
# ambil_unik.py — SUMBER DEDUP TUNGGAL
# Output:
#   TB_teks_unik.xlsx  (ID_unik, teks, n_baris)  <- master
#   TB_peta_baris.xlsx (ID_unik, teks)           <- propagasi ke baris
# ============================================================
import pandas as pd
import re
from config import DATA_DIR

FILE_INPUT   = DATA_DIR / 'Ro Data.xlsx'
PUNYA_HEADER = True
OUT_MASTER   = DATA_DIR / 'TB_teks_unik.xlsx'
OUT_PETA     = DATA_DIR / 'TB_peta_baris.xlsx'

# ── Normalisasi (HARUS identik di seluruh pipeline) ──
def norm(t):
    t = str(t).lower()
    t = t.replace('\\n', ' ').replace('\n', ' ')
    t = re.sub(r'[-•]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df = pd.read_excel(FILE_INPUT) if PUNYA_HEADER else pd.read_excel(FILE_INPUT, header=None)
KOL = df.columns[0]
df = df[[KOL]].copy(); df.columns = ['teks_asli']

s = df['teks_asli'].astype(str)
mask = df['teks_asli'].notna() & ~s.str.strip().str.lower().isin(['', '-', 'nan', 'none', 'null'])
df = df[mask].reset_index(drop=True)
df['_key'] = df['teks_asli'].apply(norm)

print(f'Baris ada teks : {len(df):,}')
print(f'Teks unik      : {df["_key"].nunique():,}')

# ── Representatif tiap grup = teks asli paling sering (tanpa groupby.apply) ──
def paling_sering(x):
    return x.value_counts().index[0]

rep = df.groupby('_key')['teks_asli'].agg(paling_sering)   # Series: _key -> teks
cnt = df.groupby('_key').size()                            # Series: _key -> n_baris

unik = pd.DataFrame({'teks': rep, 'n_baris': cnt}).reset_index()   # _key jadi kolom
unik = unik.sort_values('n_baris', ascending=False).reset_index(drop=True)
unik['ID_unik'] = [f'U{i+1:03d}' for i in range(len(unik))]

# master (3 kolom bersih; _key tidak ikut)
unik[['ID_unik', 'teks', 'n_baris']].to_excel(OUT_MASTER, index=False)

# peta propagasi ke baris
peta = df.merge(unik[['_key', 'ID_unik']], on='_key', how='left')
peta = peta[['ID_unik', 'teks_asli']].rename(columns={'teks_asli': 'teks'})
peta.to_excel(OUT_PETA, index=False)

print(f'\nMaster : {OUT_MASTER}  ({len(unik):,} teks unik)')
print(f'Peta   : {OUT_PETA}  ({peta["ID_unik"].notna().sum():,} baris terpetakan)')
print('\nCATATAN: is_radiologi (radiologi/artefak) TIDAK ditetapkan di sini;')
print('  diprediksi oleh form(radiolog)/rule-based/LLM sebagai label.')