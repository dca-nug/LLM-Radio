# ============================================================
# llm_harness_repeat.py — CEK DETERMINISME. Sama dgn llm_harness.py TAPI REPEAT=2.
# Output terpisah (LLM_predictions_repeat.xlsx) supaya TIDAK menimpa hasil utama.
# Setelah jalan, bandingkan dgn cek_determinisme.py.
# Sumber tunggal: TB_teks_unik.xlsx (ID_unik, teks). Dikunci ke ID_unik.
#
# Dua model (parameter dekoding identik):
#   gemma2:9b-instruct-q4_K_M            (generik)
#   gemma2-9b-cpt-sahabatai-v1-instruct  (Sahabat-AI, teradaptasi ID)
#
# LABEL (skema JSON identik dgn form & rule-based):
#   is_radiologi {radiologi, artefak}  -> gate: bukan bacaan foto = artefak
#   imp_tb {aktif,suspek,inaktif,tidak} + 5 temuan {ada,tidak} + kardiomegali
#
# Prinsip: prompt DIBEKUKAN (PROMPT_HASH dicatat), single pass, harness hanya
#   baca teks (tidak pernah menyentuh ground truth), temperature=0/seed=42/num_ctx=4096.
#
# Install: pip install ollama openpyxl pandas
# Jalankan: Ollama tray aktif (JANGAN `ollama serve` manual).
# ============================================================

import json, time, hashlib, subprocess, platform
from datetime import datetime
import pandas as pd
import ollama
from config import DATA_DIR

INPUT_XLSX  = DATA_DIR / 'TB_teks_unik.xlsx'    # ID_unik, teks
OUTPUT_XLSX = DATA_DIR / 'LLM_predictions_repeat.xlsx'

MODELS = [
    ('gemma2_generik', 'gemma2:9b-instruct-q4_K_M'),
    ('sahabatai',      'sahabatgemma-fix'),   # hasil ollama create (template Gemma2 diperbaiki)
]
OPTIONS = {'temperature': 0, 'seed': 42, 'num_ctx': 4096}
REPEAT  = 2

FINDING_COLS = ['infiltrat','nodul','kalsifikasi','fibrotik','kavitas']
LABEL_COLS   = ['is_radiologi','imp_tb'] + FINDING_COLS + ['kardiomegali']

# ── SKEMA JSON — urutan properti = urutan generasi (CoT dulu, lalu gate, lalu label) ──
SCHEMA = {
    'type': 'object',
    'properties': {
        'alasan':       {'type': 'string'},
        'is_radiologi': {'type': 'string', 'enum': ['radiologi', 'artefak']},
        'imp_tb':       {'type': 'string', 'enum': ['aktif', 'suspek', 'inaktif', 'tidak']},
        'infiltrat':    {'type': 'string', 'enum': ['ada', 'tidak']},
        'nodul':        {'type': 'string', 'enum': ['ada', 'tidak']},
        'kalsifikasi':  {'type': 'string', 'enum': ['ada', 'tidak']},
        'fibrotik':     {'type': 'string', 'enum': ['ada', 'tidak']},
        'kavitas':      {'type': 'string', 'enum': ['ada', 'tidak']},
        'kardiomegali': {'type': 'string', 'enum': ['ada', 'tidak']},
    },
    'required': ['alasan','is_radiologi','imp_tb','infiltrat','nodul','kalsifikasi',
                 'fibrotik','kavitas','kardiomegali'],
}

# ── PROMPT — DIBEKUKAN (revisi zero-shot final). Definisi = protokol anotasi. ──
SYSTEM_PROMPT = """Anda mengekstraksi informasi terstruktur dari SATU catatan hasil skrining tuberkulosis (TB) berbahasa Indonesia. Nilai HANYA apa yang TERTULIS.

Pertama tentukan is_radiologi:
- "radiologi": teks adalah bacaan/kesan foto toraks, TERMASUK bila hasilnya NORMAL atau tanpa kelainan. Hasil foto yang normal tetap sebuah bacaan radiologi (jadi "radiologi"), BUKAN artefak.
- "artefak"  : teks sama sekali BUKAN deskripsi foto toraks, melainkan catatan non-radiologi seperti status/kondisi pasien, penolakan pemeriksaan, pemeriksaan tidak dilakukan, status pengobatan/terapi, atau catatan administratif.
Aturan: tandai "artefak" HANYA bila tidak ada deskripsi foto sama sekali. Jika ragu, pilih "radiologi".
Bila "artefak", isi semua field lain dengan nilai default ("tidak").

Bila "radiologi", isi:

imp_tb (impresi TB, pilih SATU). Tentukan HANYA dari kata pada kalimat kesan/impresi. JANGAN menyimpulkan aktivitas dari pola atau lokasi temuan; bila kata aktivitas tidak tertulis, jangan mengarang.
- "aktif"   : tertulis kata "aktif" (bila tertulis "aktif", pilih aktif walau ada kata "lama").
- "suspek"  : TB dinyatakan ragu/kemungkinan (mis. suspek/curiga/DD TB).
- "inaktif" : tertulis lama/bekas/post/scar TANPA kata "aktif".
- "tidak"   : tidak ada pernyataan impresi TB, atau TB disingkirkan; termasuk bila hanya ada temuan tanpa pernyataan aktivitas TB.

Temuan radiografik (masing-masing "ada"/"tidak"): infiltrat, nodul, kalsifikasi, fibrotik, kavitas.
- "ada" jika disebut ada; "tidak" jika tidak disebut ATAU dinegasi ("tidak tampak nodul" -> nodul="tidak").
- Temuan dinilai apa adanya; adanya temuan TIDAK otomatis berarti TB.

kardiomegali ("ada"/"tidak"): "ada" bila jantung membesar / kardiomegali / CTR meningkat disebut.

Field "alasan": tulis SINGKAT kutipan/frasa dasar penilaian, SEBELUM label. Jawab HANYA JSON sesuai skema."""

USER_TEMPLATE = "Catatan:\n\"\"\"\n{teks}\n\"\"\""
PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + USER_TEMPLATE + json.dumps(SCHEMA, sort_keys=True)).encode()
).hexdigest()[:16]


def ollama_version():
    try: return subprocess.check_output(['ollama','--version'],text=True).strip()
    except Exception as e: return f'unknown ({e})'

def model_digest(name):
    try:
        info = ollama.show(name); d = info.get('details', {})
        return {'family': d.get('family',''), 'param_size': d.get('parameter_size',''),
                'quant': d.get('quantization_level','')}
    except Exception as e:
        return {'family': f'unknown ({e})', 'param_size': '', 'quant': ''}

def infer(name, teks):
    t0 = time.perf_counter()
    resp = ollama.chat(model=name,
        messages=[{'role':'system','content':SYSTEM_PROMPT},
                  {'role':'user','content':USER_TEMPLATE.format(teks=teks)}],
        format=SCHEMA, options=OPTIONS)
    latency = time.perf_counter() - t0
    raw = resp['message']['content']
    try: parsed, ok = json.loads(raw), True
    except Exception: parsed, ok = {}, False
    return parsed, raw, ok, latency

def derive_dx(row):
    if row.get('is_radiologi') == 'artefak': return 'artefak'
    imp = row.get('imp_tb', 'tidak')
    if imp == 'aktif':   return 'tb_aktif'
    if imp == 'suspek':  return 'tb_suspek'
    if imp == 'inaktif': return 'tb_inaktif'
    any_abn = (any(row.get(c)=='ada' for c in FINDING_COLS) or row.get('kardiomegali')=='ada')
    return 'non_tb_abnormal' if any_abn else 'normal'


def main():
    src = pd.read_excel(INPUT_XLSX)              # ID_unik, teks, n_baris
    src['teks'] = src['teks'].astype(str).str.strip()
    src = src[src['teks'] != ''].reset_index(drop=True)
    print(f'Teks unik dimuat: {len(src)} (kunci ID_unik) | prompt hash: {PROMPT_HASH}')

    meta = {'timestamp': datetime.now().isoformat(timespec='seconds'),
            'ollama_version': ollama_version(), 'host': platform.node(),
            'options': json.dumps(OPTIONS), 'prompt_hash': PROMPT_HASH,
            'repeat': REPEAT, 'n_texts': len(src)}
    for tag, name in MODELS:
        meta[f'model_{tag}'] = name
        meta[f'digest_{tag}'] = json.dumps(model_digest(name))

    records = []
    for tag, name in MODELS:
        print(f'\n=== {tag} ({name}) ===')
        for rep in range(1, REPEAT + 1):
            for pos, r in src.iterrows():
                parsed, raw, ok, latency = infer(name, r['teks'])
                rec = {'ID_unik': r['ID_unik'], 'model': tag, 'condition': 'zeroshot',
                       'repeat': rep,
                       'teks': r['teks'], 'json_valid': ok, 'latency_s': round(latency,3),
                       'alasan': parsed.get('alasan',''), 'raw': raw}
                for c in LABEL_COLS: rec[c] = parsed.get(c, None)
                rec['dx_category'] = derive_dx(rec) if ok else None
                records.append(rec)
                if (pos + 1) % 20 == 0: print(f'  rep{rep}: {pos+1}/{len(src)}')

    df = pd.DataFrame(records)
    df_meta = pd.DataFrame(list(meta.items()), columns=['key','value'])
    print(f'\nSelesai. Baris: {len(df)} | JSON gagal: {int((~df["json_valid"]).sum())}')
    print('Latensi rata2/model (dtk):'); print(df.groupby('model')['latency_s'].mean().round(2).to_string())
    with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='Predictions', index=False)
        df_meta.to_excel(w, sheet_name='RunMetadata', index=False)
    print(f'\nTersimpan: {OUTPUT_XLSX}')

if __name__ == '__main__':
    main()
