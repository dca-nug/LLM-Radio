# Offline NLP for Indonesian Chest X-ray TB Screening Reports

Code accompanying the study *[ISI: judul manuskrip]*. The pipeline extracts
structured labels from Indonesian-language chest radiograph reports and compares
three extraction systems against radiologist-annotated ground truth. All
computation runs offline on a single workstation, so no report text leaves the
machine.

This is a characterization/evaluation study for a first-in-language setting. It
is not a claim that any system is superior, and deployment is out of scope.

## Systems compared

1. **Rule-based** — medSpaCy/ConText pipeline (`tb_rontgen_medspacy.py`).
2. **Generic LLM** — Gemma2-9B-instruct, zero-shot (`llm_harness.py`).
3. **Indonesian-adapted LLM** — Sahabat-AI (Gemma2-9B CPT), zero-shot
   (same harness, model `sahabatgemma-fix`).

A one-shot condition (`llm_harness_oneshot.py`) is reported as supplementary.

## Labels (five constructs)

- `is_radiologi` ∈ {`radiologi`, `artefak`}
- `imp_tb` ∈ {`aktif`, `suspek`, `inaktif`, `tidak`}
- five radiographic findings ∈ {`ada`, `tidak`}: `infiltrat`, `nodul`,
  `kalsifikasi`, `fibrotik`, `kavitas`
- `kardiomegali` ∈ {`ada`, `tidak`}

Annotation rule: findings may be marked present when implied by the impression;
`imp_tb` is taken only from explicit impression wording (no activity inferred
from finding pattern or location). See the manuscript Methods for the full
protocol.

## Repository layout

```
config.py                 # single source of the data directory (env: TB_DATA_DIR)
ambil_unik.py             # 1. deduplicate reports -> unique texts
build_form_from_unik.py   # 2. build the radiologist annotation form
tb_rontgen_medspacy.py    # 3a. rule-based system
llm_harness.py            # 3b. LLM system, zero-shot
llm_harness_oneshot.py    # 3c. LLM system, one-shot (supplementary)
llm_harness_repeat.py     # 4a. rerun (REPEAT=2) for determinism check
cek_determinisme.py       # 4b. compare repeat 1 vs 2
evaluasi.py               # 5. compare all systems vs ground truth
modelfiles/               # Ollama Modelfile for sahabatgemma-fix
data/                     # inputs/outputs (not committed; see data/README.md)
requirements.txt
```

## Requirements

- Python 3.11.4
- Packages in `requirements.txt`
- [Ollama](https://ollama.com) server v0.30.11 for the LLM arms
- Hardware used in the study: Intel i5-12400F, 16 GB DDR5, NVIDIA RTX 3060 12 GB

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The rule-based pipeline uses only the medSpaCy tokenizer, target matcher, and
ConText. No pretrained spaCy model is required.

## Data

Raw data are not included (health screening data under ethics approval). Place
inputs in `data/` or point `TB_DATA_DIR` at the folder that holds them:

```bash
export TB_DATA_DIR=/path/to/data      # Windows PS: $env:TB_DATA_DIR="F:\data\tb-cxr"
```

Input files and column schema are documented in [`data/README.md`](data/README.md).

## LLM models (Ollama)

```bash
ollama pull gemma2:9b-instruct-q4_K_M
ollama create sahabatgemma-fix -f modelfiles/sahabatgemma-fix.Modelfile
```

The only asymmetry between the two LLM arms is Sahabat-AI's custom Modelfile
(a corrected Gemma2 chat template). Decoding parameters are identical:
`temperature=0`, `seed=42`, `num_ctx=4096`, Q4_K_M quantization.
Both LLM arms use schema-constrained JSON output. The harness reads only the
report text and never touches the ground truth.

## Run order

```bash
python ambil_unik.py              # -> TB_teks_unik.xlsx, TB_peta_baris.xlsx
python build_form_from_unik.py    # -> Form_Anotasi_TB_CXR.xlsx  (for annotation)
# --- two radiologists annotate independently; consensus resolves disagreements ---
python tb_rontgen_medspacy.py     # -> RuleBased_pred.xlsx
python llm_harness.py             # -> LLM_predictions.xlsx
python llm_harness_oneshot.py     # -> LLM_predictions_oneshot.xlsx  (optional)
python llm_harness_repeat.py      # -> LLM_predictions_repeat.xlsx
python cek_determinisme.py        # -> Cek_determinisme.xlsx
python evaluasi.py                # -> Evaluasi_hasil.xlsx
```

Prompts and rules are frozen before any system sees the test set. Annotators
stay blinded to system outputs throughout, including during consensus.

## Reproducibility notes

- Unit of analysis is the unique report text (N = 107), deduplicated from 1,347
  chest radiograph reports. Bootstrap resampling in `evaluasi.py` is clustered
  at the unique-text level.
- Frozen prompt hashes: zero-shot `1b432dfeb369b503`, one-shot `195828bed5eab4c3`
  (printed at runtime by each harness).
- Determinism was checked across two full reruns and was 100% identical over
  1,712 label decisions.

## Code and data availability (draft for the manuscript)

> The analysis code is available at [ISI: repo URL] under [ISI: license]
> ([ISI: DOI Zenodo]). The report texts contain health screening data and cannot
> be shared publicly; they may be available from the corresponding author on
> reasonable request, subject to ethics approval [ISI: nomor EC] and a data use
> agreement.

## Citation

See `CITATION.cff`. Fill the placeholders before publishing the repo.

## License

[ISI: pilih lisensi] — see `LICENSE`.
