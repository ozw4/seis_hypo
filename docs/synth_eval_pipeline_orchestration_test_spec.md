# synth_eval pipeline 分岐／設定／HypoInverse 実行 追加テスト 技術メモ（保証事項）

## 対象修正
- `src/hypo/synth_eval/pipeline.py`
  - `load_config`：モデル種別・CRE関連設定・デフォルト値の追加
  - `run_synth_eval`：`validate_elevation_correction_config` 呼び出し追加
  - `build_station_df`：`z_is_depth_positive` の設定伝播
  - `write_hypoinverse_sta`：`force_zero_pdelays` 条件付与（CRE + use_station_elev）
  - CRE 分岐：`compute_reference_elevation_km` / `compute_typical_station_elevation_km` / `compute_cre_layer_top_shift_km` / `write_cre_meta` / `write_synth_cre_models` / `patch_cmd_template_for_cre` の導入
  - CRH 分岐：既存 `write_crh` + `write_cmd_from_template` の維持
- `src/hypo/synth_eval/hypoinverse_runner.py`
  - `run_hypoinverse`：戻り値 `subprocess.CompletedProcess` 返却へ変更

---

## テストファイル
- `tests/test_hypo_synth_eval_pipeline_load_config_cre.py`
- `tests/test_hypo_synth_eval_pipeline_orchestration_cre_crh.py`
- `tests/test_hypo_synth_eval_hypoinverse_runner_run.py`

---

## `tests/test_hypo_synth_eval_pipeline_load_config_cre.py`（1. load_config 規約）

### `test_load_config_defaults_model_type_and_related_fields`
- `model_type` 欠損時の既定値 `CRH` 適用保証
- CRE関連パラメータの既定値適用保証
  - `use_station_elev=False`（CRH時）
  - `cre_reference_margin_m=0.0`
  - `cre_typical_station_elevation_m=None`
  - `cre_n_layers=1`
  - `z_is_depth_positive=True`
- 設定未記載時の安定な初期条件の固定

### `test_load_config_normalizes_model_type_and_parses_typical_elevation`
- `model_type` の正規化（`strip()` + `upper()`）保証（例：`" cre "` → `"CRE"`）
- `cre_typical_station_elevation_m` の数値入力を `float` として保持することの保証
- 型ぶれ（strのまま保持等）の排除

### `test_load_config_rejects_invalid_model_type`
- `model_type` が `CRE/CRH` 以外の場合の fail-fast 保証（`ValueError` 送出）
- パイプライン分岐条件の前提破壊の早期検知

### `test_load_config_default_use_station_elev_depends_on_model_type`
- `use_station_elev` 未指定時の既定値が `model_type` 依存であることの保証
  - `model_type=CRE` → `use_station_elev=True`
  - `model_type=CRH` → `use_station_elev=False`
- CRE/CRH での期待挙動差の固定

---

## `tests/test_hypo_synth_eval_pipeline_orchestration_cre_crh.py`（2〜6. run_synth_eval 呼び分け）

### `test_run_synth_eval_calls_validate_elevation_correction_config`
- `run_synth_eval` 冒頭で `validate_elevation_correction_config` が必ず呼ばれることの保証
- バリデーション関数が例外を送出した場合の早期停止保証（副作用発生前の停止）
- 二重補正回避ロジックの適用漏れ防止

### `test_run_synth_eval_passes_z_is_depth_positive_to_build_station_df`
- `cfg.z_is_depth_positive` が `build_station_df(..., z_is_depth_positive=...)` に伝播することの保証
- `z_is_depth_positive=False` 指定時に False が渡ることの保証（符号系設定の取り違え検知）
- CRH 分岐における `force_zero_pdelays=False` 保証
- CRH 分岐における `write_crh('SYNTH_P')` / `write_crh('SYNTH_S')` 呼び出し保証
- CRE系関数の非呼び出しを間接的に担保（CRH動作の後方互換）

### `test_run_synth_eval_cre_branch_calls_expected_functions_and_force_zero_pdelays`
- CRE 分岐時の呼び出し系列の成立保証（monkeypatch による関数呼び出し監視）
  - `compute_reference_elevation_km`：`margin_m=cfg.cre_reference_margin_m` の引き渡し保証
  - `compute_typical_station_elevation_km`：`explicit_m=cfg.cre_typical_station_elevation_m` の引き渡し保証
  - `compute_cre_layer_top_shift_km`：ref/typical の差分計算引き渡し保証
  - `write_cre_meta`：`run_dir` と3 scalar（ref/typical/shift）引き渡し保証
  - `write_synth_cre_models`：`shift_km` と `n_layers=cfg.cre_n_layers` 引き渡し保証
  - `patch_cmd_template_for_cre`：ファイル名のみ渡す契約保証
    - `sta_file == 'stations_synth.sta'`
    - `p_model == 'P.cre'`
    - `s_model == 'S.cre'`
    - `ref_elev_km` / `use_station_elev` の一致保証
- CRE 分岐時の非呼び出し保証
  - `write_cmd_from_template` 非呼び出し保証
  - `write_crh` 非呼び出し保証
- `write_hypoinverse_sta(force_zero_pdelays=...)` 条件の保証
  - `model_type=CRE` かつ `use_station_elev=True` のときのみ True
  - 上記以外では False
- CRE 分岐のオーケストレーション破壊（呼び順／引数／分岐漏れ）の検知

### `test_run_synth_eval_crh_branch_calls_expected_functions`
- CRH 分岐時の後方互換動作保証
  - `write_crh('SYNTH_P')` / `write_crh('SYNTH_S')` 呼び出し保証
  - `write_cmd_from_template` 呼び出し保証
  - `force_zero_pdelays=False` 保証
- CRH 分岐時の CRE 系関数非呼び出し保証
  - `compute_reference_elevation_km` 等が呼ばれた場合に即失敗（unexpected call）
- 分岐条件の誤判定・呼び分けミスの検知

---

## `tests/test_hypo_synth_eval_hypoinverse_runner_run.py`（7. run_hypoinverse 戻り値・subprocess契約）

### `test_run_hypoinverse_returns_completedprocess_and_calls_subprocess_run`
- `run_hypoinverse` が `subprocess.run` の戻り値をそのまま返すことの保証（`CompletedProcess` 返却）
- `subprocess.run` 呼び出し契約の保証
  - `args == [str(exe_path)]`
  - `cwd == run_dir`
  - `check == True`
  - `stdin` に cmd ファイルハンドルが渡されることの保証（read可能オブジェクト）
- 外部実行ラッパのI/O契約破壊（引数欠落、cwd不一致、check無効化、戻り値無視）の検知

---

## 本テストにより固定される仕様
- `load_config` のモデル種別・CRE関連キー・デフォルト値・正規化規約
- `run_synth_eval` の fail-fast バリデーション適用、符号設定伝播、CRE/CRH 分岐の呼び分け契約
- CRE 分岐におけるメタ生成・モデル生成・cmdパッチ・pdelayゼロ化条件の整合
- CRH 分岐における後方互換（CRH出力 + テンプレcmd適用）の維持
- HypoInverse 実行ラッパの `subprocess.run` 呼び出しと戻り値契約
