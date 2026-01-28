# 技術メモ：CRE導入による station elevation（負標高＝深度）の走時計算組込み（synth_hypoinverse_eval）

## 1. 目的（このセッションで達成したこと）
- HypoInverse の **hypoellipse 系クラストモデル（CRE）**を利用し、**station elevation（負の標高＝深度）**を走時計算へ自然に組み込む。
- 従来の **CRH + station delay（標高由来補正）近似**を回避し、幾何学的に一貫した走時計算を実現する。
- **震源が受振器より浅い**場合を含む幾何学条件でも安定動作する構成へ改善する。
- `proc` を薄く保ち、変換・生成・パッチ処理は `src` に集約する（責務分離・再利用・重複最小化）。

---

## 2. 全体方針（設計の要点）
- station elevation の反映は **CREの travel-time 計算に委譲**する。
- `.sta` に `Elevation_m` を正しく書き、cmd に **`CRE ... ref_elev_km T`（use_station_elev=T）**を注入して確実に有効化する。
- `apply_station_elevation_delay`（標高由来のdelay補正）は **原則無効**にし、二重補正を避ける。
- CRH資産（速度モデル）の再利用を前提にしつつ、**reference / typical / shift** を明示管理して深さ基準ズレを制御する。
- 速度モデルのシフトは **層境界のみ**（`i>=1`）に適用し、`top[0]=0` を固定する。

---

## 3. 実装ステップ別まとめ

### Step A：受振器ジオメトリから station CSV 生成
**目的**
- 受振器深度（z）を `Elevation_m` へ写像し、`.sta` 生成入力となる `station_synth.csv` を整備する。

**主な変更**
- `src/hypo/synth_eval/builders.py::build_station_df(..., *, z_is_depth_positive: bool)`
  - `z_is_depth_positive=True` のとき `Elevation_m = round(-z_m)`（zがdepth +down想定）
  - `z_is_depth_positive=False` のとき `Elevation_m = round(z_m)`
- `src/hypo/synth_eval/io.py::write_station_csv(df_station, out_csv)`
  - 必須列（`station_code`, `Latitude_deg`, `Longitude_deg`, `Elevation_m`）を検証し、CSVを書き出す。

---

### Step B：`.sta` 生成（深井戸対応の固定幅整形）
**目的**
- station format #2 に準拠した `.sta` を生成し、深井戸（-1000m級）の負標高でも列崩れを回避する。

**要点**
- `src/hypo/sta.py::_format_elevation_i4_and_neg_flag(elevation_m)`
  - `elevation_m < -999` のとき `(abs(elevation_m), '-')`（col86負フラグ）
  - `abs(elevation_m) > 9999` は例外（I4制約）
- `src/hypo/sta.py::format_station_line(...)`
  - col39-42 に I4 elevation、col86 に負フラグ、行長 86 を検証。
- `src/hypo/sta.py::write_hypoinverse_sta(..., *, force_zero_pdelays: bool=False)`
  - CRE運用で elevation を幾何に入れる場合に、`force_zero_pdelays=True` を指定して `pdelay1/pdelay2` を 0.0 に固定可能にした（delay二重計上の事故防止）。
  - **デフォルトは False**（汎用性保持のため）。

---

### Step C：CREパラメータ決定（reference / typical / shift）
**目的**
- CREの参照標高 `ref_elev_km` を決定し、CRH資産の深さ基準ズレを補正するための `shift_km` を算出する。

**新規ファイル**
- `src/hypo/cre.py`

**関数**
- `compute_reference_elevation_km(station_df, elevation_col='Elevation_m', margin_m=0.0)`
  - `ref_elev_km = (max(Elevation_m) + margin_m)/1000`
  - `ref_m < 0` の場合は `0` にクランプ（borehole-onlyで負refになりにくい設計）
- `compute_typical_station_elevation_km(*, explicit_m: float|None)`
  - 指定あり：`explicit_m/1000`
  - 未指定：`0.0`（固定）
- `compute_cre_layer_top_shift_km(ref_elev_km, typical_elev_km)`
  - `shift_km = ref - typical`
- `write_cre_meta(run_dir, ref_elev_km, typical_elev_km, shift_km)`
  - run_dir に `cre_ref_elev_km.txt` 等を保存し、再現性を担保。

---

### Step D：速度モデル出力（CRE用、層境界のみシフト）
**目的**
- `run_dir` 内に `P.cre` / `S.cre` を生成し、CRH→CREの深さ基準ズレを最小限補正する。

**設計原則**
- `top[0] = 0.0` 固定
- シフト適用は `i>=1` のみ
- 単層モデルは適用箇所なし（影響ゼロ）

**実装（API分離：汎用＋合成用）**
- `src/hypo/cre.py::apply_layer_top_shift_km(layer_tops_km, shift_km)`
  - `tops[0]=0`、`i>=1` のみ `+shift`、単調増加を fail-fast 検証。
- 汎用：`src/hypo/cre.py::write_cre_from_layer_tops(run_dir, vp_kms, vs_kms, layer_tops_km, shift_km)`
  - 任意topsを受け取り、shift適用後に `write_crh` を流用して `P.cre/S.cre` を書く。
- 合成用：`src/hypo/synth_eval/pipeline.py::build_synth_layer_tops_km(n_layers)`
  - `[0.0, 1.0, ..., n_layers-1]`
- 合成用：`src/hypo/synth_eval/pipeline.py::write_synth_cre_models(...)`
  - `build_synth_layer_tops_km` → `write_cre_from_layer_tops` へ委譲。

---

### Step E：cmd パッチ（CRH→CRE、elevation使用強制、SAL整備）
**目的**
- `.sta` と `.cre` を参照するcmdを生成し、station elevation 使用を確実化する。

**実装**
- `src/hypo/cre.py::format_cre_cmd_line(model_id, model_file, ref_elev_km, use_station_elev=True)`
  - `CRE 1 'P.cre' <ref> T` の形式で生成（T/F切替）。
- `src/hypo/synth_eval/hypoinverse_runner.py::patch_cmd_template_for_cre(...)`
  - `STA` 行を `STA 'stations_synth.sta'` に固定
  - `CRH/CRT` を除去/上書きして `CRE 1/2` 行注入
  - `SAL 1 2` を維持、欠落時は挿入
  - 必須行欠落（STA, crust-model 1/2）時は例外（fail-fast）

---

### Step F：二重補正防止の整合性検証（validation集約）
**目的**
- 幾何補正（CRE+use_station_elev）と標高由来delay補正（apply_station_elevation_delay）の同時有効化を禁止し、誤設定を即失敗させる。

**実装**
- `src/hypo/synth_eval/validation.py::validate_elevation_correction_config(model_type, use_station_elev, apply_station_elevation_delay)`
  - `model_type=='CRE' and use_station_elev and apply_station_elevation_delay` → 例外
  - `model_type!='CRE' and use_station_elev` → 例外（本pipelineではCRE専用とする方針）

---

### Step G：pipeline統合（成果物生成→HypoInverse実行）
**目的**
- run_dir内で成果物を順に生成し、CRE導入後の評価が成立する実行配線を行う。

**実装（主な流れ）**
- `src/hypo/synth_eval/pipeline.py::run_synth_eval(...)`
  1. validation 実行（StepF）
  2. station_df生成 → station_synth.csv 書き出し
  3. `.sta` 生成（CRE+use_station_elevなら `force_zero_pdelays=True`）
  4. ref/typical/shift 算出 → run_dir に meta 保存（StepC）
  5. `P.cre/S.cre` 生成（StepD）
  6. cmd パッチ生成（StepE、CRE行注入）
  7. HypoInverse 実行
- `src/hypo/synth_eval/hypoinverse_runner.py::run_hypoinverse(exe_path, cmd_path, run_dir) -> subprocess.CompletedProcess`
  - `check=True` のまま戻り値を返すように変更。

---

## 4. 動作確認（最小 config 差分）
- proc の synth eval の例configを CRE 用に追加し、`model_type: "CRE"` を設定。
- 主な追加/利用キー（例）：
  - `model_type: "CRE"`
  - `use_station_elev: true`
  - `z_is_depth_positive: true`
  - `cre_reference_margin_m: 0.0`
  - `cre_typical_station_elevation_m: null`（→ typical=0.0）
  - `cre_n_layers: 1`
  - `apply_station_elevation_delay: false`

---

## 5. 結果
- CRE導入により、station elevation（負標高＝深度）が走時計算に幾何学的に反映され、震源決定結果が改善した。

---

## 6. 今後の拡張余地（メモ）
- **既存CRHモデル（任意tops）→CRE化**：`write_cre_from_layer_tops()` を中核に自然拡張可能。
- site補正（表層補正）を入れる場合：
  - 標高由来補正（apply_station_elevation_delay）はOFF維持
  - 必要なら **観測点固有 delay** を別管理（station delay か pick補正）で導入する。
- `ref_elev_km` が 0 以外になるケースでは、評価側（truth比較）の座標系整合の確認が必要になりうる。
