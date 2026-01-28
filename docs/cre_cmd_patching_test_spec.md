# CRE CMD パッチング系 追加テスト 技術メモ（保証事項）

## 対象パッチ
- `src/hypo/cre.py`
  - `format_cre_cmd_line` 追加（CRE コマンド行の整形）
- `src/hypo/synth_eval/hypoinverse_runner.py`
  - `_parse_cmd_model_number` 追加（モデル番号トークンの解釈）
  - `patch_cmd_template_for_cre` 追加（テンプレ cmd の CRE 向けパッチ）

---

## テストファイル
- `tests/test_hypo_cre_cmd_line.py`
- `tests/test_hypo_synth_eval_patch_cmd_template_for_cre.py`

---

## `tests/test_hypo_cre_cmd_line.py`

### `test_format_cre_cmd_line_true_flag_and_rounding`
- `format_cre_cmd_line` の出力フォーマット契約保証
- `CRE {model_id} '{model_file}' {ref_elev_km:.6f} {T/F}` 形式の保証
- `ref_elev_km` の小数6桁丸め（`.6f`）規約の保証
- `use_station_elev=True` → 末尾フラグ `T` の保証

### `test_format_cre_cmd_line_false_flag`
- `use_station_elev=False` → 末尾フラグ `F` の保証
- モデルID=2でも同一フォーマット規約が成立することの保証

### `test_format_cre_cmd_line_rejects_invalid_model_id`
- `model_id` が {1,2} 以外の場合の fail-fast 保証（`ValueError` 送出）
- CRE 1/2 固定の前提逸脱検知

### `test_parse_cmd_model_number_accepts_signed_int`
- `_parse_cmd_model_number` の許容入力仕様保証
- 前後空白の許容保証
- `+` / `-` 符号付き整数の許容保証
- 返却値が `int` として解釈されることの保証

### `test_parse_cmd_model_number_rejects_invalid`
- 空文字・空白のみ入力の reject 保証（`ValueError`）
- 非数字混在・符号のみ・多重符号・小数表現などの reject 保証
- テンプレ cmd の破損行に対する早期検知保証

---

## `tests/test_hypo_synth_eval_patch_cmd_template_for_cre.py`

### `test_patch_cmd_template_for_cre_replaces_sta_and_models_and_inserts_sal`
- `patch_cmd_template_for_cre` の主要変換契約保証（正常系）
- コメント行（`*`）および空行の保持保証
- `STA` 行の置換保証（`STA '{sta_file}'`）
- `CRH/CRT/CRE 1` の置換保証（`CRE 1 'P.cre' ref_elev_km T/F`）
- `CRH/CRT/CRE 2` の置換保証（`CRE 2 'S.cre' ref_elev_km T/F`）
- 無関係モデル定義（例：`CRH 3 ...`）の不改変保証
- 出力/入力ファイル名の固定置換保証（既存パッチャ互換）
  - `PHS 'hypoinverse_input.arc'`
  - `PRT 'hypoinverse_run.prt'`
  - `SUM 'hypoinverse_run.sum'`
  - `ARC 'hypoinverse_run_out.arc'`
- `SAL 1 2` の自動挿入保証（テンプレに SAL 無しの場合）
- 挿入位置規約の保証（最後に置換された crust-model 行の直後、優先的に model2 行の直後）
- 出力末尾改行の付与保証（`\\n` 終端）

### `test_patch_cmd_template_for_cre_rewrites_existing_sal`
- 既存 `SAL` 行の上書き保証（任意値 `SAL 9 9` → `SAL 1 2`）
- `SAL 1 2` の重複生成防止保証（出現回数=1）
- `use_station_elev=False` を反映した CRE 行生成保証（末尾 `F`）

### `test_patch_cmd_template_for_cre_missing_sta_raises`
- テンプレ cmd に `STA` 行が存在しない場合の fail-fast 保証（`ValueError`）
- station file 未指定状態でのサイレント進行禁止保証

### `test_patch_cmd_template_for_cre_missing_model2_raises`
- テンプレ cmd に `CRH/CRT/CRE 2` が存在しない場合の fail-fast 保証（`ValueError`）
- CRE S モデル未設定状態でのサイレント進行禁止保証

---

## 本テストにより固定される仕様
- CRE コマンド行整形の文法・丸め・フラグ規約
- crust-model 行のモデル番号パース仕様（許容入力と reject 条件）
- cmd テンプレパッチ処理の必須要素（STA/モデル1/モデル2/SAL）の存在要件
- SAL の挿入・上書き・重複防止および挿入位置規約
- 既存互換のファイル名固定置換規約（PRT/SUM/ARC/PHS）
- コメント/空行保持および出力末尾改行付与のテキスト整形規約
