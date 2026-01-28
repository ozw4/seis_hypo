# CRE レイヤーシフト／CREモデル出力 追加テスト 技術メモ（保証事項）

## 対象パッチ
- `src/hypo/cre.py`
  - `apply_layer_top_shift_km`：入力 `layer_tops_km` の「厳密単調増加」検証追加
  - `write_cre_from_layer_tops`：P/S の CRE モデル（`P.cre` / `S.cre`）出力機能追加
- `src/hypo/synth_eval/pipeline.py`
  - `build_synth_layer_tops_km`：合成レイヤー境界生成の切り出し
  - `write_synth_cre_models`：`write_cre_from_layer_tops` への委譲構造へ変更

---

## テストファイル
- `tests/test_hypo_cre_layers.py`
- `tests/test_hypo_synth_eval_pipeline_cre_models.py`

---

## 共通前提（CRH/CRE ファイルの読み取り）
- 補助関数 `_read_crh_layers`
  - 先頭行：モデル名（例 `CRE_P`, `CRE_S`）の取得前提
  - 2行目以降：固定幅 `v`（先頭5文字）+ `top`（次の5文字）の数値表現前提
  - 目的：`write_crh` 出力フォーマットの最小限パースによる契約検証
  - 留意：`write_crh` の固定幅仕様が変更された場合のテスト破断可能性

---

## `tests/test_hypo_cre_layers.py`

### `test_apply_layer_top_shift_km_rejects_non_increasing`
- 追加仕様「`layer_tops_km` はシフト前から厳密単調増加」の fail-fast 保証
- 等値（例 `[0.0, 1.0, 1.0]`）入力時の `ValueError` 送出保証
- 減少（例 `[0.0, 2.0, 1.0]`）入力時の `ValueError` 送出保証
- 目的：不正な層境界がシフト処理に流入する事象の遮断

### `test_apply_layer_top_shift_km_applies_shift_only_to_i_ge_1`
- `apply_layer_top_shift_km` の変換規約保証
- `tops[0] = 0.0` の強制維持保証
- `i>=1` にのみ `shift_km` 加算されることの保証
- 目的：深さ基準（層境界先頭0固定）と CRE シフトの意図どおりの適用保証

### `test_write_cre_from_layer_tops_writes_p_and_s_cre`
- `write_cre_from_layer_tops` の出力契約保証
- `run_dir` 自動生成の成立保証（親ディレクトリ含む）
- 出力パス固定保証（`P.cre`, `S.cre`）と戻り値一致保証
- モデル名の正当性保証（`P.cre` 先頭 `CRE_P`, `S.cre` 先頭 `CRE_S`）
- 層境界シフト結果の反映保証（例：`[0.0,1.0,2.0] + shift 0.3 → [0.0,1.3,2.3]`）
- 速度一定の反映保証（P 側 `vp_kms`、S 側 `vs_kms` が全行で一定）
- 目的：新設関数のI/O契約（作る・名付ける・中身が合う）の固定

---

## `tests/test_hypo_synth_eval_pipeline_cre_models.py`

### `test_build_synth_layer_tops_km_rules`
- `build_synth_layer_tops_km` の生成規約保証
- `n_layers=1` → `[0.0]` の保証
- `n_layers=3` → `[0.0,1.0,2.0]` の保証
- 目的：合成レイヤー境界生成の仕様固定（`0.0` 起点・1 km 刻み）

### `test_build_synth_layer_tops_km_rejects_lt_1`
- `n_layers < 1` 入力時の fail-fast 保証（`ValueError` 送出）
- 目的：無効な層数による downstream の不定挙動排除

### `test_write_synth_cre_models_smoke`
- `write_synth_cre_models` の委譲構造の成立保証（スモーク）
- 呼び出しにより `P.cre` / `S.cre` が生成されることの保証
- モデル名が `CRE_P` / `CRE_S` となることの保証（委譲先の整合性）
- 合成レイヤー境界 + shift の反映保証（例：`n_layers=3`, `shift=0.25` → `[0.0,1.25,2.25]`）
- P/S 速度一定の反映保証（P=vp_kms, S=vs_kms）
- 目的：パイプライン側の関数分割・委譲後も期待動作が保たれることの確認

---

## 本テストにより固定される仕様
- シフト前の層境界配列が厳密単調増加であることの必須条件
- 先頭層境界 `0.0` 固定と、`i>=1` のみのシフト適用規約
- CREモデル出力ファイル名、モデル名、層境界値、速度値の最小契約
- 合成レイヤー境界生成規約と、委譲後の生成物整合性
