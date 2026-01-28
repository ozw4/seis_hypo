# `hypo.cre` 追加テスト 技術メモ（保証事項）

## 対象
- モジュール: `src/hypo/cre.py`
- 機能: CRE 用パラメータ計算（基準標高・典型標高・層境界シフト）およびメタ情報ファイル出力
- 目的: 計算規約の固定、入力異常の fail-fast、出力ファイルの生成契約の固定

---

## テストファイル
- `tests/test_hypo_cre.py`

---

## テスト一覧と保証内容

### `test_compute_reference_elevation_km_max_plus_margin`
- `compute_reference_elevation_km` の正常系計算規約の保証
- 定義 `ref_elev_km = (max(Elevation_m) + margin_m) / 1000` の成立保証
- `Elevation_m` が float でも正しく max を取ることの保証
- `margin_m` が加算されることの保証
- 計算結果の km 単位化の保証

---

### `test_compute_reference_elevation_km_clamps_negative_to_zero`
- 基準標高の負値クランプ規約の保証
- `max(Elevation_m) + margin_m < 0` の場合に `ref_elev_km == 0.0` となることの保証
- borehole-only 等で基準標高が負に落ちるケースの抑止保証（深さ基準の不合理化回避）

---

### `test_compute_reference_elevation_km_input_validation`（parametrize）
- 入力異常時の fail-fast 動作保証（`ValueError` 送出）
- ケース別保証
  - `station_df is None`：`ValueError` 送出保証
  - 必須列欠損（`elevation_col` 不存在）：`validate_columns` 経由の `ValueError` 送出保証
  - `station_df.empty`：`ValueError` 送出保証
  - 対象列が全て `NaN`：有効値無しとして `ValueError` 送出保証
- 目的: サイレントに不正値で計算が進行する事象の排除

---

### `test_compute_typical_station_elevation_km_rules`
- `compute_typical_station_elevation_km` の規約保証
- `explicit_m is None` → `0.0 km` の保証
- `explicit_m` 指定時 → `explicit_m / 1000`（km 化）の保証

---

### `test_compute_cre_layer_top_shift_km_is_difference`
- `compute_cre_layer_top_shift_km` の規約保証
- 定義 `shift_km = ref_elev_km - typical_elev_km` の保証
- CRE datum 変換における層境界シフト量算出の不変性保証

---

### `test_write_cre_meta_writes_three_scalars`
- `write_cre_meta` の出力契約保証
- `run_dir` 未作成でも親ディレクトリ含め生成されることの保証（`mkdir（parents=True）`）
- 生成ファイル 3 点の存在保証
  - `cre_ref_elev_km.txt`
  - `cre_typical_station_elev_km.txt`
  - `cre_layer_top_shift_km.txt`
- 各ファイル末尾改行の保証（POSIX 互換テキスト終端）
- 数値の round-trip 可能性保証（`float(text.strip())` が元の値に一致）
  - 目的: 表現形式（`.15g`）に過度に依存せず、読み戻し可能性を担保

---

## 本テストにより固定される仕様
- 基準標高の定義と単位（m → km）および margin の扱い
- 基準標高の非負クランプ規約
- 入力異常の明示的検知（None / 列欠損 / 空 / 全 NaN）
- 典型標高の既定値（None → 0 km）と km 化
- 層境界シフト量の差分定義
- CRE メタ情報ファイル生成・改行・読み戻し可能性
