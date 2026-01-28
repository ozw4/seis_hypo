# 追加テスト 技術メモ（保証事項）

## `tests/test_hypo_synth_eval_builders.py`

### `test_build_station_df_elevation_from_depth_positive`
- `build_station_df(..., z_is_depth_positive=True)` における `Elevation_m` 算出規約の保証
- 入力 `z_m` を「深さ（下向き正）」として解釈した場合の変換 `Elevation_m = round(-z_m)` の保証
- `Elevation_m` 列の `int` 変換結果一致の保証（`astype(int)` 相当の成立）
- 符号反転の欠落・誤反転・丸め順序誤り（round 前後の処理順）の検知

### `test_build_station_df_elevation_from_up_positive`
- `build_station_df(..., z_is_depth_positive=False)` における `Elevation_m` 算出規約の保証
- 入力 `z_m` を「上向き（上向き正）」として解釈した場合の変換 `Elevation_m = round(z_m)` の保証
- `Elevation_m` 列の `int` 変換結果一致の保証
- `z_is_depth_positive` 分岐の逆転・常時反転・条件分岐無効化の検知

---

## `tests/test_hypo_synth_eval_io.py`

### `test_write_station_csv_contract_and_casts`
- `write_station_csv` の必須列契約の保証（`station_code`, `Latitude_deg`, `Longitude_deg`, `Elevation_m` の存在前提）
- 出力 CSV ヘッダの列順保証（必須4列を先頭に固定し、残余列を後続に配置）
- 型変換の保証
  - `station_code`：文字列化の保証（数値入力でも CSV 出力は文字列）
  - `Latitude_deg` / `Longitude_deg`：浮動小数化の保証（整数入力でも実数として読み取れる形）
  - `Elevation_m`：整数化の保証（float 入力でも整数として書き出し）
- 出力先親ディレクトリの自動生成保証（`out.parent.mkdir(parents=True, exist_ok=True)` 成立）
- CSV 1 行目データの具体値一致による、列対応関係（列の入れ替わり・欠落）の検知

### `test_write_station_csv_missing_required_columns`
- 必須列欠損時の fail-fast 保証（`validate_columns` 経由の `ValueError` 送出）
- 欠損列がある状態での部分的書き出し・暗黙補完・サイレント成功の禁止保証
