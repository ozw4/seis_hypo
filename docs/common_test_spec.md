# common テスト仕様メモ

## 対象範囲
- `src/common/*` ユーティリティ群の基本動作確認
- 例外発生条件・境界条件の明文化
- 追加実装テスト作成時の雛形整備

---

## `tests/conftest.py`
- `./src` の import 可能性確保（レポ未インストール前提のテスト実行環境）
- `sample_event_dict` fixture：最小 `event.json` 相当ペイロード供給
- `write_text` fixture：一時ディレクトリ配下への UTF-8 テキスト生成ユーティリティ

---

## `tests/test_common_json_io.py`（`common.json_io`）

### `test_read_write_json_roundtrip`
- `write_json` → `read_json` のラウンドトリップ同値性
- UTF-8/非 ASCII（日本語）文字列の保持性
- list / dict / 数値を含む JSON オブジェクトの無損失復元性

---

## `tests/test_common_done_marker.py`（`common.done_marker`）

### `test_read_done_json_missing_empty`
- `done.json` 欠損時の空 dict 返却（`on_missing='empty'`）保証

### `test_read_done_json_missing_raise`
- `done.json` 欠損時の `FileNotFoundError` 送出（`on_missing='raise'`）保証

### `test_read_done_json_invalid_on_error_empty`
- JSON 破損時の空 dict 返却（`on_error='empty'`）保証

### `test_read_done_json_invalid_on_error_raise`
- JSON 破損時の `json.JSONDecodeError` 送出（`on_error='raise'`）保証

### `test_write_done_json_appends_newline`
- `write_done_json` による末尾改行付与（POSIX 互換テキスト終端）保証
- 親ディレクトリ自動生成の成立性

### `test_should_skip_done_run_tag_mismatch`
- `run_tag` 不一致時のスキップ無効化（常に `False`）保証

### `test_should_skip_done_ok_statuses_none`
- `ok_statuses=None` 時の無条件スキップ（`True`）保証

### `test_should_skip_done_ok_statuses_match`
- `status` が許容集合に含まれる場合のスキップ（`True`）保証

### `test_should_skip_done_ok_statuses_no_match`
- `status` が許容集合に含まれない場合のスキップ非成立（`False`）保証

---

## `tests/test_common_read_yaml.py`（`common.read_yaml`）

### `test_read_yaml_preset_file_missing`
- YAML ファイル欠損時の `FileNotFoundError` 送出保証

### `test_read_yaml_preset_root_not_mapping`
- YAML ルート非 mapping（list 等）時の `ValueError` 送出保証

### `test_read_yaml_preset_missing_preset`
- preset キー欠損時の `KeyError` 送出保証

### `test_read_yaml_preset_value_not_mapping`
- preset 値非 mapping（scalar 等）時の `ValueError` 送出保証

### `test_read_yaml_preset_ok`
- 正常系 preset 抽出結果の dict 一致保証

### `test_fieldnames_dataclass`
- dataclass フィールド名集合抽出の正確性保証

---

## `tests/test_common_load_config.py`（`common.load_config`）

### `test_load_config_requires_dataclass`
- dataclass 以外の型指定時の `TypeError` 送出保証

### `test_load_config_file_missing`
- YAML ファイル欠損時の `FileNotFoundError` 送出保証

### `test_load_config_ok_templates_and_path_cast`
- preset 読み込み成功の成立性
- `{base_dir}` `{name}` テンプレート展開の成立性
- `Path` 注釈フィールドの `Path` 変換保証
- Optional `Path | None` フィールドの `null` → `None` 保持保証

### `test_load_config_unknown_key`
- dataclass 非存在キー混入時の `ValueError` 送出保証（未知キー拒否）

### `test_load_config_missing_preset`
- preset 欠損時の `KeyError` 送出保証

### `test_load_config_template_unknown_key`
- 未定義テンプレ参照時の `KeyError` 送出保証（暗黙補完禁止）

---

## `tests/test_common_time_util.py`（`common.time_util`）

### `test_month_label`
- `YYYY-MM` 形式ラベル生成の正確性保証

### `test_iso_to_ns_matches_numpy`
- ISO8601 → ns 整数変換結果の NumPy/Pandas 整合性保証

### `test_floor_minute`
- 秒・マイクロ秒切り捨て（分頭丸め）の正確性保証

### `test_ceil_minutes`
- 秒数→分数切り上げの境界動作（60.0 秒=1 分、60.1 秒=2 分）保証
- 非正値入力時の `ValueError` 送出保証

### `test_minute_range_inclusive`
- start〜end を覆う分頭列挙の両端包含保証
- start/end の秒成分存在下での分頭整列保証

### `test_to_utc_naive_treated_as_jst_by_default`
- tz-naive `Timestamp` の JST 解釈（デフォルト）→ UTC 正規化保証

### `test_get_event_origin_utc_prefers_origin_time_jst`
- `origin_time_jst` 優先ロジック保証
- tz-naive `origin_time_jst` の JST 解釈→ UTC 変換保証

### `test_get_event_origin_utc_origin_time_naive_treated_as_utc`
- `origin_time` 使用時の tz-naive UTC 解釈保証

### `test_parse_cfg_time_utc`
- `None` 入力時の `None` 返却保証
- tz-naive 入力の JST 解釈→ UTC 変換保証
- tz-aware 入力の UTC 保持保証

---

## `tests/test_common_stride.py`（`common.stride`）

### `test_normalize_channel_stride_none`
- `None` 入力時の `None` 返却保証

### `test_normalize_channel_stride_zero_or_negative`
- 0 以下入力時の `ValueError` 送出保証

### `test_normalize_channel_stride_one_or_less_is_none`
- 1（および 1.0）入力時の stride 無効化（`None`）保証

### `test_normalize_channel_stride_gt1_returns_int`
- 2 以上入力時の `int` 正規化保証（float → int 切り捨て）

---

## `tests/test_common_array_util.py`（`common.array_util`）

### `test_as_1d_float_ok`
- 1 次元配列化の成立性
- dtype float 強制の成立性
- 値同等性（`np.allclose`）保証

### `test_as_1d_float_rejects_empty`
- 空配列入力時の `ValueError` 送出保証

### `test_as_1d_float_rejects_ndim_ne1`
- 2 次元以上入力時の `ValueError` 送出保証

---

## `tests/test_common_csv_util.py`（`common.csv_util`）

### `test_open_dict_writer_writes_header_and_rows`
- 親ディレクトリ自動生成の成立性
- ヘッダ行出力の成立性（`write_header=True`）
- 1 行書き込み結果の CSV 文字列一致保証（`a,b` / `1,2`）
