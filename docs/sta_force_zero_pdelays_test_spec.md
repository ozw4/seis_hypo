# 追加テスト 技術メモ（`write_hypoinverse_sta` の pdelay ゼロ強制）

## 対象
- 関数: `hypo.sta.write_hypoinverse_sta`
- 変更点: `force_zero_pdelays: bool = False` 引数追加による分岐導入
- リスク: station delay の二重補正／ゼロ強制の不発／既存挙動の破壊（常時ゼロ化など）

---

## テストファイル
- `tests/test_hypo_sta_force_zero_pdelays.py`

---

## 補助関数

### `_read_first_sta_line(path: Path) -> str`
- `.sta` 出力の先頭行取得
- エンコーディング ASCII 読み込みの前提固定
- 固定長フォーマットの維持確認（`len(line) == 86` で fail fast）
- 固定幅パース前提の明文化

---

## テスト 1

### `test_write_hypoinverse_sta_force_zero_pdelays_overrides_csv`
- `force_zero_pdelays=True` 指定時の上書き保証
- CSV に `pdelay1/pdelay2` が非ゼロで与えられても、`.sta` 出力では常に 0.0 となることの保証
- 二重補正回避用途（CRE 等）における仕様成立の保証
- 分岐の未適用、上書き漏れ、条件逆転の検知

#### 検証方法
- 出力 `.sta` の固定幅スライス値の数値化による比較
  - `pdelay1`: `line[49:54]`（col 50–54, F5.2）→ `float(...) == 0.0`
  - `pdelay2`: `line[55:60]`（col 56–60, F5.2）→ `float(...) == 0.0`

---

## テスト 2

### `test_write_hypoinverse_sta_default_preserves_pdelays`
- `force_zero_pdelays=False`（または未指定相当）時の回帰保証
- CSV の `pdelay1/pdelay2` が `.sta` 出力に反映されることの保証
- 既存機能の破壊（常時ゼロ化、分岐の誤適用）の検知

#### 検証方法
- 出力 `.sta` の固定幅スライス値の数値化による比較
  - `pdelay1`: `float(line[49:54]) == 1.23`
  - `pdelay2`: `float(line[55:60]) == 4.56`

---

## 固定幅参照（前提）
- `pdelay1`: col 50–54（F5.2）→ 0-based slice `line[49:54]`
- `pdelay2`: col 56–60（F5.2）→ 0-based slice `line[55:60]`

---

## 本テストにより保証される性質
- `force_zero_pdelays` 分岐の正当性（ゼロ強制の成立）
- デフォルト挙動の不変性（CSV 値の反映）
- `.sta` 行長 86 文字の維持（フォーマット破壊の早期検知）
