# 技術メモ: HypoInverse cmd パッチ（ERR/ERC 強制）テストが保証していること

## 1. 目的
本メモは、以下の変更に対する pytest（ユーザー指定 1〜6）が、何を確認し、どこまでを保証しているかを記録する。

- 新規追加: `src/hypo/hypoinverse_cmd.py`
  - `cmd_token(line: str) -> str | None`
  - `force_err_erc(lines: list[str]) -> list[str]`
- 変更: `src/hypo/synth_eval/hypoinverse_runner.py`
  - `cmd_token` を用いたトークン判定に移行
  - 最終 cmd に `ERR 1.0` / `ERC 0` を強制する処理（`force_err_erc`）を統合

今回の機能要求（誤差スケールの 1 秒仮定正規化）により、HypoInverse cmd において以下が満たされる必要がある。

- `ERR 1.0`（RDERR=1.0）と `ERC 0`（ERCOF=0）が必ず設定される
- これらは `LOC` より前（HypoInverse が有効に解釈するブロック内）に存在する
- テンプレ cmd の内容に依存せず、最終 cmd で上記が成立する


## 2. テストが保証していること

### 2.1 `cmd_token()` の正規化
該当テスト:
- `test_cmd_token_handles_blank_and_comment_lines`
- `test_cmd_token_uppercases_first_token`

保証内容:
- 空行（空文字、空白のみ）はコマンドとして扱わず `None` を返す。
- `*` で始まるコメント行（先頭空白を含む）はコマンドとして扱わず `None` を返す。
- 有効行については、strip 後の先頭トークンを抽出し、必ず大文字化して返す。

意味合い:
- runner 側の置換処理（STA/PRT/SUM/ARC/PHS/SAL/CRH/CRT/CRE 判定）が、大小混在や空行・コメントを含むテンプレに対して安定動作することを前提としている。


### 2.2 `force_err_erc()` の置換（既存 ERR/ERC を強制値に揃える）
該当テスト:
- `test_force_err_erc_replaces_existing_err_and_erc`

保証内容:
- `ERR` 行が存在する場合、内容を必ず `ERR 1.0` に置換する。
- `ERC` 行が存在する場合、内容を必ず `ERC 0` に置換する。
- `LOC` より前に `ERR 1.0` と `ERC 0` がそれぞれ 1 回ずつ存在することを確認する。

意味合い:
- テンプレ側が異なる値（例: `ERR 0.5`, `ERC 3`）を持っていても、最終 cmd では 1 秒仮定に正規化される。


### 2.3 `force_err_erc()` の挿入（欠損時に LOC より前へ注入）
該当テスト:
- `test_force_err_erc_inserts_after_phs_and_before_loc_when_missing`

保証内容:
- `ERR`/`ERC` が欠損していても `LOC` より前に自動挿入される。
- `PHS` が `LOC` より前に存在する場合、`PHS` の直後に `ERR 1.0` → `ERC 0` の順で挿入される（順序が固定される）。

意味合い:
- ERR/ERC をテンプレ運用で書き忘れても、runner が最終 cmd に必ず注入できる。

補足（設計意図）:
- 実装は `PHS` が無い場合に `SAL 1 2` の直後、さらに無い場合に最初の `FIL` の前、さらに無い場合に `LOC` の直前へ挿入する。
- ただし本テストは、最優先ケース（PHS 直後）の成立を最小保証として固定している。


### 2.4 `force_err_erc()` の重複排除（有効ブロック内で 1 回にする）
該当テスト:
- `test_force_err_erc_dedupes_err_and_erc_in_effective_block`

保証内容:
- `LOC` より前（有効ブロック）に `ERR`/`ERC` が複数回出現しても、最終的に以下を満たす。
  - `ERR 1.0` は 1 行のみ
  - `ERC 0` は 1 行のみ
  - `cmd_token(x) == 'ERR'` および `cmd_token(x) == 'ERC'` の行数もそれぞれ 1

意味合い:
- テンプレの記述揺れ（複数定義）により、HypoInverse がどれを採用するか不明確になるリスクを排除する。


### 2.5 runner 統合（CRH テンプレ系）: 既存 I/O 置換 + コメント保持 + ERR/ERC 保証
該当テスト:
- `test_patch_cmd_template_patches_io_names_preserves_comments_and_forces_err_erc`

保証内容:
- `patch_cmd_template(lines)` が従来の I/O 置換を維持する。
  - `STA` → `STA 'stations_synth.sta'`
  - `PHS` → `PHS 'hypoinverse_input.arc'`
  - `PRT` → `PRT 'hypoinverse_run.prt'`
  - `SUM` → `SUM 'hypoinverse_run.sum'`
  - `ARC` → `ARC 'hypoinverse_run_out.arc'`
  - `CRH 1` → `CRH 1 'P.crh'`
  - `CRH 2` → `CRH 2 'S.crh'`
  - それ以外（例: `CRH 3 ...`）は保持される
- 入力のコメント行・空行が保持される（先頭行がコメント、次行が空行のまま出力される）。
- 出力において `LOC` より前に `ERR 1.0` と `ERC 0` がそれぞれ 1 回のみ存在する。
- `PHS` の直後に `ERR` → `ERC` の順で挿入されていることを確認する。

意味合い:
- token 化により置換条件が変わっても、従来の出力ファイル命名とテンプレ保持が壊れていないことを担保する。
- かつ、CRH テンプレ経由でも ERR/ERC 正規化が最終 cmd に反映される。


### 2.6 runner 統合（CRE 分岐）: ファイル I/O を伴う最終 cmd の ERR/ERC 保証
該当テスト:
- `test_patch_cmd_template_for_cre_forces_err_erc_before_loc`

保証内容:
- `patch_cmd_template_for_cre(template, out_cmd, ...)` が生成した出力 cmd ファイルにおいて、`LOC` より前に `ERR 1.0` と `ERC 0` がそれぞれ 1 回のみ存在する。

意味合い:
- CRE 分岐は CRH/CRT/CRE のモデル行処理や SAL 挿入などが絡むが、その経路でも最終 cmd の ERR/ERC 正規化が成立することを担保する。
- 文字列リストだけでなく、実際のファイル書き出しを含めて確認している。


## 3. テストが保証していないこと（非ゴール）
- HypoInverse 本体が `ERR`/`ERC` をどのように解釈し、PRT 出力の誤差楕円体が期待スケールになるか（外部バイナリ依存のため対象外）。
- `force_err_erc()` の挿入位置ロジックの全分岐（SAL/FIL/LOC 直前）を網羅的に検証してはいない（本セットは PHS 直後の最優先ケースを固定確認）。
- `LOC` が存在しないテンプレに対するエラーメッセージ分岐や例外経路は対象外。
- `patch_cmd_template_for_cre()` の I/O 置換（STA/PRT/SUM/ARC/PHS 等）を詳細に検証していない（本セットは ERR/ERC の成立と位置に主眼）。


## 4. メンテナンス指針
- 将来 `ERR`/`ERC` の既定値を変更する場合、テストは必ず追随させ、1 秒仮定正規化の前提が破られないことをレビューで確認する。
- cmd テンプレのバリエーションが増えた場合は、挿入位置の追加分岐（SAL 優先、FIL 優先、LOC 直前フォールバック）を個別テストとして追加する。
