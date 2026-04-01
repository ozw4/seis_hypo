# JMA震源決定モデル学習データ向けダウンロードフロー分離計画

## 1. 背景

現在の JMA 系ダウンロード処理は `proc/prepare_data/jma` と `src/jma` に存在し、以下のような用途の異なる処理が同じ名前空間・同じ出力先を共有している。

- `get_event_waveform` によるイベント波形取得
- `_active.ch` 作成
- `missing_continuous` 判定
- `get_continuous_waveform` による補完
- 48観測点までの Hi-net 補充
- 可視化・SNR解析・GaMMA変換・QC などの別用途処理

現状は、学習データ作成フロー専用の派生物が event directory 直下に混在している。

例:

- `*_active.ch`
- `*_missing_continuous.txt`
- `*_mapping_log.csv`
- `continuous/`
- `*_continuous_done_*.json`
- `*_fill_to_48_done_*.json`

この状態だと、他の `proc` が event directory を参照したときに、学習データ作成フローの中間生成物と混線しやすい。

## 2. 目的

JMA震源決定モデルの学習データを作るためのダウンロード処理を、既存の JMA 汎用処理から論理的・物理的に分離する。

分離対象は以下の一連の処理である。

1. `get_event_waveform` で検測観測点をまず取得する
2. `_active.ch` を作成し、取得済み station を確定する
3. `measurements.csv` と station mapping に基づき missing station を確定する
4. `get_continuous_waveform` で missing station を補完する
5. 48観測点に満たない場合、Hi-net (`0101`) から震源近傍 station を追加補完する
6. 最終的に 100 Hz・学習用フォーマットへ書き出せる構成にする

## 3. スコープ

### 3.1 この移行で行うこと

- 学習データ作成フロー専用の `proc` を新設する
- 学習データ作成フロー専用の `src` 名前空間を新設する
- 出力先を event directory 直下から専用 flow directory へ移す
- path 解決を専用 helper に集約する
- 既存 `src/jma` の共通ライブラリは極力再利用する
- 既存 `proc/prepare_data/jma` には直接ロジックを足さない
- Codex が小さな差分単位で安全に実行できる作業順に整理する

### 3.2 この移行で行わないこと

- `stationcode_match/v1` の再設計
- `mapping_report.csv` 生成ロジックの全面改修
- 学習データの最終特徴量設計
- JMA 以外のワークフローの整理
- `src/jma/download.py` や `src/jma/win32_reader.py` の全面リライト

## 4. 現状の主要ファイル

### 4.1 現在の workflow entrypoint

- `proc/prepare_data/jma/run_get_event_waveform.py`
- `proc/prepare_data/jma/get_active_ch.py`
- `proc/prepare_data/jma/make_missing_continuous.py`
- `proc/prepare_data/jma/run_get_missing_continuous_waveform.py`
- `proc/prepare_data/jma/run_fill_to_48_stations.py`

### 4.2 現在の共通ロジック

- `src/jma/download.py`
- `src/jma/missing_continuous.py`
- `src/jma/picks.py`
- `src/jma/stationcode_mappingdb.py`
- `src/jma/stationcode_resolve.py`
- `src/jma/stationcode_presence.py`
- `src/jma/win32_reader.py`
- `src/jma/prepare/active_channel.py`
- `src/jma/prepare/event_paths.py`
- `src/jma/prepare/missing_io.py`

### 4.3 現在の混線ポイント

1. `win_event_dir` が `/workspace/data/waveform/jma/event` に固定されている箇所が多い
2. 中間生成物が event directory 直下へ書かれる
3. `event_paths.py` が event directory 直下構成を前提にしている
4. done marker が他用途と衝突しうる命名規則になっている
5. `proc/prepare_data/jma` 配下に別目的の script が多く、見通しが悪い

## 5. 目標アーキテクチャ

### 5.1 新しいディレクトリ構成

```text
proc/
  jma_model_dataset/
    01_run_get_event_waveform.py
    02_get_active_ch.py
    03_make_missing_continuous.py
    04_run_get_missing_continuous_waveform.py
    05_run_fill_to_48_stations.py
    06_export_dataset_100hz.py
    config/

src/
  jma_model_dataset/
    paths.py
    step1_event_waveform.py
    step1_active_channel.py
    step1_missing_targets.py
    step2_missing_continuous.py
    step3_fill_to_48.py
    export_100hz.py
```

### 5.2 出力先の分離

event directory 直下には raw data のみを置き、学習データ作成フロー固有の成果物は専用ディレクトリに置く。

```text
<event_dir>/
  raw/
    *.evt
    *.ch
    *.txt
  flows/
    jma_model_dataset/
      active/
        *_active.ch
      missing/
        *_missing_continuous.txt
        *_mapping_log.csv
      continuous/
        *.cnt
        *.ch
      logs/
        *_continuous_download_log.csv
      done/
        *_continuous_done_*.json
        *_fill_to_48_done_*.json
      export/
        waveforms_100hz.zarr
        stations.csv
        picks.csv
```

## 6. 設計原則

1. **共通部品は再利用する**
   - `src/jma/download.py`
   - `src/jma/win32_reader.py`
   - `src/jma/picks.py`
   - `src/jma/stationcode_*`

2. **workflow は新規 namespace に隔離する**
   - `proc/jma_model_dataset`
   - `src/jma_model_dataset`

3. **既存 script への追記ではなく、新規追加を基本とする**
   - 既存運用を壊さない
   - 比較検証しやすい

4. **path 解決を最初に分離する**
   - 混線の大半は path と出力配置に起因する

5. **1 task = 1 reviewable diff に分割する**
   - Codex に長い一括改修をさせない

6. **フォールバックを暗黙に入れない**
   - エラーは原因を明示して即時失敗
   - 例外的な救済は明示的な step とログで扱う

## 7. ファイル移行方針

### 7.1 既存のまま残すもの

- `src/jma/download.py`
- `src/jma/win32_reader.py`
- `src/jma/station_reader.py`
- `src/jma/picks.py`
- `src/jma/stationcode_mappingdb.py`
- `src/jma/stationcode_presence.py`
- `src/jma/stationcode_resolve.py`
- `src/jma/prepare/missing_io.py`
- `src/jma/prepare/inventory.py`

### 7.2 workflow 側へコピーまたは薄いラッパー化するもの

- `proc/prepare_data/jma/run_get_event_waveform.py`
- `proc/prepare_data/jma/get_active_ch.py`
- `proc/prepare_data/jma/make_missing_continuous.py`
- `proc/prepare_data/jma/run_get_missing_continuous_waveform.py`
- `proc/prepare_data/jma/run_fill_to_48_stations.py`

### 7.3 先に新規作成するもの

- `src/jma_model_dataset/paths.py`
- `src/jma_model_dataset/step1_event_waveform.py`
- `src/jma_model_dataset/step1_active_channel.py`
- `src/jma_model_dataset/step1_missing_targets.py`
- `src/jma_model_dataset/step2_missing_continuous.py`
- `src/jma_model_dataset/step3_fill_to_48.py`

## 8. 実装フェーズ

### Phase 0: 作業ブランチ・安全策

- 専用 branch を作る
- 既存 `proc/prepare_data/jma` は read-only 扱いで開始する
- 既存 path を変える前に、新しい path helper を作る
- 最初の段階では既存 script を削除しない

### Phase 1: path helper の導入

新規に `src/jma_model_dataset/paths.py` を作る。

責務は以下に限定する。

- `flow_root(event_dir)`
- `raw_root(event_dir)`
- `active_dir(event_dir)`
- `missing_dir(event_dir)`
- `continuous_dir(event_dir)`
- `logs_dir(event_dir)`
- `done_dir(event_dir)`
- `export_dir(event_dir)`
- `active_ch_path(event_dir, stem)`
- `missing_txt_path(event_dir, stem)`
- `mapping_log_path(event_dir, stem)`
- `continuous_done_path(event_dir, stem, run_tag, network_code)`
- `fill_to_48_done_path(event_dir, stem, run_tag)`

この phase では、既存 `src/jma/prepare/event_paths.py` は変更しない。

### Phase 2: Step1 の独立化

以下を `proc/jma_model_dataset` 側に作る。

- `01_run_get_event_waveform.py`
- `02_get_active_ch.py`
- `03_make_missing_continuous.py`

実装方針:

- `run_get_event_waveform.py` は raw 出力専用にする
- `_active.ch` は `flows/jma_model_dataset/active/` に出す
- `*_missing_continuous.txt` と `*_mapping_log.csv` は `flows/jma_model_dataset/missing/` に出す
- 既存 `src/jma/missing_continuous.py` をそのまま使わず、必要なら output path だけ差し替えた薄い wrapper を `src/jma_model_dataset/step1_missing_targets.py` として持つ

### Phase 3: Step2 の独立化

- `04_run_get_missing_continuous_waveform.py`
- `src/jma_model_dataset/step2_missing_continuous.py`

実装方針:

- input は `flows/jma_model_dataset/missing/*_missing_continuous.txt`
- output は `flows/jma_model_dataset/continuous/`
- done marker は `flows/jma_model_dataset/done/`
- log は `flows/jma_model_dataset/logs/`
- network selection のロジックは `src/jma/download.py` を再利用する

### Phase 4: Step3 の独立化

- `05_run_fill_to_48_stations.py`
- `src/jma_model_dataset/step3_fill_to_48.py`

実装方針:

- active station と Step2 station を統合して station 数を判定する
- 不足分のみ `0101` から補う
- `monthly_presence.csv` に存在し、イベント月に運用されている station だけを候補にする
- done marker は Step2 と同様に flow 配下へ置く

### Phase 5: 100 Hz export の導入

- `06_export_dataset_100hz.py`
- `src/jma_model_dataset/export_100hz.py`

この phase で、学習用に必要な保存形式を固定する。

例:

- `waveforms_100hz.zarr`
- `stations.csv`
- `picks.csv`
- `metadata.json`

## 9. Codex 向け作業単位

Codex はフォルダまたは Git repository を選んで作業し、複数 agent を並列に扱え、差分を review しやすい isolated worktree 前提で進められる。したがって、今回の移行も「大きなリライト」ではなく「レビューしやすい小さな差分列」に分割する方が適している。OpenAI の公式説明でも、Codex app は複数 agent の並列作業、isolated worktree 上の clean diff review、reusable skills を前提にしている。
参考: OpenAI Codex の getting started / app 紹介 / prompting guide を参照。

### Task 1

`src/jma_model_dataset/paths.py` の追加

受け入れ条件:

- 既存コードを壊さない
- 新規 helper だけで path を解決できる
- event dir 直下に派生物を書かない API になっている

### Task 2

`proc/jma_model_dataset/01_run_get_event_waveform.py` の追加

受け入れ条件:

- raw 専用出力先へ書く
- 既存 `proc/prepare_data/jma/run_get_event_waveform.py` を変更しない
- 取得直後に event directory の raw 構成を作れる

### Task 3

`proc/jma_model_dataset/02_get_active_ch.py` と `src/jma_model_dataset/step1_active_channel.py` の追加

受け入れ条件:

- `_active.ch` が `flows/jma_model_dataset/active/` に出る
- 既存 event dir 直下に `_active.ch` を書かない

### Task 4

`proc/jma_model_dataset/03_make_missing_continuous.py` と `src/jma_model_dataset/step1_missing_targets.py` の追加

受け入れ条件:

- `mapping_report.csv`
- `near0_suggestions.csv`
- `monthly_presence.csv`

を入力として受け、missing 判定結果を flow 配下へ出力する

### Task 5

`proc/jma_model_dataset/04_run_get_missing_continuous_waveform.py` の追加

受け入れ条件:

- `*_missing_continuous.txt` を flow 配下から読む
- `continuous/`、`logs/`、`done/` へ分離して書く
- event dir 直下に done marker を置かない

### Task 6

`proc/jma_model_dataset/05_run_fill_to_48_stations.py` の追加

受け入れ条件:

- Step2 の結果を参照して不足分だけ埋める
- `0101` 候補 station を距離順に選ぶ
- 既存 Step3 script を変更しない

### Task 7

設定ファイルの分離

- `proc/jma_model_dataset/config/*.yaml`

受け入れ条件:

- `win_event_dir` などが新しい flow 前提の path を向く
- 既存 YAML を上書きしない

### Task 8

最終 export 導入

受け入れ条件:

- 100 Hz 波形を学習用に再現可能な形式で保存できる
- export は flow 配下で完結する

## 10. Codex への指示テンプレート

### 10.1 各タスク共通

```text
対象: seis_hypo repository
目的: JMA学習データ作成フローを既存 proc と混線しないように分離する
制約:
- 既存の proc/prepare_data/jma は原則変更しない
- 共通ロジックは src/jma から再利用する
- 新しい workflow は proc/jma_model_dataset と src/jma_model_dataset に追加する
- event directory 直下へ派生物を書かない
- エラーは握りつぶさず、原因がわかる形で失敗させる
- 後方互換レイヤは作らない
作業:
- まず変更対象ファイル一覧と差分方針を出す
- その後、1 task 分だけ実装する
- 変更後に import と path の整合性を確認する
- 変更内容、影響範囲、未実装点を最後に要約する
```

### 10.2 Task 1 用

```text
src/jma_model_dataset/paths.py を新規作成してください。
責務は path helper のみです。
既存 src/jma/prepare/event_paths.py は変更しないでください。
flow root は event_dir / "flows" / "jma_model_dataset" とし、
active, missing, continuous, logs, done, export の各 path helper を実装してください。
path の自動 fallback は不要です。
```

### 10.3 Task 4 用

```text
既存の jma.missing_continuous のロジックを参考にしつつ、
出力先を flow 配下へ分離した wrapper を src/jma_model_dataset/step1_missing_targets.py と
proc/jma_model_dataset/03_make_missing_continuous.py に実装してください。
要件:
- measurements.csv と epicenters.csv を使う
- mapping_report.csv, near0_suggestions.csv, monthly_presence.csv を使う
- output は flows/jma_model_dataset/missing 配下へ出す
- event dir 直下には書かない
- 既存 proc は変更しない
```

## 11. テスト・検証方針

### 11.1 最低限の確認

- import error がない
- path helper が期待通りの path を返す
- 新規 script が flow 配下へだけ書き込む
- 既存 event dir 直下に新たな派生物を作らない

### 11.2 機能確認

1. Step1 実行
   - raw event が取得される
   - `_active.ch` が flow 配下へ出る
   - missing file が flow 配下へ出る

2. Step2 実行
   - `continuous/` に補完波形が出る
   - `done/` と `logs/` が flow 配下に出る

3. Step3 実行
   - 48局未満の event のみ追加補完される
   - 追加 station が `0101` から距離順に選ばれる

4. Export 実行
   - 100 Hz 波形と対応 metadata を再利用可能な形式で保存できる

### 11.3 回帰確認

- 既存 `proc/prepare_data/jma/*.py` が import error なく残る
- 既存 config や既存出力先を書き換えない

## 12. 完了条件

以下を満たしたら移行完了とする。

- 学習データ作成フローが `proc/jma_model_dataset` だけで実行できる
- 中間生成物と done marker が event dir 直下に書かれない
- 既存の `proc/prepare_data/jma` と同時に存在しても混線しない
- 主要 workflow が `src/jma_model_dataset` に集約されている
- 100 Hz export までの導線が確保されている

## 13. 既知の注意点

1. `mapping_outputs.xlsx` を runtime で直接読むのではなく、現状実装は `mapping_report.csv` と `near0_suggestions.csv` を入力としている
2. `get_event_waveform` で取得した `.evt/.ch/.txt` と、補完用 continuous data は性質が異なるので、保存場所と命名を分ける必要がある
3. 100 Hz は現状「読込時 resample」であり、保存時点では固定されていない箇所がある
4. event time の `-10 sec ~ +90 sec` を厳密に保証する export step は別途明示的に実装する必要がある

## 14. 推奨する commit 戦略

- Commit 1: `src/jma_model_dataset/paths.py`
- Commit 2: `proc/jma_model_dataset/01_run_get_event_waveform.py`
- Commit 3: `02_get_active_ch.py` と active helper
- Commit 4: `03_make_missing_continuous.py` と missing helper
- Commit 5: `04_run_get_missing_continuous_waveform.py`
- Commit 6: `05_run_fill_to_48_stations.py`
- Commit 7: config 分離
- Commit 8: export 導入
- Commit 9: README / docs 更新

1 commit 1目的を徹底し、Codex に一度に広範囲を触らせない。
