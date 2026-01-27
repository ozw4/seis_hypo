* ==========================================================
*  Hypoinverse command TEMPLATE for JMA2001A 1D P+S model
*  ここにある「TUNE_*」ブロックの行だけを編集・置換して使う想定。
*
*  例: @hypoinverse jma2001a_template.cmd
* ==========================================================

* ---- 年代設定（Y2000 フォーマット）----
200 T 1900 0

* ---- station / phase フォーマットとマッチング設定 ----
H71 1 1 3      * station format #2 を使用
LES 3          * 3文字チャンネル（HHZなど）でマッチ
LET 5 1 0       * SITE 5文字 + NET 1文字 + CHAN 3文字

* ==========================================================
*  TUNE_BLOCK: ここから下のブロックの行をパイプラインで置換してよい
* ==========================================================

* ---- 速度モデル（CRH ファイルパス）----
*  P 波モデル
CRH 1 '/workspace/data/velocity/jma_crh/JMA2001A_P.crh'
*  S 波モデル
CRH 2 '/workspace/data/velocity/jma_crh/JMA2001A_S.crh'

*  P モデルと S モデルの対応付け
SAL 1 2

* ---- 観測点リスト ----
STA '/workspace/data/station/jma/stations_hypoinverse_w_das.sta'

* ---- 出力ファイル設定（作業フォルダ相対パス）----
PRT 'hypoinverse_run.prt'
SUM 'hypoinverse_run.sum'
ARC 'hypoinverse_run_out.arc'

* ---- 入力位相ファイル（作業フォルダ相対パス）----
PHS 'hypoinverse_input.arc'
WET 1.0 0.5 0.3 0.2
* ==========================================================
*  TUNE_BLOCK: ここまで
* ==========================================================

* ---- 入力位相ファイルのフォーマットを自動判定 ----
FIL

* ---- 震源決定の実行 ----
LOC

* ---- プログラム終了 ----
STO
