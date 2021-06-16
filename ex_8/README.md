# 第八回B4輪講課題

## 課題の概要

本課題では，モデルの予測を行う．

## 課題

データからモデルを予測してみよう！

## 課題の進め方

- データの読み込み
  - 必要なデータはpickleで配布する
  - 読み込み方法や階層構造はヒントを参照
- 出力系列の尤度を計算
  - ForwardアルゴリズムとViterbiアルゴリズムを実装
  - 出力系列ごとにどのモデルから生成されたか推定
- 正解ラベルと比較
  - 混同行列 (Confusion Matrix) を作成
  - 精度 （Accuracy） を算出
  - アルゴリズムごとの精度を計算時間を比較
- 発表 （次週）
  - 取り組んだ内容を周りにわかるように説明
  - コードの解説
    - 工夫したところ，苦労したところの解決策はぜひ共有しましょう
  - 発表者は当日にランダムに決めるので**スライドは全員準備**
  - 結果の考察，応用先の調査など
  - 発表資料は研究室NASにアップロードしておくこと (`/procyon/all/発表資料\B4輪講/2021/<発表日>`)

## ヒント

- pickleデータの読み込み

```python
import pickle

data = pickle.load(open("data1.pickle", "rb"))
```

pickleデータの中にはディクショナリ型でデータが入っている

- dataの階層構造

```
data
├─answer_models # 出力系列を生成したモデル（正解ラベル）
├─output # 出力系列
└─models # 定義済みHMM
  ├─PI # 初期確率
  ├─A # 状態遷移確率行列
  └─B # 出力確率
```

- data1とdata2はLeft-to-Right HMM
- data3とdata4はErgodic HMM

## 結果例

![result](./figs/result.png)

## 余裕がある人は

- 出来る限り可読性，高速化を意識しましょう
  - 冗長な記述はしていませんか
  - for文は行列演算に置き換えることはできませんか
- 関数は一般化しましょう
  - 課題で与えられたデータ以外でも動作するようにしましょう
  - N次元の入力にも対応できますか
- 処理時間を意識しましょう
  - どれだけ高速化できたか，`scipy`の実装にどれだけ近づけたか
  - pythonで実行時間を測定する方法は[こちら](http://st-hakky.hatenablog.com/entry/2018/01/26/214255)

## 注意

- 武田研究室の場合はセットアップで作成した`virtualenv`環境を利用すること  
  - アクティベート例：`source ~/workspace3/myvenv/bin/activate`  
  - アクティベート後`pip install ...`でライブラリのインストールを行う  
- 自分の作業ブランチで課題を行うこと
- プルリクエストを送る前に[REVIEW.md](https://github.com/TakedaLab/B4Lecture/blob/master/REVIEW.md)を参照し直せるところは直すこと
- プルリクエストをおくる際には**実行結果の画像も載せること**
- 作業前にリポジトリを最新版に更新すること

```
$ git checkout master
$ git fetch upstream
$ git merge upstresam/master
```
