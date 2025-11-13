# Moshi ファインチューニング完全ガイド

このドキュメントは、Moshi音声対話モデルをLoRA（Low-Rank Adaptation）を使用してファインチューニングするための詳細なガイドです。

## 目次

1. [概要](#概要)
2. [環境構築](#環境構築)
3. [データセット準備](#データセット準備)
4. [設定ファイルの詳細](#設定ファイルの詳細)
5. [トレーニングの実行](#トレーニングの実行)
6. [チェックポイントと推論](#チェックポイントと推論)
7. [トラブルシューティング](#トラブルシューティング)
8. [技術的詳細](#技術的詳細)

---

## 概要

**Moshi-Finetune**は、[Moshiモデル](https://github.com/kyutai-labs/moshi)を**LoRA（低ランク適応）**を使用して効率的にファインチューニングするためのフレームワークです。

### 主な特徴

- **LoRAによる軽量学習**: モデル全体ではなく、少数のパラメータのみを学習
- **ステレオ音声対応**: 左チャンネル（Moshi）と右チャンネル（ユーザー）を分離して処理
- **FSDP対応**: PyTorchのFullyShardedDataParallelによる分散学習
- **自動文字起こし**: Whisperを使用した音声データの自動アノテーション

### システム要件

- Python 3.10以上
- CUDA対応GPU（推奨: H100以上）
- 十分なGPUメモリ（単一H100で約40GB）

---

## 環境構築

### 方法1: uv を使用（推奨）

[`uv`](https://docs.astral.sh/uv/)は高速なパッケージマネージャーで、pipの約10倍高速です。

```bash
# uvのインストール（まだの場合）
curl -LsSf https://astral.sh/uv/install.sh | sh

# リポジトリのクローン
git clone git@github.com:kyutai-labs/moshi-finetune.git
cd moshi-finetune

# 依存関係は自動的にインストールされます
# すべてのコマンドに "uv run" を前置するだけです
uv run torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml
```

### 方法2: pip を使用

```bash
# 仮想環境の作成（推奨）
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# リポジトリのクローン
git clone git@github.com:kyutai-labs/moshi-finetune.git
cd moshi-finetune

# 依存関係のインストール
pip install -e .

# 開発用ツールのインストール（オプション）
pip install -e .[dev]
```

### コード品質ツールのセットアップ

```bash
# pre-commitフックのインストール
pre-commit install

# すべてのファイルでリンターを実行
pre-commit run --all-files

# または個別に実行
flake8 --extend-exclude '.venv,venv,.env,env'
ruff check
```

---

## データセット準備

### データセット形式の理解

Moshi-Finetuneは特定の形式のデータセットを必要とします：

1. **ステレオWAVファイル**
   - 左チャンネル: Moshiの音声出力
   - 右チャンネル: ユーザーの音声入力
   - サンプリングレート: 任意（自動的にリサンプリングされます）

2. **JSONLマニフェストファイル**
   - 各行が1つの音声ファイルを表す
   - フォーマット: `{"path": "相対パス/ファイル.wav", "duration": 秒単位の長さ}`

3. **JSON文字起こしファイル**
   - 各WAVファイルに対応する`.json`ファイル
   - タイムスタンプ付きの文字起こし情報を含む

### ディレクトリ構造の例

```
data/
├── dataset.jsonl              # マニフェストファイル
└── audio/
    ├── conversation_001.wav   # ステレオ音声ファイル
    ├── conversation_001.json  # 文字起こしファイル
    ├── conversation_002.wav
    ├── conversation_002.json
    └── ...
```

### JSONLマニフェストファイルの作成

音声ファイルからマニフェストを自動生成するPythonスクリプト：

```python
import sphn
import json
from pathlib import Path

# 音声ファイルのパスを取得
paths = [str(f) for f in Path("data/audio").glob("*.wav")]

# 音声の長さを取得
durations = sphn.durations(paths)

# JSONLファイルに書き込み
with open("data/dataset.jsonl", "w") as fobj:
    for p, d in zip(paths, durations):
        if d is None:
            continue
        json.dump({"path": p, "duration": d}, fobj)
        fobj.write("\n")
```

### サンプルデータセットのダウンロード

公式のサンプルデータセット（14GB）を使用できます：

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    "kyutai/DailyTalkContiguous",
    repo_type="dataset",
    local_dir="./daily-talk-contiguous"
)
```

### 音声データの自動アノテーション

既存のステレオ音声ファイルから自動的にJSON文字起こしを生成：

```bash
# 基本的な使用方法
python annotate.py data/dataset.jsonl

# SLURMを使用した分散アノテーション（64シャードに分割）
python annotate.py data/dataset.jsonl --shards 64 --partition 'your-partition-name'
```

**アノテーションの仕組み:**
- Whisper（音声認識モデル）を使用
- VAD（Voice Activity Detection）で音声区間を検出
- タイムスタンプ付きで文字起こしを生成
- 左右チャンネルを個別に処理

---

## 設定ファイルの詳細

トレーニングはYAML形式の設定ファイルで制御されます。`example/moshi_7B.yaml`を参考にしてください。

### 基本設定

```yaml
# データ設定
data:
  train_data: 'data/dataset.jsonl'  # 必須: トレーニングデータのパス
  eval_data: ''                     # オプション: 評価データのパス
  shuffle: true                      # データシャッフルの有効化（推奨）

# 実行ディレクトリ
run_dir: "runs/experiment_001"      # 必須: チェックポイントとログの保存先
```

### モデル設定

```yaml
# モデルパス
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"  # HuggingFaceからロード
  # または、ローカルパスを指定
  # moshi_path: "/path/to/moshi_weights.safetensors"
  # mimi_path: "/path/to/mimi_weights.safetensors"
  # tokenizer_path: "/path/to/tokenizer.model"
  # config_path: "/path/to/config.json"
```

### LoRA設定

```yaml
# ファインチューニングモード
full_finetuning: false  # false = LoRA, true = 全パラメータ学習

# LoRA設定
lora:
  enable: true          # LoRAの有効化
  rank: 128             # LoRAのランク（推奨: ≤128）
  scaling: 2.0          # LoRAのスケーリング係数
  ft_embed: false       # 埋め込み層も学習する場合はtrue
```

**LoRAパラメータの説明:**
- `rank`: LoRA行列のランク。大きいほど表現力が増すが、メモリ使用量も増加
- `scaling`: LoRA出力のスケーリング。通常は2.0が推奨
- `ft_embed`: 埋め込み層もファインチューニングするかどうか

### 損失関数の重み設定

```yaml
first_codebook_weight_multiplier: 100.0  # 最初のコードブック（意味トークン）の重み
text_padding_weight: 0.5                 # テキストパディングトークンの重み
```

**重要な概念:**
- Mimiエンコーダは9つのコードブックを生成します
- 最初のコードブック（codebook 0）は**意味情報**を含むため、より重要
- `first_codebook_weight_multiplier`を高く設定（例: 100.0）することで、意味の学習を優先
- テキストストリームは主にパディングで構成されるため、`text_padding_weight`を低く設定（例: 0.5）

### 最適化設定

```yaml
# トレーニングハイパーパラメータ
duration_sec: 100        # シーケンスの最大長（秒）
batch_size: 16           # GPU毎のバッチサイズ
max_steps: 2000          # 総トレーニングステップ数
gradient_checkpointing: true  # メモリ節約のためのグラディエントチェックポイント

# オプティマイザ設定
optim:
  lr: 2e-6              # 学習率（推奨開始値: 2e-6）
  weight_decay: 0.1     # 重み減衰（正則化）
  pct_start: 0.05       # ウォームアップの割合（OneCycleLR）
```

**トレーニングパラメータの詳細:**

- **duration_sec**: 長いほど文脈を捉えられるが、メモリ使用量が増加
- **batch_size**: 大きいほど安定するが、メモリ使用量が増加
- **max_steps**: 総処理トークン数 = `max_steps × num_gpus × batch_size × duration_sec × 9 × 12.5`
  - 9 = Mimiの各ステップあたりのトークン数
  - 12.5 = Mimiのフレームレート（Hz）
- **gradient_checkpointing**: メモリ不足時はtrueに設定（若干速度低下）

### チェックポイント設定

```yaml
# チェックポイント設定
do_ckpt: true           # チェックポイント保存の有効化
ckpt_freq: 100          # 保存頻度（ステップ数）
save_adapters: true     # true = LoRAのみ保存, false = 統合モデル保存
num_ckpt_keep: 3        # 保持するチェックポイント数
```

**save_adaptersの選択:**
- `true`: LoRA重みのみを`lora.safetensors`として保存（推奨）
  - メモリ使用量が少ない
  - `python -m moshi.server --lora-weight=...`で直接使用可能
- `false`: LoRAをベースモデルにマージして`consolidated.safetensors`として保存
  - 大量のCPU/GPUメモリが必要
  - スタンドアロンモデルとして使用可能

### ロギングと評価

```yaml
# ロギング
seed: 0                 # 再現性のためのランダムシード
log_freq: 1             # ログ出力頻度（ステップ数）

# 評価
do_eval: false          # 評価の有効化
eval_freq: 100          # 評価頻度（ステップ数）

# Weights & Biases（オプション）
# wandb:
#   project: "moshi-finetune"   # WandBプロジェクト名
#   run_name: "experiment_001"  # 実験名
#   key: "YOUR_API_KEY"         # WandB APIキー
#   offline: false              # オフラインモード
```

---

## トレーニングの実行

### 単一GPU での実行

```bash
# torchrunを使用（単一GPUでも必須）
torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml

# uvを使用する場合
uv run torchrun --nproc-per-node 1 -m train example/moshi_7B.yaml
```

**重要:** 単一GPUの場合でも`torchrun`を使用する必要があります。

### 複数GPU での実行

```bash
# 8 GPUの例
torchrun --nproc-per-node 8 --master_port $RANDOM -m train example/moshi_7B.yaml

# 特定のポートを指定
torchrun --nproc-per-node 8 --master_port 29500 -m train example/moshi_7B.yaml
```

**分散トレーニングのポイント:**
- `--nproc-per-node`: ノードあたりのGPU数
- `--master_port`: 通信用ポート（$RANDOMでランダム選択可能）
- 各GPUは独自のプロセスを実行
- バッチサイズはGPU毎に適用されます

### 期待されるパフォーマンス

推奨設定（`duration_sec: 100`, `batch_size: 16`, `lora.rank: 128`）での性能：

| 構成 | トークン/秒 | ピークメモリ使用量 |
|------|------------|-------------------|
| 1×H100 | 約12,000 | 39.6GB |
| 8×H100 | 約10,700 | 23.7GB |

### トレーニングの監視

```bash
# TensorBoardでログを確認
tensorboard --logdir runs/experiment_001

# WandBを使用する場合（設定ファイルで有効化）
# ブラウザでWandBダッシュボードを開く
```

---

## チェックポイントと推論

### チェックポイントの構造

```
runs/experiment_001/
├── args.yaml                    # トレーニング設定
├── checkpoints/
│   ├── checkpoint_000100/
│   │   └── consolidated/
│   │       ├── lora.safetensors      # LoRA重み（save_adapters=true）
│   │       ├── consolidated.safetensors  # 統合モデル（save_adapters=false）
│   │       └── config.json           # モデル設定
│   ├── checkpoint_000200/
│   └── checkpoint_000300/
└── train/                       # TensorBoardログ
```

### 推論の実行

#### 1. Moshiサーバーのインストール

```bash
# 既にインストール済みの場合はスキップ
pip install git+https://git@github.com/kyutai-labs/moshi.git#egg=moshi&subdirectory=moshi
```

#### 2. LoRA重みを使用した推論

```bash
# save_adapters=true でチェックポイントを保存した場合
CHECKPOINT_DIR="runs/experiment_001/checkpoints/checkpoint_000500"

python -m moshi.server \
  --lora-weight=$CHECKPOINT_DIR/consolidated/lora.safetensors \
  --config-path=$CHECKPOINT_DIR/consolidated/config.json
```

#### 3. 統合モデルを使用した推論

```bash
# save_adapters=false でチェックポイントを保存した場合
CHECKPOINT_DIR="runs/experiment_001/checkpoints/checkpoint_000500"

python -m moshi.server \
  --moshi-weight=$CHECKPOINT_DIR/consolidated/consolidated.safetensors \
  --config-path=$CHECKPOINT_DIR/consolidated/config.json
```

#### 4. Webインターフェースへのアクセス

サーバーが起動したら、ブラウザで表示されるURLにアクセス（通常は`http://localhost:8998`）

---

## トラブルシューティング

### Out of Memory (OOM) エラー

GPUメモリ不足エラーが発生した場合、以下の対策を試してください：

#### 対策1: バッチサイズを減らす

```yaml
batch_size: 8  # 16から8に変更
```

#### 対策2: シーケンス長を減らす

```yaml
duration_sec: 50  # 100から50に変更
```

**注意:** `duration_sec`を小さくしすぎると、推論時にモデルが早く沈黙する可能性があります。

#### 対策3: グラディエントチェックポイントを有効化

```yaml
gradient_checkpointing: true
```

#### 対策4: LoRAランクを減らす

```yaml
lora:
  rank: 64  # 128から64に変更
```

#### 対策5: マイクロバッチを使用

```yaml
num_microbatches: 2  # グラディエント蓄積
batch_size: 8        # 実効バッチサイズは 8×2=16
```

### トレーニングが進まない・収束しない

#### 学習率を調整

```yaml
optim:
  lr: 4e-6  # 2e-6から増やす
```

または

```yaml
optim:
  lr: 1e-6  # 2e-6から減らす
```

#### 最大ステップ数を増やす

```yaml
max_steps: 5000  # 2000から増やす
```

### 音声品質が低い

#### シーケンス長を増やす

```yaml
duration_sec: 150  # より長い文脈
```

#### LoRAランクを増やす

```yaml
lora:
  rank: 256  # より表現力のあるアダプター
```

#### セマンティックトークンの重みを調整

```yaml
first_codebook_weight_multiplier: 150.0  # 100.0から増やす
```

### ディストリビューションエラー

```
RuntimeError: NCCL error
```

**解決策:**
- 異なるポート番号を試す: `--master_port $(shuf -i 20000-29999 -n 1)`
- NCCLデバッグを有効化: `export NCCL_DEBUG=INFO`
- ネットワーク設定を確認

### run_dirが既に存在する

```
RuntimeError: Run dir runs/experiment_001 already exists.
```

**解決策:**
- 異なる`run_dir`を使用
- または、既存のディレクトリを削除: `rm -rf runs/experiment_001`
- または、設定で上書きを許可: `overwrite_run_dir: true`

---

## 技術的詳細

### アーキテクチャ概要

#### 1. トレーニングパイプライン（`train.py`）

トレーニングプロセスは以下の段階で構成されます：

1. **分散セットアップ**: NCCLバックエンドで分散環境を初期化
2. **モデルロード**: MoshiとMimiをHuggingFaceまたはローカルから読み込み
3. **FSDP ラッピング**: メモリ効率的な分散トレーニングのためモデルをラップ
4. **データパイプライン**: ステレオ音声をロード、Mimiでエンコード、音声/テキストトークンをインターリーブ
5. **トレーニングループ**: 順伝播、損失計算、勾配蓄積、混合精度更新
6. **チェックポイント**: LoRAアダプターまたは統合モデルを保存

#### 2. モデルラッピング（`finetune/wrapped_model.py`）

**FSDP戦略:**
- 各Transformerレイヤーが独自のFSDPシャードになる
- `FULL_SHARD`戦略により、パラメータ、勾配、オプティマイザ状態をすべてシャード
- `BACKWARD_PRE`プリフェッチで通信と計算をオーバーラップ

**LoRA処理:**
- LoRAパラメータ（trainable）と凍結パラメータで別々のFSDPグループを作成
- LoRA層の初期化:
  - A行列: Kaiming uniform初期化
  - B行列: ゼロ初期化（学習開始時は恒等写像）
- パラメータフリーズ: LoRAパラメータ（と必要に応じて埋め込み）のみ`requires_grad=True`

#### 3. データ処理（`finetune/data/`）

**データフロー:**

```
ステレオWAVファイル
    ↓
Mimiエンコーダ（12.5Hz、9コードブック）
    ↓
音声トークン [batch, 9, time]
    ↓
テキストトークン（SentencePiece）
    ↓
インターリーバー
    ↓
結合されたストリーム [batch, 1+9, time]
```

**Interleaver（`interleaver.py`）:**
- 音声トークンとテキストトークンを時間的に整合
- 特殊トークン:
  - `text_padding`: テキストストリームのパディング
  - `end_of_text_padding`: 単語の終わりを示す
  - `zero_padding`: 埋め込みの代わりにゼロを使用
- 話者識別（SPEAKER_MAIN）でフィルタリング可能

#### 4. 損失計算（`finetune/loss.py`）

**2種類の損失:**

1. **テキスト損失:**
   ```python
   text_loss = CrossEntropy(text_logits, text_targets)
   text_loss *= text_padding_weight  # パディングトークンの重みを減らす
   ```

2. **音声損失:**
   ```python
   audio_loss = CrossEntropy(audio_logits, audio_targets)
   audio_loss[codebook_0] *= first_codebook_weight_multiplier  # セマンティック重視
   ```

**総損失:**
```python
total_loss = text_loss + audio_loss
```

#### 5. 混合精度トレーニング

**精度管理:**
- **順伝播・逆伝播**: bfloat16でパラメータを保持
- **オプティマイザステップ前**: float32にアップキャスト
- **勾配クリッピング**: float32で実行
- **オプティマイザステップ後**: bfloat16にダウンキャスト

**メリット:**
- メモリ使用量の削減（bfloat16は16ビット）
- 数値安定性の維持（オプティマイザはfloat32）

#### 6. チェックポイント（`finetune/checkpointing.py`）

**LoRAのみ保存（`save_adapters=True`）:**
```python
# 全FSDPモジュールからtrainableパラメータのみを抽出
for module in trainable_modules:
    with module.summon_full_params(offload_to_cpu=True):
        lora_states.update(module.state_dict())
save(lora_states, "lora.safetensors")
```

**統合モデル保存（`save_adapters=False`）:**
```python
# LoRAを元の重みにマージ
for lora_module in model.modules():
    weight_merged = base_weight + lora_A @ lora_B * scaling
save(weight_merged, "consolidated.safetensors")
```

### 分散トレーニングの詳細

**FSDP（Fully Sharded Data Parallel）:**
- データ並列とモデル並列のハイブリッド
- パラメータ、勾配、オプティマイザ状態をGPU間でシャード
- 順伝播時に必要なパラメータのみをall-gatherで収集
- 逆伝播後にreduce-scatterで勾配を集約

**メモリ効率:**
- 8 GPUの場合、各GPUは約1/8のモデルパラメータを保持
- 計算時に必要な部分のみを一時的に収集
- グラディエントチェックポイントでさらにメモリ削減

### LoRAの数学

**標準的な線形層:**
```
y = Wx + b
```

**LoRA適用後:**
```
y = (W + BA)x + b
```

ここで:
- `W`: 凍結されたベース重み [out_dim, in_dim]
- `A`: 学習可能なLoRA行列 [rank, in_dim]
- `B`: 学習可能なLoRA行列 [out_dim, rank]
- `rank << min(in_dim, out_dim)`

**パラメータ数の削減:**
- 元の重み: `out_dim × in_dim`
- LoRA追加: `rank × (in_dim + out_dim)`
- 典型的な削減率: 99%以上（rank=128, dim=4096の場合）

### 音声トークン化（Mimi）

**Mimiエンコーダ:**
- 入力: ステレオ音声波形
- 出力: 9つのコードブック、各12.5Hz
- コードブック0: セマンティック情報（最重要）
- コードブック1-8: 音響的詳細（音色、韻律など）

**時間的整合性:**
- 音声: 12.5 frames/秒 × 9 codebooks = 112.5 tokens/秒
- テキスト: 可変長、音声フレームに同期

---

## 推奨ワークフロー

### 1. 小規模実験から開始

```yaml
# 設定: quick_test.yaml
duration_sec: 50
batch_size: 4
max_steps: 100
lora:
  rank: 64
```

```bash
torchrun --nproc-per-node 1 -m train quick_test.yaml
```

### 2. ハイパーパラメータ調整

- 学習率: `[1e-6, 2e-6, 4e-6]`を試す
- バッチサイズ: メモリが許す限り大きく
- duration_sec: 推論時の望ましい応答長に合わせる

### 3. 本格的トレーニング

```yaml
# 設定: production.yaml
duration_sec: 100
batch_size: 16
max_steps: 5000
lora:
  rank: 128
```

```bash
torchrun --nproc-per-node 8 --master_port $RANDOM -m train production.yaml
```

### 4. 評価とイテレーション

```bash
# 定期的にチェックポイントで推論テスト
python -m moshi.server \
  --lora-weight=runs/production/checkpoints/checkpoint_001000/consolidated/lora.safetensors \
  --config-path=runs/production/checkpoints/checkpoint_001000/consolidated/config.json
```

---

## よくある質問（FAQ）

### Q: LoRAと全パラメータファインチューニングの違いは？

**A:** LoRAは全パラメータの1%未満のみを学習します。メモリ効率が良く、過学習しにくいですが、表現力は若干制限されます。全パラメータファインチューニングはより柔軟ですが、大量のメモリと大規模なデータセットが必要です。

### Q: どのくらいのデータが必要？

**A:** 最低でも数時間分のステレオ会話音声を推奨します。データが多いほど良い結果が得られますが、LoRAは少ないデータでも効果的です。

### Q: duration_secはどう設定すべき？

**A:** 推論時にモデルが維持すべき文脈の長さに合わせます。短い（30-50秒）と早く沈黙する可能性があり、長い（100-150秒）とメモリを多く消費しますが、より長い会話が可能です。

### Q: first_codebook_weight_multiplierの適切な値は？

**A:** デフォルトの100.0から開始し、音声の明瞭さに基づいて調整します。値が高すぎると音響品質が低下する可能性があります。

### Q: トレーニングにどのくらい時間がかかる？

**A:** H100 1台で、推奨設定（max_steps=2000）なら数時間です。8台なら約8倍速くなります。

### Q: 既存のチェックポイントから再開できる？

**A:** 現在、自動的な再開機能は実装されていません。新しいrun_dirを使用するか、手動でチェックポイントをロードする必要があります。

---

## まとめ

このガイドでは、Moshi音声対話モデルのファインチューニングに必要なすべての情報を説明しました：

- 環境構築とデータセット準備
- 詳細な設定パラメータの説明
- トレーニングの実行と監視
- チェックポイントと推論
- トラブルシューティング
- 技術的な実装詳細

さらなる情報は、以下を参照してください：
- [公式README](README.md)
- [Moshiプロジェクト](https://github.com/kyutai-labs/moshi)
- [Issueトラッカー](https://github.com/kyutai-labs/moshi-finetune/issues)

Happy fine-tuning! 🎙️
