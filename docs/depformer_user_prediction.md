# Depformerによるユーザー発話予測の可能性について

## 質問
ファインチューニングによりユーザ発話を予測するdepformerを訓練することは可能か？

## 回答

### 現状：**制限付きで不可**

現在のmoshi-finetuneの実装では、以下の理由により**ユーザー発話を直接予測するdepformer訓練は困難**です：

#### 1. データアノテーションの制約
```python
# annotate.py では左チャンネル（Moshi）のみをアノテート
process_one(path, out_file, channel=0, ...)  # channel=0固定
```
- 右チャンネル（ユーザー）の文字起こしが生成されない
- すべてのアライメントが`SPEAKER_MAIN`（Moshi）としてラベル付け

#### 2. 訓練パイプラインの設計
```python
# train.py:161 - ハードコードされた設定
interleaver = Interleaver(..., keep_main_only=True)
```
- メインスピーカー（Moshi）のテキストのみを学習ターゲットとして使用
- ユーザー発話のテキストアノテーションが学習に含まれない

#### 3. 損失計算の範囲
- テキスト損失: Moshiの発話のみ
- 音声損失: ステレオ全体だが、テキストとの対応がMoshiのみ

### Moshiのアーキテクチャ

**Depth Transformer（depformer）**:
- 各時間ステップでの9つのコードブック間の依存関係をモデル化
- 理論的にはユーザー音声の予測も可能

**Temporal Transformer**:
- 時間的な依存関係をモデル化（7Bパラメータ）

### 理論的可能性：**変更すれば可能**

以下の変更を行えば、ユーザー発話予測の訓練は**理論的に可能**です：

#### 必要な変更

##### 1. アノテーションスクリプトの拡張
```python
# annotate.py の修正例（擬似コード）

# 左チャンネル（Moshi）
left_annotations = process_one(path, channel=0, speaker_label="SPEAKER_MOSHI")

# 右チャンネル（ユーザー） - 新規追加
right_annotations = process_one(path, channel=1, speaker_label="SPEAKER_USER")

# 統合
combined_annotations = {
    "alignments": left_annotations + right_annotations
}
```

##### 2. Interleaverの設定変更
```python
# train.py の修正
interleaver = Interleaver(
    spm,
    mimi.frame_rate,
    model.text_padding_token_id,
    model.end_of_text_padding_id,
    model.zero_token_id,
    keep_main_only=False,      # 変更: 両スピーカーを保持
    use_bos_eos=True,          # 変更: 話者交代マーカーを挿入
    main_speaker_label="SPEAKER_MOSHI",
)
```

##### 3. 損失計算の拡張
```python
# ユーザー発話に対する損失も計算
user_text_loss = compute_loss_with_mask(
    output.text_logits,
    user_text_targets,  # ユーザーのテキストトークン
    output.user_text_mask,
    mode="text",
)

user_audio_loss = compute_loss_with_mask(
    output.logits,
    user_audio_targets,  # ユーザーの音声トークン
    output.user_audio_mask,
    mode="audio",
)

total_loss = moshi_text_loss + moshi_audio_loss + \
             user_text_loss + user_audio_loss
```

##### 4. データセット準備
```
data/
├── conversation_001.wav    # ステレオ（左=Moshi、右=ユーザー）
├── conversation_001.json   # 両チャンネルのアノテーション
    {
      "alignments": [
        ["hello", [0.0, 0.5], "SPEAKER_USER"],
        ["hi there", [0.6, 1.2], "SPEAKER_MOSHI"],
        ["how are you", [1.3, 2.0], "SPEAKER_USER"],
        ...
      ]
    }
```

### 技術的な課題

#### 1. チャンネル分離の品質
- 左右チャンネルが完全に分離されている必要がある
- クロストークがあると学習が不安定になる可能性

#### 2. 話者ダイアライゼーション
- ステレオでない既存データの場合、話者識別が必要
- Whisperのdiarizationや別ツール（pyannote.audio）の使用が必要

#### 3. モデルアーキテクチャの理解
- depformerがどのようにユーザー/Moshiの音声を区別するか
- 条件付けメカニズムの必要性

### 実装の難易度

| タスク | 難易度 | 説明 |
|--------|--------|------|
| アノテーション拡張 | 中 | `annotate.py`の修正が必要 |
| Interleaver変更 | 低 | 既存パラメータの変更のみ |
| 損失計算拡張 | 中〜高 | モデル出力の理解が必要 |
| データ準備 | 高 | 高品質なステレオデータが必須 |

### 推奨アプローチ

#### オプション1: 段階的アプローチ（推奨）
1. まず、Moshiの発話のみで基本的なファインチューニングを行う（現状のまま）
2. 小規模な実験でユーザー発話予測を試す
3. 両者を統合した訓練を行う

#### オプション2: フルデュプレックス訓練
- 最初からユーザーとMoshi両方の発話を予測
- より複雑だが、対話的な振る舞いを学習できる可能性

### 結論

**短期的**: 現在の実装では**不可**（コード変更が必要）

**中長期的**: **可能**だが、以下が必要：
- アノテーションパイプラインの拡張
- 訓練コードの修正
- 高品質なステレオ対話データ
- モデルアーキテクチャの深い理解

**最も実現可能な方法**:
1. まず、元のMoshiリポジトリでdepformerの詳細を確認
2. 小規模な概念実証（両チャンネルのアノテーション）
3. 段階的に訓練パイプラインを拡張

### 参考情報

- Moshiの論文: https://arxiv.org/abs/2410.00037（詳細なアーキテクチャ情報）
- Moshiリポジトリ: https://github.com/kyutai-labs/moshi（depformerの実装）
- 現在のコードベース: 主にMoshi発話の学習にフォーカス
