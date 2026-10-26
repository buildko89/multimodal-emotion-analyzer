"""
テキスト感情分析モジュール
発話内容から感情・極性・キーワードを抽出
"""

import re
from typing import Dict, List, Tuple


class TextSentimentAnalyzer:
    """テキスト感情分析クラス"""

    def __init__(self, language: str = "ja"):
        """
        初期化

        Args:
            language: 言語 ("ja", "en")
        """
        self.language = language

        # 感情キーワード辞書（日本語）
        self.emotion_keywords_ja = {
            "positive": [
                "嬉しい", "楽しい", "幸せ", "最高", "素晴らしい", "良い", "好き",
                "ありがとう", "感謝", "助かる", "満足", "いいね", "やった",
                "すごい", "わくわく", "ラッキー", "気持ちいい"
            ],
            "negative": [
                "悲しい", "辛い", "苦しい", "困る", "嫌", "最悪", "ダメ",
                "疲れ", "無理", "もういい", "許せない", "腹が立つ", "イライラ",
                "不安", "心配", "怖い", "嫌い", "失敗"
            ],
            "anger": [
                "怒", "腹が立つ", "イライラ", "ムカつく", "許せない", "頭にくる"
            ],
            "sadness": [
                "悲しい", "辛い", "切ない", "寂しい", "泣", "落ち込"
            ],
            "fear": [
                "怖い", "不安", "心配", "恐", "ドキドキ", "緊張"
            ],
            "joy": [
                "嬉しい", "楽しい", "幸せ", "最高", "やった", "わくわく"
            ]
        }

        # 否定語
        self.negation_words_ja = [
            "ない", "ません", "ぬ", "ず", "ん", "じゃない", "ではない"
        ]

        # フィラー（間投詞）
        self.filler_words_ja = [
            "えっと", "あの", "その", "まあ", "なんか", "ちょっと",
            "うーん", "えー", "あー", "んー"
        ]

    def analyze_sentiment(self, text: str) -> Dict:
        """
        テキストの感情を分析

        Args:
            text: 分析するテキスト

        Returns:
            感情分析結果
        """
        text_lower = text.lower()

        # ポジティブ/ネガティブスコア
        positive_count = sum(1 for word in self.emotion_keywords_ja["positive"] if word in text)
        negative_count = sum(1 for word in self.emotion_keywords_ja["negative"] if word in text)

        # 極性判定
        if positive_count > negative_count:
            polarity = "POSITIVE"
            score = min(positive_count / (positive_count + negative_count + 0.01), 1.0)
        elif negative_count > positive_count:
            polarity = "NEGATIVE"
            score = min(negative_count / (positive_count + negative_count + 0.01), 1.0)
        elif positive_count + negative_count > 0:
            polarity = "NEUTRAL"
            score = 0.5
        else:
            polarity = "UNKNOWN"
            score = None

        # 詳細な感情分類
        emotions = {}
        for emotion, keywords in self.emotion_keywords_ja.items():
            if emotion not in ["positive", "negative"]:
                emotions[emotion] = sum(1 for word in keywords if word in text)

        # 支配的な感情
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if max(emotions.values()) > 0 else "neutral"

        return {
            "polarity": polarity,
            "score": score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "method": "japanese_keyword_rules", "score_kind": "keyword_ratio_not_probability"
        }

    def extract_keywords(self, text: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        重要キーワードを抽出（簡易版）

        Args:
            text: テキスト
            top_n: 抽出する上位N個

        Returns:
            (キーワード, スコア)のリスト
        """
        # 形態素解析がない場合の簡易実装
        # 文字数でフィルタリング
        words = re.findall(r'[ぁ-んァ-ヶー一-龠]+', text)
        words = [w for w in words if len(w) >= 2]  # 2文字以上

        # 頻度カウント
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # スコア計算（頻度 × 文字数）
        word_scores = [(word, freq * len(word)) for word, freq in word_freq.items()]
        word_scores.sort(key=lambda x: x[1], reverse=True)

        return word_scores[:top_n]

    def detect_negation(self, text: str) -> int:
        """
        否定表現の数を数える

        Args:
            text: テキスト

        Returns:
            否定表現の数
        """
        return sum(1 for neg in self.negation_words_ja if neg in text)

    def detect_fillers(self, text: str) -> Dict:
        """
        フィラー（間投詞）を検出

        Args:
            text: テキスト

        Returns:
            フィラー情報
        """
        fillers_found = [filler for filler in self.filler_words_ja if filler in text]
        filler_count = len(fillers_found)

        # テキストの総単語数（大まかな推定）
        total_words = len(re.findall(r'[ぁ-んァ-ヶー一-龠]+', text))
        filler_ratio = filler_count / max(total_words, 1)

        return {
            "fillers": fillers_found,
            "count": filler_count,
            "ratio": filler_ratio,
            "interpretation": self._interpret_filler_ratio(filler_ratio)
        }

    def detect_questions(self, text: str) -> Dict:
        """
        疑問文を検出

        Args:
            text: テキスト

        Returns:
            疑問文情報
        """
        question_markers = ["？", "?", "か", "ですか", "ますか"]
        question_words = ["なぜ", "どう", "何", "いつ", "どこ", "誰", "どれ"]

        has_question_marker = any(marker in text for marker in question_markers)
        question_word_count = sum(1 for word in question_words if word in text)

        return {
            "has_question": has_question_marker or question_word_count > 0,
            "question_word_count": question_word_count,
            "interpretation": "疑問表現あり" if has_question_marker or question_word_count > 0 else "疑問表現なし"
        }

    def detect_repetition(self, text: str) -> Dict:
        """
        繰り返し表現を検出

        Args:
            text: テキスト

        Returns:
            繰り返し情報
        """
        words = re.findall(r'[ぁ-んァ-ヶー一-龠]+', text)

        if len(words) < 2:
            return {"has_repetition": False, "repeated_words": [], "count": 0}

        # 連続する同じ単語を検出
        repeated = []
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) >= 2:
                repeated.append(words[i])

        return {
            "has_repetition": len(repeated) > 0,
            "repeated_words": list(set(repeated)),
            "count": len(repeated),
            "interpretation": "繰り返し表現あり" if len(repeated) > 0 else "繰り返し表現なし"
        }

    def analyze_comprehensive(self, text: str) -> Dict:
        """
        テキストを総合的に分析

        Args:
            text: 分析するテキスト

        Returns:
            総合分析結果
        """
        sentiment = self.analyze_sentiment(text)
        keywords = self.extract_keywords(text)
        negation_count = self.detect_negation(text)
        fillers = self.detect_fillers(text)
        questions = self.detect_questions(text)
        repetition = self.detect_repetition(text)

        # テキストの長さ
        text_length = len(text)
        word_count = len(re.findall(r'[ぁ-んァ-ヶー一-龠]+', text))

        return {
            "text": text,
            "text_length": text_length,
            "word_count": word_count,
            "sentiment": sentiment,
            "keywords": keywords,
            "negation_count": negation_count,
            "fillers": fillers,
            "questions": questions,
            "repetition": repetition,
            "schema_version": 2,
            "method": "japanese_keyword_rules",
            "acoustic_emotion": "not_implemented",
        }

    def _interpret_filler_ratio(self, ratio: float) -> str:
        return 'フィラー比率: ' + format(ratio, '.2f')
