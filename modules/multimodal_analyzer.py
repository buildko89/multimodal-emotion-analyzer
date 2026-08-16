"""
マルチモーダル統合分析モジュール
表情認識 × 音声認識の統合分析
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from modules.text_sentiment import TextSentimentAnalyzer


class MultiModalAnalyzer:
    """マルチモーダル統合分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.text_analyzer = TextSentimentAnalyzer(language="ja")
        
        # 履歴データ
        self.emotion_history = []
        self.speech_history = []
        self.analysis_history = []
        
        # 表情と音声の感情対応マッピング
        self.emotion_mapping = {
            "angry": ["anger", "negative"],
            "disgust": ["negative"],
            "fear": ["fear", "negative"],
            "happy": ["joy", "positive"],
            "sad": ["sadness", "negative"],
            "surprise": ["surprise"],
            "neutral": ["neutral"]
        }
    
    def analyze(
        self,
        emotion_data: Optional[Dict] = None,
        speech_data: Optional[Dict] = None,
        timestamp: Optional[float] = None
    ) -> Dict:
        """
        マルチモーダル統合分析
        
        Args:
            emotion_data: 表情認識データ
            speech_data: 音声認識データ
            timestamp: タイムスタンプ
            
        Returns:
            統合分析結果
        """
        if timestamp is None:
            timestamp = time.time()
        
        result = {
            "timestamp": timestamp,
            "emotion_data": emotion_data,
            "speech_data": speech_data,
            "has_emotion": emotion_data is not None,
            "has_speech": speech_data is not None,
            "contradictions": [],
            "overall_state": {},
            "trust_score": 0.0,
            "recommendations": []
        }
        
        # 音声データがある場合、テキスト感情分析
        if speech_data and "text" in speech_data:
            text_sentiment = self.text_analyzer.analyze_comprehensive(speech_data["text"])
            result["text_sentiment"] = text_sentiment
        
        # 両方のデータがある場合、統合分析
        if emotion_data and speech_data:
            result["contradictions"] = self.detect_contradictions(
                emotion_data, 
                result.get("text_sentiment", {})
            )
            result["overall_state"] = self.calculate_overall_state(
                emotion_data,
                result.get("text_sentiment", {}),
                result["contradictions"]
            )
            result["trust_score"] = self.calculate_trust_score(
                result["contradictions"]
            )
            result["recommendations"] = self.generate_recommendations(
                result["overall_state"],
                result["contradictions"]
            )
        elif emotion_data:
            # 表情のみ
            result["overall_state"] = self._emotion_only_state(emotion_data)
        elif speech_data:
            # 音声のみ
            result["overall_state"] = self._speech_only_state(
                result.get("text_sentiment", {})
            )
        
        # 履歴に追加
        self.analysis_history.append(result)
        
        return result
    
    def detect_contradictions(
        self,
        emotion_data: Dict,
        text_sentiment: Dict
    ) -> List[Dict]:
        """
        表情と音声の矛盾を検出
        
        Args:
            emotion_data: 表情データ
            text_sentiment: テキスト感情分析結果
            
        Returns:
            矛盾のリスト
        """
        contradictions = []
        
        if not emotion_data or not text_sentiment:
            return contradictions
        
        face_emotion = emotion_data.get("emotion", "neutral")
        text_polarity = text_sentiment.get("sentiment", {}).get("polarity", "NEUTRAL")
        
        # 1. 笑顔 × ネガティブ発言
        if face_emotion == "happy" and text_polarity == "NEGATIVE":
            contradictions.append({
                "type": "fake_smile",
                "severity": "medium",
                "description": "笑顔だが、発言内容はネガティブ",
                "interpretation": "作り笑い、感情を隠している可能性",
                "face_emotion": face_emotion,
                "text_polarity": text_polarity
            })
        
        # 2. 悲しい表情 × ポジティブ発言（強がり）
        if face_emotion == "sad" and text_polarity == "POSITIVE":
            contradictions.append({
                "type": "hiding_sadness",
                "severity": "high",
                "description": "悲しい表情だが、ポジティブな発言",
                "interpretation": "本心を隠している、強がっている",
                "face_emotion": face_emotion,
                "text_polarity": text_polarity
            })
        
        # 3. 怒り表情 × 穏やかな発言
        if face_emotion == "angry" and text_polarity in ["POSITIVE", "NEUTRAL"]:
            contradictions.append({
                "type": "suppressed_anger",
                "severity": "high",
                "description": "怒りの表情だが、発言は穏やか",
                "interpretation": "怒りを抑えている、我慢している",
                "face_emotion": face_emotion,
                "text_polarity": text_polarity
            })
        
        # 4. 無表情 × 感情的な発言
        if face_emotion == "neutral" and text_polarity in ["POSITIVE", "NEGATIVE"]:
            sentiment_strength = text_sentiment.get("sentiment", {}).get("score", 0)
            if sentiment_strength > 0.7:
                contradictions.append({
                    "type": "emotional_suppression",
                    "severity": "medium",
                    "description": "無表情だが、感情的な発言",
                    "interpretation": "感情を表に出さないようにしている",
                    "face_emotion": face_emotion,
                    "text_polarity": text_polarity
                })
        
        # 5. フィラーが多い × 自信ありげな表情
        filler_ratio = text_sentiment.get("fillers", {}).get("ratio", 0)
        if filler_ratio > 0.2 and face_emotion in ["happy", "neutral"]:
            contradictions.append({
                "type": "uncertain_confidence",
                "severity": "low",
                "description": "自信ありげだが、発言に迷いが多い",
                "interpretation": "実際は不安・迷いがある",
                "filler_ratio": filler_ratio
            })
        
        return contradictions
    
    def calculate_overall_state(
        self,
        emotion_data: Dict,
        text_sentiment: Dict,
        contradictions: List[Dict]
    ) -> Dict:
        """
        総合的な状態を算出
        
        Args:
            emotion_data: 表情データ
            text_sentiment: テキスト感情分析結果
            contradictions: 矛盾リスト
            
        Returns:
            総合状態
        """
        # 感情スコア
        face_emotion = emotion_data.get("emotion", "neutral")
        face_confidence = emotion_data.get("confidence", 0.0)
        text_polarity = text_sentiment.get("sentiment", {}).get("polarity", "NEUTRAL")
        text_score = text_sentiment.get("sentiment", {}).get("score", 0.5)
        
        # ストレススコア計算
        stress_score = 0
        
        # 表情からのストレス
        if face_emotion in ["angry", "fear", "sad"]:
            stress_score += 30
        elif face_emotion == "disgust":
            stress_score += 25
        
        # テキストからのストレス
        if text_polarity == "NEGATIVE":
            stress_score += 20 * text_score
        
        # テキスト特徴からのストレス
        text_state = text_sentiment.get("overall_state", {})
        if text_state:
            stress_score += text_state.get("stress_score", 0) * 0.3
        
        # 矛盾によるストレス加算
        contradiction_stress = len([c for c in contradictions if c["severity"] in ["high", "medium"]]) * 15
        stress_score += contradiction_stress
        
        stress_score = min(stress_score, 100)
        
        # ポジティブ度計算
        positive_score = 50
        if face_emotion == "happy":
            positive_score += 30 * face_confidence
        if text_polarity == "POSITIVE":
            positive_score += 20 * text_score
        
        positive_score = max(0, min(positive_score, 100))
        
        # 総合判定
        if stress_score < 30 and positive_score > 60:
            state = "良好"
            state_en = "good"
        elif stress_score < 50 and positive_score > 40:
            state = "普通"
            state_en = "normal"
        elif stress_score < 70:
            state = "やや不調"
            state_en = "slightly_poor"
        else:
            state = "不調・高ストレス"
            state_en = "poor_high_stress"
        
        return {
            "state": state,
            "state_en": state_en,
            "stress_score": round(stress_score, 1),
            "positive_score": round(positive_score, 1),
            "face_emotion": face_emotion,
            "face_confidence": round(face_confidence, 2),
            "text_polarity": text_polarity,
            "text_score": round(text_score, 2),
            "contradiction_count": len(contradictions),
            "high_severity_contradictions": len([c for c in contradictions if c["severity"] == "high"])
        }
    
    def calculate_trust_score(self, contradictions: List[Dict]) -> float:
        """
        信頼度スコアを計算（矛盾が少ないほど高い）
        
        Args:
            contradictions: 矛盾リスト
            
        Returns:
            信頼度スコア (0.0 ~ 1.0)
        """
        if not contradictions:
            return 1.0
        
        # 重大度による減点
        penalty = 0
        for contradiction in contradictions:
            if contradiction["severity"] == "high":
                penalty += 0.3
            elif contradiction["severity"] == "medium":
                penalty += 0.15
            else:
                penalty += 0.05
        
        trust_score = max(0.0, 1.0 - penalty)
        return round(trust_score, 2)
    
    def generate_recommendations(
        self,
        overall_state: Dict,
        contradictions: List[Dict]
    ) -> List[str]:
        """
        推奨アクションを生成
        
        Args:
            overall_state: 総合状態
            contradictions: 矛盾リスト
            
        Returns:
            推奨アクションのリスト
        """
        recommendations = []
        
        stress_score = overall_state.get("stress_score", 0)
        state = overall_state.get("state_en", "normal")
        
        # ストレススコアに基づく推奨
        if stress_score > 70:
            recommendations.append("高ストレス状態です。休憩を取ることを強く推奨します")
        elif stress_score > 50:
            recommendations.append("ストレスが蓄積しています。深呼吸やストレッチを試してください")
        
        # 矛盾に基づく推奨
        for contradiction in contradictions:
            if contradiction["type"] == "fake_smile":
                recommendations.append("本心を話せる環境を提供することが重要です")
            elif contradiction["type"] == "hiding_sadness":
                recommendations.append("サポートが必要な可能性があります。注意深く見守ってください")
            elif contradiction["type"] == "suppressed_anger":
                recommendations.append("感情を表現できる安全な場を提供してください")
        
        # 状態に基づく推奨
        if state == "poor_high_stress":
            recommendations.append("専門家への相談を検討してください")
        
        return recommendations
    
    def _emotion_only_state(self, emotion_data: Dict) -> Dict:
        """表情のみの状態評価"""
        face_emotion = emotion_data.get("emotion", "neutral")
        face_confidence = emotion_data.get("confidence", 0.0)
        
        stress_map = {
            "angry": 80,
            "fear": 70,
            "sad": 60,
            "disgust": 50,
            "surprise": 30,
            "neutral": 20,
            "happy": 10
        }
        
        stress_score = stress_map.get(face_emotion, 20)
        
        return {
            "state": "表情のみ",
            "stress_score": stress_score,
            "face_emotion": face_emotion,
            "face_confidence": face_confidence
        }
    
    def _speech_only_state(self, text_sentiment: Dict) -> Dict:
        """音声のみの状態評価"""
        if not text_sentiment:
            return {"state": "音声データなし"}
        
        return {
            "state": "音声のみ",
            "stress_score": text_sentiment.get("overall_state", {}).get("stress_score", 50),
            "text_polarity": text_sentiment.get("sentiment", {}).get("polarity", "NEUTRAL")
        }
    
    def get_history_summary(self, last_n: int = 10) -> Dict:
        """
        履歴のサマリーを取得
        
        Args:
            last_n: 最新N件
            
        Returns:
            サマリー
        """
        recent = self.analysis_history[-last_n:] if len(self.analysis_history) > last_n else self.analysis_history
        
        if not recent:
            return {"count": 0}
        
        stress_scores = [r["overall_state"].get("stress_score", 0) for r in recent if r.get("overall_state")]
        contradiction_counts = [len(r.get("contradictions", [])) for r in recent]
        
        return {
            "count": len(recent),
            "avg_stress": round(np.mean(stress_scores), 1) if stress_scores else 0,
            "max_stress": round(max(stress_scores), 1) if stress_scores else 0,
            "total_contradictions": sum(contradiction_counts),
            "avg_contradictions": round(np.mean(contradiction_counts), 1) if contradiction_counts else 0
        }


def test_multimodal():
    """テスト関数"""
    print("=== マルチモーダル統合分析モジュール テスト ===\n")
    
    analyzer = MultiModalAnalyzer()
    
    # テストケース1: 一致（笑顔 × ポジティブ）
    print("--- テストケース1: 笑顔 × ポジティブ発言 ---")
    result1 = analyzer.analyze(
        emotion_data={"emotion": "happy", "confidence": 0.89},
        speech_data={"text": "今日は本当に嬉しいです！最高の一日でした！"}
    )
    print(f"総合状態: {result1['overall_state']['state']}")
    print(f"ストレススコア: {result1['overall_state']['stress_score']}")
    print(f"矛盾数: {len(result1['contradictions'])}")
    print(f"信頼度: {result1['trust_score']}\n")
    
    # テストケース2: 矛盾（笑顔 × ネガティブ）
    print("--- テストケース2: 笑顔 × ネガティブ発言 ---")
    result2 = analyzer.analyze(
        emotion_data={"emotion": "happy", "confidence": 0.75},
        speech_data={"text": "疲れた...もう無理かもしれない"}
    )
    print(f"総合状態: {result2['overall_state']['state']}")
    print(f"ストレススコア: {result2['overall_state']['stress_score']}")
    print(f"矛盾数: {len(result2['contradictions'])}")
    if result2['contradictions']:
        print(f"矛盾タイプ: {result2['contradictions'][0]['type']}")
        print(f"解釈: {result2['contradictions'][0]['interpretation']}")
    print(f"信頼度: {result2['trust_score']}")
    print(f"推奨: {result2['recommendations']}\n")
    
    # 履歴サマリー
    print("--- 履歴サマリー ---")
    summary = analyzer.get_history_summary()
    print(f"分析回数: {summary['count']}")
    print(f"平均ストレス: {summary['avg_stress']}")
    print(f"総矛盾数: {summary['total_contradictions']}")


if __name__ == "__main__":
    test_multimodal()
