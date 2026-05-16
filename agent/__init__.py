"""
video2ai.agent
==============
Video2AIの変化検知エンジンとLLM Vision・PyAutoGUIを組み合わせた
自律デスクトップエージェントモジュール。

アーキテクチャ:
    ScreenCapture  → 画面を定期的にキャプチャ
    ChangeDetector → Video2AIの変化検知アルゴリズムで差分スコアを計算
    LLMVision      → 変化が検知されたフレームをLLMに渡して内容を解析
    ActionExecutor → LLMの指示に従ってマウス/キーボードを操作
    SessionLogger  → 全セッションをJSONLで記録
"""
