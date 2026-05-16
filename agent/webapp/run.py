"""
Video2AI Desktop Agent Web App - 起動スクリプト

使用方法:
    cd video2ai
    python -m agent.webapp.run [--host HOST] [--port PORT]
"""
import sys
import os
import argparse

# video2ai ルートを sys.path に追加（agent モジュールの import 解決用）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main():
    parser = argparse.ArgumentParser(description="Video2AI Desktop Agent Web App")
    parser.add_argument("--host", default="127.0.0.1", help="バインドアドレス (デフォルト: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="ポート番号 (デフォルト: 8765)")
    args = parser.parse_args()

    import uvicorn
    from agent.webapp.server import app

    print(f"Video2AI Desktop Agent を起動中...")
    print(f"ブラウザで http://{args.host}:{args.port} を開いてください")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
