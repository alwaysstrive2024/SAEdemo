"""
colab_start.py
──────────────
在 Google Colab 里运行此脚本来正确启动 FastAPI 后端。
不能直接用 `python main.py`，因为会阻塞 Colab cell。

使用方法（Colab cell 里）：
    !pip install fastapi uvicorn[standard] pydantic
    exec(open('colab_start.py').read())
"""

import subprocess
import time
import requests
import os
import sys

PORT = 8000

# ── 杀掉占用端口的老进程 ──────────────────────────────────────────────────
try:
    subprocess.run(f"fuser -k {PORT}/tcp", shell=True, capture_output=True)
    time.sleep(1)
except Exception:
    pass

# ── 在后台启动 uvicorn ────────────────────────────────────────────────────
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "main:app",
     "--host", "0.0.0.0",
     "--port", str(PORT),
     "--log-level", "info"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
)

print(f"[startup] uvicorn PID={proc.pid} starting on port {PORT}…")

# ── 等待服务就绪（最多 30 秒）────────────────────────────────────────────
for i in range(30):
    time.sleep(1)
    try:
        r = requests.get(f"http://localhost:{PORT}/", timeout=2)
        if r.status_code == 200:
            print(f"[startup] ✅ FastAPI is UP after {i+1}s")
            print(f"[startup] Response: {r.json()}")
            break
    except Exception:
        print(f"[startup] Waiting… ({i+1}/30)")
else:
    print("[startup] ❌ Server did not start in 30s — check logs below")
    # 打印 uvicorn 输出帮助诊断
    try:
        out, _ = proc.communicate(timeout=2)
        print(out.decode())
    except Exception:
        pass
