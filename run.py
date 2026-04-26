import subprocess, sys, threading, time, os, traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

LOG_FILE = "/tmp/training.log"

# ── Build the app — full API if imports work, minimal fallback if not ──────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from api.app import app
    print("[RUN] Full LogiCrisis API loaded OK")
except Exception as _e:
    traceback.print_exc()
    print(f"[RUN] WARNING: Full API failed to load ({_e}). Starting minimal fallback.")
    app = FastAPI(title="LogiCrisis")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    def root():
        return {"status": "ok", "env": "LogiCrisis", "note": "full API failed to load — check /startup_error"}

    @app.get("/startup_error")
    def startup_error():
        return {"error": str(_e)}


# ── Start training in background ───────────────────────────────────────────────
def start_training():
    time.sleep(8)
    with open(LOG_FILE, "w") as f:
        f.write("[INIT] Starting LogiCrisis GRPO training...\n")
        f.flush()
        proc = subprocess.Popen(
            [sys.executable, "train_on_hf.py"],
            stdout=f, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        proc.wait()
        f.write("[DONE] Training exit code: " + str(proc.returncode) + "\n")


threading.Thread(target=start_training, daemon=True).start()
print("[RUN] Training will begin in 8s. API server starting on :7860")
uvicorn.run(app, host="0.0.0.0", port=7860)
