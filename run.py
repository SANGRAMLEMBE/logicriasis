import subprocess, sys, threading, time
import uvicorn

LOG_FILE = "/tmp/training.log"


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
print("[RUN] Training will begin in 8s. Starting API server on :7860")
uvicorn.run("api.app:app", host="0.0.0.0", port=7860)
