"""
Entry point — run the FastAPI server locally.
Usage: python run.py [--port 8000] [--demo]
"""
import argparse
import subprocess
import sys
import os

def run_api(port: int = 8000):
    print(f"Starting LogiCrisis API on http://localhost:{port}")
    print(f"Docs: http://localhost:{port}/docs")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
    ])

def run_demo(port: int = 7860):
    print(f"Starting Gradio demo on http://localhost:{port}")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "demo/app.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--demo", action="store_true", help="Run Gradio demo instead of API")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_api(args.port)
