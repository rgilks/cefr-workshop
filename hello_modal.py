"""
Hello Modal - Quick test to verify Modal is working.

Usage:
    modal run hello_modal.py
    
Expected output:
    Hello from Modal!
    Running on: <hostname>
    GPU available: <True/False>
"""
import modal

app = modal.App("hello-workshop")


@app.function()
def hello():
    """Simple function that runs on Modal's infrastructure."""
    import platform
    import torch
    
    return {
        "message": "Hello from Modal!",
        "hostname": platform.node(),
        "gpu_available": torch.cuda.is_available(),
    }


@app.local_entrypoint()
def main():
    """Entry point when running `modal run hello_modal.py`."""
    result = hello.remote()
    print(f"✅ {result['message']}")
    print(f"   Running on: {result['hostname']}")
    print(f"   GPU available: {result['gpu_available']}")
