"""
Hello Modal - Quick test to verify Modal is working.

Usage:
    uv run modal run hello_modal.py
    
Expected output:
    ✅ Hello from Modal!
       Running on: modal-runner-xxx
"""
import modal

app = modal.App("hello-workshop")


@app.function()
def hello():
    """Simple function that runs on Modal's infrastructure."""
    import platform
    
    return f"Hello from {platform.node()}!"


@app.local_entrypoint()
def main():
    """Entry point when running `modal run hello_modal.py`."""
    result = hello.remote()
    print(result)
