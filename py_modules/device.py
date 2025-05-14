
import subprocess

def _cuda_present_on_device():
    try:
        subprocess.check_output('nvidia-smi')
        return True
    except Exception:
        return False
    
CUDA_AVAILABLE = _cuda_present_on_device()