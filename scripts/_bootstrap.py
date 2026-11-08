"""Make the backend package importable when a script is run from anywhere."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def keep_awake() -> None:
    """Ask Windows not to enter standby on its idle timeout while this process runs.

    Standby during a long run makes its runtimes meaningless. The request ends with the process and changes
    no power setting. The power button and the lid still put the machine to sleep.
    """
    if sys.platform == "win32":
        import ctypes

        es_continuous, es_system_required, es_display_required = 0x80000000, 0x00000001, 0x00000002
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required | es_display_required)
