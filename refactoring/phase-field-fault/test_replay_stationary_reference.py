"""The stationary replay must validate inline overrides before using a donor build."""
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class StationaryReplayTest(unittest.TestCase):
    def test_inherited_required_setting(self):
        replay = Path(__file__).with_name("replay_stationary_reference.py")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_output = root / "output"
            parent_output.mkdir()
            marker = parent_output / "accepted"
            marker.write_text("parent state\n")
            parent = root / "parent.prm"
            included = root / "included.prm"
            settings = {"Output directory": parent_output, "Prescribed faults file": "fault.txt",
                        "Structural point spacing": ".5", "Stress rotation": "Jaumann DG material Lie",
                        "Material profile transport": "bounded MacCormack", "Material tensor inflow": "accepted trace",
                        "Material stress representation": "direct physical stress", "Additional shared libraries": "probe.so",
                        "Composition polynomial degree": "2", "Use discontinuous composition discretization": "true",
                        "Function expression": "0; .02+t"}
            for key in ("Output directory", "Stress rotation"):
                included.write_text(f"set {key} = {settings[key]}\n")
                parent.write_text(f"include {included}\n" + "".join(
                    f"set {name} = {value}\n" for name, value in settings.items() if name != key))
                for flags in ([], ["-O"]):
                    with self.subTest(setting=key, flags=flags):
                        dest = root / "replay"
                        result = subprocess.run([sys.executable, *flags, replay, root, root, parent, dest],
                                                capture_output=True, text=True)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn(f"missing inline setting {key}", result.stderr)
                        self.assertFalse(dest.exists())
                        self.assertEqual(list(parent_output.iterdir()), [marker])
                        self.assertEqual(marker.read_text(), "parent state\n")


if __name__ == "__main__":
    unittest.main()
