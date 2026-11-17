import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src")
)

from restricted_zone import is_inside_zone


zone = (200, 100, 500, 400)

assert is_inside_zone((300, 250), zone) is True
assert is_inside_zone((100, 100), zone) is False
assert is_inside_zone((600, 500), zone) is False

print("All restricted-zone tests passed.")