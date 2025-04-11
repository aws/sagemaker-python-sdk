"""
Integ test file script_1.py
"""

import pathlib

if __name__ == "__main__":

    print("writing file to /opt/ml/processing/test/test.py...")
    pathlib.Path("/opt/ml/processing/test").mkdir(parents=True, exist_ok=True)
    with open("/opt/ml/processing/test/test.py", "w") as f:
        f.write('print("test...")')
