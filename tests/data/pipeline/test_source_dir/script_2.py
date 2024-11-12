"""
Integ test file script_2.py
"""

if __name__ == "__main__":

    print("reading file: /opt/ml/procesing/test/test.py")
    with open("/opt/ml/processing/test/test.py", "r") as f:
        print(f.read())
