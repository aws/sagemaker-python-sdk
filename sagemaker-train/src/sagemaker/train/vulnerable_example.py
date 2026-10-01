"""Example utility module with intentional security vulnerabilities for Fortress scan testing."""
import os
import subprocess
import sqlite3
import pickle
import tempfile


# Vulnerability 1: Hardcoded credentials
AWS_SECRET_KEY = "FAKE_SECRET_DO_NOT_USE_1234567890abcdef"
DATABASE_PASSWORD = "FAKE_PASSWORD_FOR_TESTING_ONLY"
API_TOKEN = "FAKE_TOKEN_0000000000000000000000000000"


def execute_training_command(user_input):
    """Vulnerability 2: Command injection - unsanitized user input passed to shell."""
    command = f"python train.py --config {user_input}"
    os.system(command)


def get_training_metrics(job_name):
    """Vulnerability 3: SQL injection - string concatenation in SQL query."""
    conn = sqlite3.connect("metrics.db")
    cursor = conn.cursor()
    query = "SELECT * FROM metrics WHERE job_name = '" + job_name + "'"
    cursor.execute(query)
    return cursor.fetchall()


def load_model_config(config_path):
    """Vulnerability 4: Path traversal - no validation on user-supplied path."""
    full_path = os.path.join("/data/configs", config_path)
    with open(full_path, "r") as f:
        return f.read()


def deserialize_model(data):
    """Vulnerability 5: Insecure deserialization - pickle with untrusted data."""
    return pickle.loads(data)


def run_remote_script(url):
    """Vulnerability 6: SSRF - fetching arbitrary URLs without validation."""
    import urllib.request
    response = urllib.request.urlopen(url)
    return response.read()


def process_training_output(output_dir):
    """Vulnerability 7: Command injection via subprocess with shell=True."""
    subprocess.call(f"tar -czf archive.tar.gz {output_dir}", shell=True)
