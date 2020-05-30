from os.path import join

SAMPLES_DIRECTORY = "/Users/owahab/Desktop/personal/sagemaker-python-sdk/tests/unit/v2/samples/"


def get_sample_file(filename):
    file_path = join(SAMPLES_DIRECTORY, filename)
    with open(file_path) as file_content:
        return file_content.read()
