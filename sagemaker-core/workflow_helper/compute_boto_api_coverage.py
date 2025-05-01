from sagemaker.core.utils.utils import configure_logging
from sagemaker.core.tools.resources_extractor import ResourcesExtractor


def main():
    """
    This function computes the number of APIs covered and uncovered by sagemaker core to the ones in Botocore.
    """
    configure_logging("ERROR")  # Disable other log messages
    resources_extractor = ResourcesExtractor()
    # Print the number of unsupported Botocore API and supported Botocore API
    print(len(resources_extractor.actions), len(resources_extractor.actions_under_resource))


if __name__ == "__main__":
    main()
