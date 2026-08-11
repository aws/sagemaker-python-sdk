import unittest
from sagemaker.serve.validations.check_image_uri import (
    is_1p_image_uri,
    all_accounts,
    validate_hub_ecr_address,
)


class TestValidateHubEcrAddress(unittest.TestCase):
    """Defense-in-depth: reject hub-sourced image URIs that spoof an ECR host.

    Guards the parser-confusion vulnerability where an untrusted hub could publish an EcrAddress
    that looks like ECR but whose real registry host is attacker-controlled, causing
    LocalContainerMode to forward the ECR authorization token to that host.
    """

    def test_valid_ecr_uri_passes(self):
        validate_hub_ecr_address("123456789012.dkr.ecr.us-east-1.amazonaws.com/my-repo:latest")

    def test_valid_ecr_uri_china_partition_passes(self):
        validate_hub_ecr_address("123456789012.dkr.ecr.cn-north-1.amazonaws.com.cn/my-repo:latest")

    def test_attacker_spoofed_host_raises(self):
        with self.assertRaises(ValueError):
            validate_hub_ecr_address("attacker.com/x.dkr.ecr.us-east-1.amazonaws.com/repo:tag")

    def test_ecr_substrings_in_repo_path_raises(self):
        with self.assertRaises(ValueError):
            validate_hub_ecr_address("evil.example.com/a.dkr.ecr.b.amazonaws.com:latest")

    def test_public_image_passes(self):
        # Not ECR-like at all -> not our concern, must pass through untouched.
        for image in ("nginx:latest", "docker.io/library/nginx:latest", "ubuntu"):
            validate_hub_ecr_address(image)

    def test_iso_partition_ecr_host_passes(self):
        # ISO ECR hosts use a non-amazonaws.com TLD; they are legitimate and must NOT fail closed
        # just because the strict pattern does not enumerate them.
        validate_hub_ecr_address("123456789012.dkr.ecr.us-iso-east-1.c2s.ic.gov/my-repo:latest")

    def test_fips_endpoint_passes(self):
        # FIPS endpoints (ecr-fips) lack the ".dkr.ecr." substring, so they are not ECR-like and
        # must pass through rather than fail closed.
        validate_hub_ecr_address("123456789012.dkr.ecr-fips.us-east-1.amazonaws.com/my-repo:latest")

    def test_empty_uri_is_noop(self):
        validate_hub_ecr_address(None)
        validate_hub_ecr_address("")


class TestCheckImageUri(unittest.TestCase):
    def test_is_1p_image_uri_true(self):
        # Use a known 1P account from the list
        image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch:latest"
        self.assertTrue(is_1p_image_uri(image_uri))

    def test_is_1p_image_uri_false(self):
        # Use a non-1P account
        image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom:latest"
        self.assertFalse(is_1p_image_uri(image_uri))

    def test_is_1p_image_uri_another_1p_account(self):
        # Test with another known 1P account
        image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/tensorflow:latest"
        self.assertTrue(is_1p_image_uri(image_uri))

    def test_all_accounts_is_set(self):
        self.assertIsInstance(all_accounts, set)
        self.assertGreater(len(all_accounts), 0)

    def test_all_accounts_contains_known_accounts(self):
        # Verify some known AWS accounts are in the set
        self.assertIn("763104351884", all_accounts)
        self.assertIn("246618743249", all_accounts)


if __name__ == "__main__":
    unittest.main()
