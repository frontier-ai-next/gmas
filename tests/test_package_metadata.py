from importlib import metadata

import gmas


def test_package_version_matches_distribution_metadata() -> None:
    assert gmas.__version__ == metadata.version("frontier-ai-gmas")
