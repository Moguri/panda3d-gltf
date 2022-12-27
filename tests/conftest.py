import os

import pytest
import panda3d.core as p3d

@pytest.fixture
def modelroot():
    return p3d.Filename.from_os_specific(
        os.path.join(
            os.path.dirname(__file__),
            'models',
        )
    )
