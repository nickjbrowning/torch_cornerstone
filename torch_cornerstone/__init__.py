import os
import warnings

import torch

import sysconfig

site_packages = sysconfig.get_paths()["purelib"]


def _lib_path():

    name = "libtorch_cornerstone.so"

    path = os.path.join(os.path.join(
        site_packages, "torch_cornerstone/lib"), name)

    if os.path.isfile(path):
        return path

    raise ImportError(
        "Could not find torch_cornerstone shared library at " + path)

torch.classes.load_library(_lib_path())
