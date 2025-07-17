from distutils.command.install_lib import install_lib as _install_lib
from setuptools import setup, find_packages
import os
import re

import sys
from setuptools import Extension, setup
from setuptools.command.bdist_egg import bdist_egg
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel
import subprocess

ROOT = os.path.realpath(os.path.dirname(__file__))


class universal_wheel(bdist_wheel):
    # When building the wheel, the `wheel` package assumes that if we have a
    # binary extension then we are linking to `libpython.so`; and thus the wheel
    # is only usable with a single python version. This is not the case for
    # here, and the wheel will be compatible with any Python >=3.6. This is
    # tracked in https://github.com/pypa/wheel/issues/185, but until then we
    # manually override the wheel tag.
    def get_tag(self):
        tag = bdist_wheel.get_tag(self)
        # tag[2:] contains the os/arch tags, we want to keep them
        return ("py3", "none") + tag[2:]


class bdist_egg_disabled(bdist_egg):
    """Disabled version of bdist_egg

    Prevents setup.py install performing setuptools' default easy_install,
    which it should never ever do.
    """

    def run(self):
        sys.exit(
            "Aborting implicit building of eggs. "
            + "Use `pip install .` or `python setup.py bdist_wheel && pip "
            + "install dist/sphericart-*.whl` to install from source."
        )


class cmake_ext(build_ext):
    """Build the native library using cmake"""

    def run(self):
        source_dir = os.path.join(ROOT, "torch_cornerstone")
        build_dir = os.path.join(ROOT, "build", "cmake-build")
        install_dir = os.path.join(
            os.path.realpath(self.build_lib), "torch_cornerstone")

        os.makedirs(build_dir, exist_ok=True)

        cmake_options = [f"-DCMAKE_INSTALL_PREFIX={install_dir}"]

        CUDA_HOME = os.environ.get("CUDA_HOME")

        # ARCHFLAGS is used by cibuildwheel to pass the requested arch to the
        # compilers
        ARCHFLAGS = os.environ.get("ARCHFLAGS")
        if ARCHFLAGS is not None:
            cmake_options.append(f"-DCMAKE_C_FLAGS={ARCHFLAGS}")
            cmake_options.append(f"-DCMAKE_CXX_FLAGS={ARCHFLAGS}")

        if CUDA_HOME is not None:
            subprocess.run(
                ["cmake", source_dir, *cmake_options],
                cwd=build_dir,
                check=True,
            )
            subprocess.run(
                ["cmake", "--build", build_dir, "--target", "install"],
                check=True,
            )
        else:
            print("Warning... CUDA_HOME is None")


if __name__ == "__main__":
    setup(
        version="0.1",
        ext_modules=[
            Extension(name="tradas_blocks", sources=[]),
        ],
        cmdclass={
            "build_ext": cmake_ext,
            "bdist_egg": bdist_egg if "bdist_egg" in sys.argv else bdist_egg_disabled,
            "bdist_wheel": universal_wheel,
        },
        package_data={
            "torch_cornerstone": [
                "torch_cornerstone/lib/*",
                "torch_cornerstone/include/*",
            ]
        },
    )
