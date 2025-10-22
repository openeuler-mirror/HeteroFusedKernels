import os
import subprocess
import stat
import shutil
import multiprocessing
import sysconfig
import sys
from distutils.version import LooseVersion
from setuptools import find_packages, setup
from setuptools import Extension
from setuptools.command.build_clib import build_clib
from setuptools.command.build_ext import build_ext
from pathlib import Path
import torch
import torch_npu
import glob
import logging
from torch_npu.utils.cpp_extension import NpuExtension
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BUILD_PERMISSION = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERSION = 'pcie_through'
ROOT_DIR = Path(__file__).parent

import heterofusedkernels

def which(thefile):
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        fnames = [fname]
        for name in fnames:
            if os.access(name, os.F_OK | os.X_OK) and not os.path.isdir(name):
                return name
    return None


def get_cmake_command():
    def _get_version(cmd):
        for line in subprocess.check_output([cmd, '--version']).decode('utf-8').split('\n'):
            if 'version' in line:
                return LooseVersion(line.strip().split(' ')[2])
        raise RuntimeError('no version found')
    "Returns cmake command."
    cmake_command = 'cmake'
    cmake3 = which('cmake3')
    cmake = which('cmake')
    if cmake3 is not None and _get_version(cmake3) >= LooseVersion("3.18.0"):
        cmake_command = 'cmake3'
        return cmake_command
    elif cmake is not None and _get_version(cmake) >= LooseVersion("3.18.0"):
        return cmake_command
    else:
        raise RuntimeError('no cmake or cmake3 with version >= 3.18.0 found')

def _get_npu_soc():
    """
    Retrieves the NPU SoC version by parsing the output of the npu-smi command.

    This function handles two known output formats:
    1. Standard format with "Chip Name" (e.g., Ascend910B4).
    2. A newer format with both "Chip Name" and "NPU Name" (e.g., Ascend910_9392).

    Returns:
        str: The determined SoC version string.
    
    Raises:
        RuntimeError: If the npu-smi command fails or the output is malformed.
    """
    _soc_version = os.getenv("SOC_VERSION")
    if _soc_version:
        return _soc_version

    try:
        npu_smi_cmd = ["npu-smi", "info", "-t", "board", "-i", "0", "-c", "0"]
        full_output = subprocess.check_output(npu_smi_cmd, text=True)

        npu_info = {}
        for line in full_output.strip().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                npu_info[key.strip()] = value.strip()

        chip_name = npu_info.get("Chip Name", None)
        npu_name = npu_info.get("NPU Name", None)

        if not chip_name:
            raise RuntimeError("Could not find 'Chip Name' in npu-smi output.")

        if npu_name:
            # New Format for npu-smi info: "Ascend910_9392"
            _soc_version = f"{chip_name}_{npu_name}"
        else:
            # Old Format for npu-smi info: "Ascend910B4"
            _soc_version = "Ascend"+chip_name

        return _soc_version

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(f"Failed to execute npu-smi command and retrieve SoC version: {e}")
    except RuntimeError as e:
        raise e


class CPPLibBuild(build_clib, object):
    def run(self):
        cmake = get_cmake_command()
        soc_version = _get_npu_soc()

        if cmake is None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        self.cmake = cmake
        build_py = self.get_finalized_command("build_py")
        extension_dir = os.path.join(BASE_DIR, build_py.build_lib, build_py.get_package_dir(VERSION))

        build_dir = os.path.join(BASE_DIR, "build")
        build_type_dir = os.path.join(build_dir)
        output_lib_path = os.path.join(build_type_dir, "lib")
        os.makedirs(build_type_dir, exist_ok=True)
        os.chmod(build_type_dir, mode=BUILD_PERMISSION)
        os.makedirs(output_lib_path, exist_ok=True)
        self.build_lib = os.path.relpath(os.path.join(build_dir))
        self.build_temp = os.path.relpath(build_type_dir)
        python_executable = sys.executable

        try:
            # if pybind11 is installed via pip
            pybind11_cmake_path = (
                subprocess.check_output(
                    [python_executable, "-m", "pybind11", "--cmakedir"]
                )
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as e:
            # else specify pybind11 path installed from source code on CI container
            raise RuntimeError(f"CMake configuration failed: {e}")
        python_include_path = sysconfig.get_path("include", scheme="posix_prefix")

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + os.path.realpath(output_lib_path),
            '-DTORCH_PATH=' + os.path.realpath(os.path.dirname(torch.__file__)),
            '-DTORCH_NPU_PATH=' + os.path.realpath(os.path.dirname(torch_npu.__file__)),
            '-DHETERO_FUSED_KERNELS_PATH=' + os.path.realpath(os.path.dirname(heterofusedkernels.__file__)),
            '-DSOC_VERSION=' + soc_version,
            '-DPYTHON_EXECUTABLE=' + python_executable,
            '-DCMAKE_PREFIX_PATH=' + pybind11_cmake_path,
            '-DPYTHON_INCLUDE_PATH=' + python_include_path
            ]

        if torch.compiled_with_cxx11_abi():
            cmake_args.append('-DGLIBCXX_USE_CXX11_ABI=1')
        else:
            cmake_args.append('-DGLIBCXX_USE_CXX11_ABI=0')

        max_jobs = os.getenv("MAX_JOBS", str(min(16, multiprocessing.cpu_count())))
        build_args = ['-j', max_jobs]

        subprocess.check_call([self.cmake, BASE_DIR] + cmake_args, cwd=build_type_dir, env=os.environ)
        for base_dir, dirs, files in os.walk(build_type_dir):
            for dir_name in dirs:
                dir_path = os.path.join(base_dir, dir_name)
                os.chmod(dir_path, mode=BUILD_PERMISSION)
            for file_name in files:
                file_path = os.path.join(base_dir, file_name)
                os.chmod(file_path, mode=BUILD_PERMISSION)

        subprocess.check_call(['make'] + build_args, cwd=build_type_dir, env=os.environ)
        dst_dir = os.path.join(extension_dir, 'lib')
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.copytree(output_lib_path, dst_dir)


class Build(build_ext, object):

    def run(self):
        self.run_command('build_clib')
        lib_dir = os.path.abspath(os.path.join(BASE_DIR, "build", "lib"))
        self.library_dirs.append(lib_dir)
        logger.info(f"Lib Dirs: {self.library_dirs}")
        super(Build, self).run()
        package_lib_dest = os.path.join(self.build_lib, VERSION, "lib")
        os.makedirs(package_lib_dest, exist_ok=True)
        logger.info(f"Package Lib Dirs: {package_lib_dest}")

        built_sofiles = glob.glob(os.path.join(lib_dir, "*.so"))
        logger.info(f"Found so files: {built_sofiles}")

        dev_dst_lib = os.path.join(BASE_DIR, VERSION, "lib")
        os.makedirs(dev_dst_lib, exist_ok=True)

        copied_so = set()

        for so_file in built_sofiles:
            filename = os.path.basename(so_file)
            dst_path = os.path.join(package_lib_dest, filename)
            if filename in copied_so:
                continue
            copied_so.add(filename)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.copy2(so_file, dst_path)
            logger.info(f"Copied {so_file} to {dst_path}")
            if self.inplace:
                logger.info("In editable mode - copy to basedir")
                dev_dst_path = os.path.join(dev_dst_lib, filename)
                if os.path.exists(dev_dst_path):
                    os.remove(dev_dst_path)
                shutil.copy2(so_file, dev_dst_path)
                logger.info(f"Copied {so_file} to {dev_dst_path}")

        # remove dummy _C.so
        dummy_so = os.path.join(self.build_lib, VERSION, "_C*.so")
        dummy_dev_so = os.path.join(BASE_DIR, VERSION, "_C*.so")
        for search_path in [dummy_so, dummy_dev_so]:
            for f in glob.glob(search_path):
                os.remove(f)

setup(name=VERSION,
      description='PcieThrough Transfer Kernel for Ascend NPUs',
      packages=find_packages(),
      ext_modules=[NpuExtension(f"{VERSION}._C", sources=[])],
      package_data={
        VERSION: ["lib/*.so"],
      },
      include_package_data=True,
      cmdclass={
          'build_clib': CPPLibBuild,
          'build_ext': Build,
                }
      )
