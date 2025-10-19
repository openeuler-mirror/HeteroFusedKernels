# SPDX-License-Identifier: Apache-2.0
# Standard
from pathlib import Path
import os
import sys

# Third Party
from setuptools import find_packages, setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

import glob
import logging
import subprocess
import platform
import shutil

ROOT_DIR = Path(__file__).parent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_ascend_home_path():
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

def _get_ascend_env_path():
    env_script_path = os.path.realpath(
        os.path.join(_get_ascend_home_path(), "..", "set_env.sh")
    )
    if not os.path.exists(env_script_path):
        raise ValueError(
            f"The file '{env_script_path}' is not found, "
            "please make sure environment variable 'ASCEND_HOME_PATH' is set correctly."
        )
    return env_script_path

class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwargs) -> None:
        super().__init__(name, sources=[], **kwargs)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)

class AscendBuildExt(build_ext):
    def build_extension(self, ext):
        ext_name = ext.name.split(".")[-1]
        logger.info(f"Building {ext_name} ...")
        
        BUILD_DIR = os.path.join(ROOT_DIR, "build")
        os.makedirs(BUILD_DIR, exist_ok=True)

        ascend_home_path = _get_ascend_home_path()
        env_path = _get_ascend_env_path()
        
        # Get compilers from environment or use defaults
        _cxx_compiler = os.getenv("CXX", "g++")
        _cc_compiler = os.getenv("CC", "gcc")
        python_executable = sys.executable

        # Get pybind11 cmake path
        try:
            pybind11_cmake_path = subprocess.check_output(
                [python_executable, "-m", "pybind11", "--cmakedir"]
            ).decode().strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"pybind11 not found: {e}")

        # Get torch and torch_npu paths
        try:
            import torch_npu
            torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
        except ImportError:
            raise RuntimeError("torch_npu not installed")
            
        try:
            import torch
            torch_path = os.path.dirname(os.path.abspath(torch.__file__))
        except ImportError:
            raise RuntimeError("torch not installed")
        
        _cxx_11_abi = 1 if torch.compiled_with_cxx11_abi() else 0
        arch = platform.machine()
        install_path = os.path.join(BUILD_DIR, "install")
        if isinstance(self.distribution.get_command_obj("develop"), develop):
            install_path = BUILD_DIR

        # Build CMake command
        cmake_cmd = [
            f"source {env_path} && ",
            f"cmake -S {ROOT_DIR} -B {BUILD_DIR}",
            f" -DCMAKE_CXX_COMPILER={_cxx_compiler}",
            f" -DCMAKE_C_COMPILER={_cc_compiler}",
            f" -DPYTHON_EXECUTABLE={python_executable}",
            f" -DCMAKE_PREFIX_PATH={pybind11_cmake_path}",
            f" -DCMAKE_BUILD_TYPE=Release",
            f" -DTORCH_NPU_PATH={torch_npu_path}",
            f" -DTORCH_PATH={torch_path}",
            f" -DCMAKE_INSTALL_PREFIX={install_path}",
            f" -DARCH={arch}",
            f" -DASCEND_CANN_PACKAGE_PATH={ascend_home_path}",
            f" -DGLIBCXX_USE_CXX11_ABI={str(_cxx_11_abi)}",
            f" && cmake --build {BUILD_DIR} -j{min(16, os.cpu_count())} --verbose",
            f" && cmake --install {BUILD_DIR}"
        ]
        
        cmake_cmd = "".join(cmake_cmd)
        logger.info(f"Running CMake command: {cmake_cmd}")

        try:
            subprocess.run(cmake_cmd, cwd=ROOT_DIR, shell=True, executable="/bin/bash", check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build extension: {e}")

        build_lib_dir = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(build_lib_dir), exist_ok=True)

        package_name = ext.name.split(".")[0]  # e.g., 'heterofusedkernels'
        src_dir = os.path.join(ROOT_DIR, package_name)

        # Expected file patterns (using glob patterns for flexibility)
        expected_patterns = [
            "memory*.so", 
            "libmanaged_memory.so"
        ]

        # Search for files matching our patterns
        so_files = []
        for pattern in expected_patterns:
            # Search in main directory and common subdirectories
            search_paths = [
                install_path,
                os.path.join(install_path, "lib"),
                os.path.join(install_path, "lib64")
            ]

            for search_path in search_paths:
                if os.path.exists(search_path):
                    matches = glob.glob(os.path.join(search_path, pattern))
                    so_files.extend(matches)
        
        # For develop mode, also copy to source directory
        is_develop_mode = isinstance(
            self.distribution.get_command_obj("develop"), develop
        )
        # Remove duplicates
        so_files = list(dict.fromkeys(so_files))
        if not so_files:
            raise RuntimeError(f"No .so files found matching patterns {expected_patterns}")

        logger.info(f"Found {len(so_files)} .so files:")
        for so_file in so_files:
            logger.info(f"  - {so_file}")

        # Copy each file (same copying logic as above)

        def _copy(files):
            for src_path in files:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(os.path.dirname(build_lib_dir), filename)
                
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"Copied {filename} to {dst_path}")
                
                if is_develop_mode:
                    src_dir_file = os.path.join(src_dir, filename)
                    if os.path.abspath(src_path) != os.path.abspath(src_dir_file):
                        if os.path.exists(src_dir_file):
                            os.remove(src_dir_file)
                        shutil.copy2(src_path, src_dir_file)
                        logger.info(f"Copied {filename} to source directory: {src_dir_file}")
        
        _copy(so_files)
        # copy the header
        pattern = os.path.join(install_path, "include", "*.h")
        matches = glob.glob(pattern, recursive=True)
        assert len(matches) > 0, f" {pattern} does not match"
        logger.info(f"Found header files: {matches}")
        _copy(matches)

        logger.info("All files copied successfully")

def get_extensions():
    logger.info("Configuring Ascend extensions with Torch support")
    return [CMakeExtension(name="heterofusedkernels.memory")]

setup(
    description="HeteroFusedKernels Common module with Ascend NPU and Torch support",
    packages=find_packages(exclude=("csrc","build", "dist", "*.egg-info")),
    ext_modules=get_extensions(),
    cmdclass={
        "build_ext": AscendBuildExt,
    },
    package_data={"heterofusedkernels": ["*.so", "managed_memory.h"]},
    include_package_data=True,
)