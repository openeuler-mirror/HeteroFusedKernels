import pybind11
import datetime
import io
import os
import platform
import re
from typing import List, Optional
import setuptools
import torch
import torch.utils.cpp_extension as TorchExtension
from setuptools.command.build_py import build_py as _build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PKG_ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
PYTORCH_NPU_INSTALL_PATH = os.path.join(PKG_ROOT_PATH, "torch_npu")
PLATFORM_ARCH = platform.machine() + "-linux"

class CustomBuildExtension(BuildExtension):
    def run(self):
        super().run()
        
def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)

def read_readme()->str:
    with open(get_path("README.md"), "r",encoding="utf-8") as f:
        return f.read()

def _find_cann_home()->Optional[str]:
    cann_home = os.environ.get("ASCEND_HOME_PATH")
    if cann_home is None:
        raise RuntimeError("ASCEND_HOME is not set")
    return cann_home

def find_version(filepath:str):
    with open(filepath) as fp:
        content = fp.read()
        version_match = re.search(r"__version__ = \"(.*?)\"", content, re.M)
        if version_match:
            today = datetime.date.today()
            data_str = today.strftime("%Y%m%d")
            data_str = f".dev{data_str}"
            version_and_data = version_match.group(1) + data_str
            return version_and_data
        raise RuntimeError("Unable to find version string.")

class BuildPyWithCompileNpuOps(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

def OECCLExtension(name, sources,*args, **kwargs):
    cann_home = _find_cann_home()
    include_dirs = kwargs.get("include_dirs",[])
    include_dirs.append(os.path.join(cann_home,PLATFORM_ARCH,"include"))
    include_dirs.append(os.path.join(PYTORCH_NPU_INSTALL_PATH,"include"))
    include_dirs.append(os.path.join(cann_home,PLATFORM_ARCH,"include/experiment/msprof/"))
    include_dirs.append(os.path.join(cann_home,PLATFORM_ARCH,"include/hccl"))
    include_dirs.append("/usr/include/")
    include_dirs.append(os.path.join("."))
    include_dirs += TorchExtension.include_paths()
    kwargs["include_dirs"] = include_dirs
    
    library_dirs = kwargs.get("library_dirs",[])
    library_dirs.append(os.path.join(cann_home,PLATFORM_ARCH,"lib64"))
    library_dirs.append(os.path.join(PYTORCH_NPU_INSTALL_PATH,"lib"))
    library_dirs.append("/usr/lib/x86_64-linux-gnu/")
    
    library_dirs += TorchExtension.library_paths()
    kwargs["library_dirs"] = library_dirs
    
    libraries = kwargs.get("libraries",[])
    libraries.append("torch_npu")
    libraries.append("torch_cpu")
    libraries.append("torch")
    libraries.append("torch_python")
    libraries.append("c10")
    libraries.append("ascendcl")
    libraries.append("hccl")
    libraries.append("numa")

    kwargs["libraries"] = libraries
    kwargs["language"] = "c++"
    return CppExtension(name,sources,*args,**kwargs)
    
        
       
ext_modules=[
    OECCLExtension(name="oeccl.ops",sources=["csrc/oeccl_base.cpp"]),
    ]


setuptools.setup(
    name="oeccl",
    version=find_version(get_path("_init_.py")),
    description="openEuler Collective Communication Library",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(
        exclude=("benchmarks","crsc_ascend","examples","tests","tools")    
    ),
    ext_modules=ext_modules,
    python_requires=">=3.9",
    cmdclass={
        "build_ext": CustomBuildExtension,"build_py":BuildPyWithCompileNpuOps
    },
    package_data={
        "oeccl": ["*.so","*.o"],
    },
)
    