# On Windows, importing torch after Qt has loaded its DLLs fails with
# "WinError 1114 ... c10.dll". Load torch before pytest-qt starts Qt.
import torch  # noqa: F401
