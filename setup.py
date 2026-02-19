import os
from setuptools import setup,find_packages

setup(
    name="catemb",
    version="1.0",
    description="A stereoelectronic-aware catalyst embeddings",
    keywords=[],
    url="https://github.com/licheng-xu-echo/CatEmb",
    author="Li-Cheng Xu",
    author_email="licheng_xu@zju.edu.cn",
    license="MIT License",
    packages=find_packages(),
    install_package_data=True,
    zip_safe=False,
    install_requires=[],
    package_data={"":["*.csv","*.pt"]},
)