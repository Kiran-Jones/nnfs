from setuptools import setup, find_packages

setup(
    name="nnfs",
    version="0.1.0",
    author="Kiran Jones",
    description="A NumPy-based neural, network library",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    url="https://github.com/kiran-jones/nnfs",
    packages=find_packages(include=["py_modules"])
)