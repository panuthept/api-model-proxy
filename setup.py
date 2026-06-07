from setuptools import setup, find_packages

setup(
    name="api-model-proxy",
    version="0.1.0",
    description="A proxy layer for API-based language models",
    author="panuthept",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[],
)
