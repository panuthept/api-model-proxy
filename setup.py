from pathlib import Path

from setuptools import find_packages, setup

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="api-model-proxy",
    version="0.1.2",
    description="A proxy layer for OpenAI-compatible APIs with pre/post-processing hooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="panuthept",
    url="https://github.com/panuthept/api-model-proxy",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="openai proxy api proxy fastapi middleware logging caching rate-limiting",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "openai>=1.0.0",
        "httpx>=0.24.0",
        "pyyaml>=6.0",
    ],
)
