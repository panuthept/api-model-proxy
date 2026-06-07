from setuptools import setup, find_packages

setup(
    name="api-model-proxy",
    version="0.1.0",
    description="A proxy layer for API-based language models",
    author="panuthept",
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
