from setuptools import setup, find_packages

# python dependencies listed here will be automatically installed with the package
install_deps = [
    "numpy",
    "open3d>=0.18.0",
    "opencv-python>=4.11",
    "torch>=2.6.0",
    "torchvision>=0.21.0",
    "typeguard",
    "transformers>=4.49.0",
    "scipy",
    "jinja2",
    "matplotlib",
    "pillow>=11.1.0",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="smart_palletizer",
    version="0.0.0",
    author="Robert Krug",
    author_email="krug.r1@gmail.com",
    maintainer="Robert Krug",
    maintainer_email="krug.r1@gmail.com",
    description=(
        "This package does: palletized box detection and pose estimation."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_deps,
    python_requires=">=3.10.12",
)
