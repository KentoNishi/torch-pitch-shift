import setuptools
import sys

with open("readme.md", "r") as f:
    long_description = f.read()


with open("requirements.txt", "r") as f:
    requirements = f.read().split("\n")

versionName = sys.argv[1].replace("refs/tags/v", "")
del sys.argv[1]

setuptools.setup(
    name="torch_pitch_shift",
    version=versionName,
    author="KentoNishi",
    author_email="kento24gs@outlook.com",
    description=long_description.split("\n")[1],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KentoNishi/torch-pitch-shift",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.4",
    install_requires=requirements,
)
