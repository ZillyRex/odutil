import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="odutil",
    version="0.0.12",
    author="ZillyRex",
    author_email="zillyrain@gmail.com",
    description="A group of utils for object detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZillyRex/odutil",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
