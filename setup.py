from setuptools import setup, find_packages
import setuptools 
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

_version_="0.0.1"
REPO_NAME = "Kidney-Diseases-Classification"
AUTHOR_USER_NAME = "yousefre14"
SRCP_REPO = 'cnnClassifier'
AUTHOR_EMAIL = "yousef.jo.reda14@gmail.com"

setuptools.setup(
    name=SRCP_REPO,
    version=_version_,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small package for CNN based classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

