import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="seg2stream",
    version="0.0.1",
    author="Daking Liou",
    author_email="lioudaking@foxmail.com",
    description="Real-time pipeline for segmenting text into a sentence stream.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shenduldh/seg2stream",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=["stanza==1.10.1"],
    keywords="sentence segmentation, tts",
)
