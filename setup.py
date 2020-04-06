import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="audio_dspy",
    version="0.0.4",
    author="Jatin Chowdhury",
    author_email="jatin@ccrma.stanford.edu",
    description="A package for audio DSP tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jatinchowdhury18/audio_dspy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    test_suite='nose.collector',
    tests_require=['nose']
)
