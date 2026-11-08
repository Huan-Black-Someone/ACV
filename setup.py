from setuptools import setup, find_packages

setup(
    name="atrag",
    version="1.0.0",
    description="ATraG: Traditional-CV-Guided Adaptive Multimodal Knowledge Graphs Beyond Brute-Force Ensembles",
    author="ATraG Research Team",
    url="https://github.com/Huan-Black-Someone/ACV",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.40.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
