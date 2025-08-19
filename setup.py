"""
Setup script for Multi-Class Image Classification project
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="multi-class-image-classification",
    version="1.0.0",
    author="Deep Learning Practitioner",
    author_email="contact@example.com",
    description="Multi-Class Image Classification with Deep Neural Networks achieving 98%+ accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/multi-class-image-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=3.0',
            'black>=21.0',
            'flake8>=4.0',
            'pre-commit>=2.15',
        ],
        'gpu': [
            'tensorflow-gpu>=2.15.0',
        ],
        'visualization': [
            'tensorboard>=2.7.0',
            'wandb>=0.12.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'image-classify=main:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.yml', '*.yaml'],
    },
    keywords=[
        'deep learning', 'image classification', 'computer vision',
        'neural networks', 'tensorflow', 'keras', 'CNN', 'machine learning'
    ],
    project_urls={
        'Bug Reports': 'https://github.com/username/multi-class-image-classification/issues',
        'Source': 'https://github.com/username/multi-class-image-classification',
        'Documentation': 'https://github.com/username/multi-class-image-classification/wiki',
    },
)
