from setuptools import setup, find_packages

# Read the contents of README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='EvoloPy',
    version='4.0.6',
    description='An open source nature-inspired optimization toolbox with parallel processing',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='EvoloPy Team',
    author_email='raneem.qaddoura@gmail.com',
    maintainer='Jaber Jaber',
    maintainer_email='jaber2jabet@gmail.com',  # Replace with your actual email if desired
    url='https://github.com/7ossam81/EvoloPy',
    # Explicitly specify package directories to ensure proper capitalization
    package_dir={'EvoloPy': 'EvoloPy'},
    packages=['EvoloPy', 'EvoloPy.optimizers'],
    include_package_data=True,
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'scipy>=1.5.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.23.0',
        'psutil>=5.8.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    keywords=['optimization', 'meta-heuristic', 'evolutionary', 'parallel-processing'],
    project_urls={
        'Bug Reports': 'https://github.com/7ossam81/EvoloPy/issues',
        'Source': 'https://github.com/7ossam81/EvoloPy',
        'Contributors': 'https://github.com/7ossam81/EvoloPy/graphs/contributors',
    },
    entry_points={
        'console_scripts': [
            'evolopy-run=EvoloPy.cli:run_cli',
        ],
    },
    extras_require={
        'gpu': ['torch>=1.7.0'],
    },
)