from setuptools import setup, find_packages

setup(
    name='EvoloPy',  # Replace with your package name
    version='0.1.0',  # Set the version of your package
    description='A framework for metaheuristic optimization algorithms',
    author='Evo-ML',  # Replace with your name or organization
    author_email='raneem.qaddoura@gmail.com',  # Replace with your email
    url='https://github.com/7ossam81/EvoloPy',  # Replace with your GitHub repository URL
    packages=find_packages(where='EvoloPy'),  # Automatically find all the packages inside the EvoloPy directory
    include_package_data=True,  # To include any additional files specified in MANIFEST.in
    install_requires=[  # List of dependencies
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'scikit-learn'
    ],
    classifiers=[  # Optional classifiers for categorizing your project
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache-2.0 license',  # Adjust if you are using a different license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Adjust this based on the Python version you support
)