#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

import setuptools

# Function to load requirements from a file
def load_requirements(filename):
    try:
        with open(filename, 'r') as file:
            requirements = []
            for line in file:
                # Remove whitespace and skip comments and empty lines
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
            return requirements
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# ## Grab requirements form file
# with open("./requirements.txt") as f:
#     req_list = f.readlines()
#     req_list = [x.strip() for x in req_list if not x.startswith('#') and x.strip() != '']
#     # req_LIST = [str(req) for req in req_list]
# # req_list = ['scikit-learn>=1.2.2','pandas>=1.5.3','matplotlib>=3.7.1', 'seaborn>=0.12.2'
# #             'statsmodels>=0.13.5']#, 'plotly>=5.15.0']
# requirements = ['Click>=7.0', *req_list  ]

test_requirements = ['sphinx_rtd_theme' ]

setup(
    author="James Irving",
    author_email='jirving@codingdojo.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12'
    ],
    description="Code from Coding Dojo's Online Part-Time Data Science boot camp",
    entry_points={
        'console_scripts': [
            'dojo_ds=dojo_ds.cli:main',
        ],
    },
    install_requires=load_requirements("./requirements.txt"),
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dojo_ds',
    name='dojo_ds',
    packages=find_packages(include=['dojo_ds', 'dojo_ds.*']),
    test_suite='tests',
    tests_require=load_requirements("./requirements.txt") + test_requirements,
    url='https://github.com/coding-dojo-data-science/dojo_ds',
    version='1.1.15',
    zip_safe=False,
)
