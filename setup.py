#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

## Grab requirements form file
# with open("requirements.txt") as f:
#     req_list = f.readlines()
#     req_LIST = [str(req) for req in req_list]
req_list = ['scikit-learn>=1.2.2','pandas>=1.5.3','matplotlib>=3.7.1', 'seaborn>=0.12.2']#, 'plotly>=5.15.0']
requirements = ['Click>=7.0', *req_list  ]

test_requirements = [ ]

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
        
    ],
    description="Code from Coding Dojo's Online Part-Time Data Science boot camp",
    entry_points={
        'console_scripts': [
            'dojo_ds=dojo_ds.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dojo_ds',
    name='dojo_ds',
    packages=find_packages(include=['dojo_ds', 'dojo_ds.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/coding-dojo-data-science/dojo_ds',
    version='1.0.9',
    zip_safe=False,
)
