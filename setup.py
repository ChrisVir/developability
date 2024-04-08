#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pandas', 'openmm', 'abnumber']

test_requirements = []

mutate1 = 'developability.developability:mutate_antibody'
mutate_many = 'developability.developability:mutate_multiple_antibodies'

setup(
    author="Chris Rivera",
    author_email='crivera@vir.bio',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    description="A project for mAb developability",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            f'mutate_antibody = {mutate1}',
            f'mutate_multiple_antibodies ={mutate_many}'
                          ]
                          },
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='developability',
    name='developability',
    packages=find_packages(include=['developability', 'developability.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com//developability',
    version='0.1.0',
    zip_safe=False,
)
