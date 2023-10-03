#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['typer']

test_requirements = ['pytest>=3', "hypothesis"]

setup(
    author="Chakravarthi Kanduri",
    author_email='chakra.kanduri@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description=" ",
    entry_points={
        'console_scripts': [
            'simulate_data=scripts.cli_scripts.simulate_data:execute',
            'statistical_test=scripts.cli_scripts.statistical_test:execute'
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fdr_hacking',
    name='fdr_hacking',
    packages=find_packages(include=['fdr_hacking', 'fdr_hacking.*', 'cli_scripts']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/kanduric/fdr_hacking',
    version='0.0.1',
    zip_safe=False,
)
