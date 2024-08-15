from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['typer']

test_requirements = ['pytest>=3', "hypothesis"]

setup(
    author="Chakravarthi Kanduri, Maria Mamica",
    author_email='chakra.kanduri@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Snakemake research project for investigating impact of correlations in the dataset on results of "
                "FDR/FWER corrections for multiple hypotheses testing",
    entry_points={
        'console_scripts': [
            'simulate_data=scripts.cli.simulate_data:execute',
            'simulate_semi_real_world_data=scripts.cli.simulate_semi_real_world_data:execute',
            'statistical_test=scripts.cli.statistical_test:execute',
            'execute_statistical_test=scripts.cli.execute_statistical_test:execute',
            'calculate_histogram_results=scripts.cli.calculate_histogram_results:execute',
            'plot_histograms=scripts.cli.plot_histograms:execute',
            'plot_comparative_boxcharts=scripts.cli.plot_comparative_boxcharts:execute',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fdr, hypothesis testing, multiple hypothesis testing, false discovery rate, false positive rate',
    name='fdr_under_dependencies',
    packages=find_packages(include=['scripts.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/uio-bmi/fdr_under_dependencies',
    version='0.0.1',
    zip_safe=False,
)
