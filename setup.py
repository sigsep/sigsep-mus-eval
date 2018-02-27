import setuptools

if __name__ == "__main__":
    setuptools.setup(
        # Name of the project
        name='museval',

        # Version
        version="0.2.0",

        # Description
        description='Evaluation tools for the SIGSEP MUS database',
        url='https://github.com/sigsep/sigsep-mus-eval',

        # Your contact information
        author='Fabian-Robert Stoeter',
        author_email='mail@faroit.com',

        # License
        license='MIT',

        # Packages in this project
        # find_packages() finds all these automatically for you
        packages=setuptools.find_packages(),

        entry_points={
            'console_scripts': [
                'museval=museval.cli:museval',
                'bsseval=museval.cli:bsseval'
            ],
        },
        # Dependencies, this installs the entire Python scientific
        # computations stack
        install_requires=[
            'musdb>=0.2.0',
            'numpy',
            'scipy',
            'six',
            'simplejson',
            'soundfile',
            'jsonschema'
        ],

        package_data={
            'museval': ['musdb.schema.json'],
        },

        extras_require={
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
            ],
            'docs': [
                'sphinx',
                'sphinx_rtd_theme',
                'numpydoc',
            ]
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Plugins',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=False,
    )
