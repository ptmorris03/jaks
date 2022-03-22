import setuptools

setuptools.setup(
    name='jaks',
    version='1',
    author='Paul Morris',
    author_email='pmorris2012@fau.edu',
    description='jaks',
    long_description='jaks',
    long_description_content_type="text/markdown",
    url='https://github.com/ptmorris03/jaks',
    project_urls = {
        "Bug Tracker": "https://github.com/ptmorris03/jaks/issues"
    },
    license='to kill',
    packages=['jaks', 'jaks.modules'],
    install_requires=['matplotlib', 'numpy', 'tensorflow', 'tfds-nightly', 'tqdm', 'typer'],
)
