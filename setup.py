from setuptools import find_packages, setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='code',
    url='https://github.com/nyu-big-data/final-project-group_27/tree/main/Code',
    author='Giulio Duregon, Joby George, Jonah P',
    author_email='gjd9961@nyu.edu',
    # Needed to actually package something
    packages=find_packages(),
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    description='Code Package for Big Data Final Project'
    # We will also need a readme eventually (there will be a warning)
    #,long_description=open('README.md').read(),
)