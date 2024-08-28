from setuptools import setup, find_packages

setup(
    name='worm-rod-engine',
    version='0.1.0',
    author='Lukas Deutz',
    author_email='DeutzLukas@protonmail.com',
    description=('Implements a active Cosserat rod immersed in viscous environment suited'
        'to simulate the locomotion of slender organism'),
    packages=find_packages(),
)


