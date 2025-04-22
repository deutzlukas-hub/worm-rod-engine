from setuptools import setup

setup(
    name='worm-rod-engine',
    version='0.1.0',
    author='Lukas Deutz',
    author_email='DeutzLukas@protonmail.com',
    description=('Implementation of an active Cosserat rod immersed in viscous environment suited to simulate the locomotion of slender organism'),
    packages=['worm_rod_engine'],
    # pip installation of fenics is not stable use environmen.yml instead
    install_requires=[
        #'fenics',
        # 'matplotlib',
        # 'scipy',
        # 'tqdm',
        # 'pint'
    ]
)


