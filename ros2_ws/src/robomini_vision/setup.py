from setuptools import find_packages, setup

name = 'robomini_vision'

setup(
    name=name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + name]),
        ('share/' + name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    entry_points={'console_scripts': [
        'vision = robomini_vision.node:main',
        'record = robomini_vision.record:main',
    ]},
)
