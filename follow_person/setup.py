from setuptools import setup
import os
from glob import glob

package_name = 'follow_person'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics>=8.0.0',
        'torch>=2.0.0',
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
    ],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='example@example.com',
    description='Person following package using Realsense and Lidar',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'person_follower = follow_person.person_follower:main',
        ],
    },
)
