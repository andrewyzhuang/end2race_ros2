import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'end2race_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.pt')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=True,
    maintainer='Mobility Transformation Lab, University of Michigan',
    maintainer_email='andrewyz@umich.edu',
    description='End2Race ROS 2 deployment package for Neobotics NeoRacer',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = end2race_ros2.policy_node:main',
            'lidar_diagnostics_node = end2race_ros2.lidar_diagnostics_node:main',
        ],
    },
)