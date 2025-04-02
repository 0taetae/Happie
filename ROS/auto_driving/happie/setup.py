from setuptools import setup

package_name = 'happie'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SSAFY',
    maintainer_email='teayoung0812@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = happie.my_node:main',
            'controller = happie.controller:main',
            'talker = happie.publisher_member_function:main',
            'listener = happie.subscriber_member_function:main',
            'communication = happie.communication:main',
            'odometry = happie.odometry_node:main',
            'lidar_subscriber = happie.lidar_subscriber:main',
            'image_parser = happie.perception:main',
            'hand_control = happie.handcontrol:main',
            'make_path = happie.make_path:main',
            'path_tracking = happie.path_tracking:main',
            'path_pub = happie.path_pub:main',
            'load_map = happie.load_map:main',
            'a_star = happie.a_star:main',
            'a_star_local_path = happie.a_star_local_path:main',
            'ex_calib = happie.ex_calib:main',
            'human_detector = happie.human_detector:main',
            'seg_binarizer = happie.seg_binarizer:main',
            'run_mapping = happie.run_mapping:main',
            'run_localization = happie.run_localization:main',
        ],
    },
)
