import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('end2race_ros2')
    default_config_path = os.path.join(pkg_share, 'config', 'policy_params.yaml')
    default_model_path = os.path.join(pkg_share, 'models', 'end2race_policy_jit_fp16.pt')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock if true'
    )
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_path,
        description='Path to the ROS 2 parameters YAML file'
    )
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=default_model_path,
        description='Path to exported TorchScript policy model'
    )

    policy_node = Node(
        package='end2race_ros2',
        executable='policy_node',
        name='end2race_policy_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'model_path': LaunchConfiguration('model_path'),
            }
        ]
    )

    diagnostics_node = Node(
        package='end2race_ros2',
        executable='lidar_diagnostics_node',
        name='lidar_diagnostics_node',
        output='screen',
        parameters=[
            LaunchConfiguration('config_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ]
    )

    return LaunchDescription([
        use_sim_time_arg,
        config_file_arg,
        model_path_arg,
        policy_node,
        diagnostics_node,
    ])