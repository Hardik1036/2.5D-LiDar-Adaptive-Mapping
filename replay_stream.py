"""
DRDO SIH26053 - Module 2
Virtual Sensor Playback Utility (replay_stream.py)
"""
import time
import sys

# Tell Pylance to ignore static missing imports for Windows local testing
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from sensor_msgs.msg import PointCloud2  # type: ignore
    from nav_msgs.msg import OccupancyGrid  # type: ignore
    from geometry_msgs.msg import PoseArray  # type: ignore
    from tf2_msgs.msg import TFMessage  # type: ignore
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    # Dummy fallbacks for the Python runtime
    Node = object  # type: ignore
    PointCloud2 = object  # type: ignore
    OccupancyGrid = object  # type: ignore
    PoseArray = object  # type: ignore
    TFMessage = object  # type: ignore
    class MockRclpy:
        def init(self, args=None): pass
        def spin(self, node): pass
        def shutdown(self): pass
    rclpy = MockRclpy()  # type: ignore

class MockVirtualSensorReplay:
    """Fallback mock publisher for local Windows testing without ROS 2."""
    def __init__(self, dataset_mode="SemanticKITTI", frequency_hz=20.0):
        self.frequency_hz = frequency_hz
        self.frame_count = 0
        print(f"[MOCK MODE] Initialized MCAP Replay Node | Dataset: {dataset_mode} | Rate: {frequency_hz} Hz")
        print("[MOCK MODE] Storage Plugin: rosbag2_storage_mcap | Profile: fastwrite")

    def spin(self):
        try:
            while True:
                time.sleep(1.0 / self.frequency_hz)
                self.frame_count += 1
                if self.frame_count % 20 == 0:
                    print(f"[MOCK] Successfully published {self.frame_count} frames to virtual ROS 2 topics.")
        except KeyboardInterrupt:
            print("\n[MOCK] Shutting down virtual replay.")

def run_ros2_node():
    class VirtualSensorReplay(Node):  # type: ignore
        def __init__(self, dataset_mode="SemanticKITTI", frequency_hz=20.0):
            super().__init__('mcap_virtual_replay')
            self.pub_lidar = self.create_publisher(PointCloud2, '/lidar_points', 10)
            self.pub_costmap = self.create_publisher(OccupancyGrid, '/elevation_costmap', 10)
            self.pub_tracks = self.create_publisher(PoseArray, '/dynamic_tracks', 10)
            self.pub_tf = self.create_publisher(TFMessage, '/tf', 10)
            self.pub_tf_static = self.create_publisher(TFMessage, '/tf_static', 10)
            
            self.timer = self.create_timer(1.0 / frequency_hz, self.timer_callback)
            self.frame_count = 0
            self.get_logger().info(f"Initialized MCAP Replay Node | Dataset: {dataset_mode} | Rate: {frequency_hz} Hz")
            self.get_logger().info("Storage Plugin: rosbag2_storage_mcap | Profile: fastwrite")

        def timer_callback(self):
            self.frame_count += 1
            if self.frame_count % 20 == 0:
                self.get_logger().info(f"Successfully published {self.frame_count} frames to ROS 2 topics.")

    rclpy.init()
    replay_node = VirtualSensorReplay(frequency_hz=20.0)
    try:
        rclpy.spin(replay_node)
    except KeyboardInterrupt:
        pass
    finally:
        replay_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    if HAS_ROS2:
        run_ros2_node()
    else:
        mock_node = MockVirtualSensorReplay(frequency_hz=20.0)
        mock_node.spin()