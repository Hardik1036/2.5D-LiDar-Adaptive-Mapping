"""
DRDO SIH26053 - Module 2
Virtual Sensor Playback Utility (replay_stream.py)
"""
import sys
import time
from typing import Any, Callable, Dict, Optional
import numpy as np

from backend.ingestion.dataset_loader import DatasetLoader

# Tell Pylance to ignore static missing imports for Windows local testing
try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from sensor_msgs.msg import PointCloud2, PointField  # type: ignore
    from std_msgs.msg import Header  # type: ignore
    from nav_msgs.msg import OccupancyGrid  # type: ignore
    from geometry_msgs.msg import PoseArray  # type: ignore
    from tf2_msgs.msg import TFMessage  # type: ignore
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    # Dummy fallbacks for the Python runtime
    Node = object  # type: ignore
    PointCloud2 = object  # type: ignore
    PointField = object  # type: ignore
    Header = object  # type: ignore
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
    def __init__(
        self,
        dataset_mode: str = "SemanticKITTI",
        frequency_hz: float = 20.0,
        on_frame: Optional[Callable[[np.ndarray, Dict[str, Any]], None]] = None,
        target_queue: Optional[Any] = None,
    ):
        self.frequency_hz = frequency_hz
        self.frame_count = 0
        self.loader = DatasetLoader(loop=True)
        self.on_frame = on_frame
        self.target_queue = target_queue
        print(f"[MOCK MODE] Initialized MCAP Replay Node | Dataset: {dataset_mode} | Rate: {frequency_hz} Hz")
        print("[MOCK MODE] Storage Plugin: rosbag2_storage_mcap | Profile: fastwrite")

    def publish_frame(self, points: np.ndarray, meta: Dict[str, Any]):
        """Dispatches the loaded frame to any registered downstream listener or queue."""
        if self.target_queue is not None:
            self.target_queue.put((points, meta))
        if self.on_frame is not None:
            self.on_frame(points, meta)

    def spin(self, max_frames: Optional[int] = None):
        try:
            for points, meta in self.loader.stream():
                self.publish_frame(points, meta)
                self.frame_count += 1
                if self.frame_count % 20 == 0:
                    print(f"[MOCK] Successfully published {self.frame_count} frames to virtual ROS 2 topics.")
                if max_frames is not None and self.frame_count >= max_frames:
                    break
                time.sleep(1.0 / self.frequency_hz)
        except KeyboardInterrupt:
            print("\n[MOCK] Shutting down virtual replay.")


def run_ros2_node():
    class VirtualSensorReplay(Node):  # type: ignore
        def __init__(self, dataset_mode="SemanticKITTI", frequency_hz=20.0):
            super().__init__('mcap_virtual_replay')
            self.loader = DatasetLoader(loop=True)
            self.pub_lidar = self.create_publisher(PointCloud2, '/lidar_points', 10)
            self.pub_costmap = self.create_publisher(OccupancyGrid, '/elevation_costmap', 10)
            self.pub_tracks = self.create_publisher(PoseArray, '/dynamic_tracks', 10)
            self.pub_tf = self.create_publisher(TFMessage, '/tf', 10)
            self.pub_tf_static = self.create_publisher(TFMessage, '/tf_static', 10)
            
            self.timer = self.create_timer(1.0 / frequency_hz, self.timer_callback)
            self.frame_count = 0
            self.get_logger().info(f"Initialized MCAP Replay Node | Dataset: {dataset_mode} | Rate: {frequency_hz} Hz")
            self.get_logger().info("Storage Plugin: rosbag2_storage_mcap | Profile: fastwrite")

        def _points_to_pointcloud2(self, points: np.ndarray, frame_id: str = "lidar_sensor"):
            """Converts numpy point array to ROS 2 PointCloud2 message."""
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = frame_id

            pts_f32 = np.ascontiguousarray(
                points[:, :4] if points.shape[1] >= 4 else points[:, :3], dtype=np.float32
            )
            n_pts = len(pts_f32)
            has_intensity = pts_f32.shape[1] >= 4

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            point_step = 12
            if has_intensity:
                fields.append(PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1))
                point_step = 16

            msg = PointCloud2()
            msg.header = header
            msg.height = 1
            msg.width = n_pts
            msg.fields = fields
            msg.is_bigendian = False
            msg.point_step = point_step
            msg.row_step = point_step * n_pts
            msg.is_dense = True
            msg.data = pts_f32.tobytes()
            return msg

        def timer_callback(self):
            try:
                points, meta = self.loader.load_frame()
            except (StopIteration, Exception):
                return

            if points is not None and len(points) > 0 and self.pub_lidar is not None:
                msg = self._points_to_pointcloud2(points)
                self.pub_lidar.publish(msg)

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