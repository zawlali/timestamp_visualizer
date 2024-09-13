import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import PointCloud2, Imu  # Import Imu message type
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import defaultdict, deque
import threading
from functools import partial
import matplotlib.dates as mdates
import datetime
import numpy as np  # Import NumPy

from matplotlib.ticker import MultipleLocator

class TimestampVisualizer(Node):
    def __init__(self):
        super().__init__('timestamp_visualizer')
        
        # Explicitly define attributes
        self.topics = None
        self.window_size = None
        self.callback_group = None
        self.subscriptions_set = None
        self.timestamps = None
        self.publish_rates = None
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.scatter_plots = None
        self.rate_bars = None
        self.rate_labels = None  # For refresh rate labels
        self.start_time = None
        self.ani = None
        self.lock = threading.Lock()
        self.topic_positions = None

        self.declare_parameter('topics', ['topic1', 'topic2', 'topic3'])
        self.declare_parameter('window_size', 10.0)  # seconds
        
        self.topics = self.get_parameter('topics').value
        print(f"Subscribing to topics: {self.topics}")
        self.window_size = self.get_parameter('window_size').value
        
        self.callback_group = ReentrantCallbackGroup()

        self.timestamps = defaultdict(lambda: deque(maxlen=1000))
        self.publish_rates = defaultdict(float)
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscriptions_set = []
        
        # Map topics to message types
        self.topic_types = {}
        for topic in self.topics:
            # Determine the message type for each topic
            if topic == '/sensing/lidar/top/points_raw':
                msg_type = PointCloud2
            elif topic == '/imu':
                msg_type = Imu
            else:
                # Default to PointCloud2 or handle unknown topics
                msg_type = PointCloud2  # Replace with appropriate default or raise exception
            self.topic_types[topic] = msg_type

            try:
                sub = self.create_subscription(
                    msg_type,
                    topic,
                    partial(self.topic_callback, topic=topic),
                    qos_profile,
                    callback_group=self.callback_group
                )
                self.subscriptions_set.append(sub)
                self.get_logger().info(f"Successfully subscribed to topic: {topic}")
            except Exception as e:
                self.get_logger().error(f"Failed to create subscription for topic '{topic}': {str(e)}")

        # Set up the plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.scatter_plots = {}
        self.rate_bars = {}
        self.rate_labels = {}

        # Map topics to y-axis positions
        self.topic_positions = {topic: i for i, topic in enumerate(self.topics)}

        # Create scatter plots for each topic
        for topic in self.topics:
            sc = self.ax1.scatter([], [], s=100, marker='o', label=topic)
            self.scatter_plots[topic] = sc
            bar_container = self.ax2.bar(topic, 0)
            self.rate_bars[topic] = bar_container[0]  # Store Rectangle object
            # Add a text label under each bar using annotate
            label = self.ax2.annotate(
                '',
                xy=(bar_container[0].get_x() + bar_container[0].get_width() / 2, 0),
                xytext=(0, -15),  # Offset the text by -15 points in the y-direction
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=10,
                color='blue',
            )
            self.rate_labels[topic] = label

        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Topics')
        self.ax1.set_title('Topic Timestamps')
        self.ax1.set_ylim(-1, len(self.topics))
        self.ax1.set_yticks(list(self.topic_positions.values()))
        self.ax1.set_yticklabels(self.topics)
        self.ax1.legend()

        # Initial x-axis formatter
        self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        self.ax2.set_xlabel('Topics')
        self.ax2.set_ylabel('Publish Rate (Hz)')
        self.ax2.set_title('Topic Publish Rates')

        self.start_time = self.get_clock().now().nanoseconds * 1e-9

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def topic_callback(self, msg, *, topic):
        print(f"Received message on topic '{topic}'")
        current_time = self.get_clock().now().nanoseconds * 1e-9  # Current ROS time in seconds

        with self.lock:
            self.timestamps[topic].append(current_time)
            if len(self.timestamps[topic]) > 1:
                time_diff = self.timestamps[topic][-1] - self.timestamps[topic][-2]
                if time_diff > 0:
                    self.publish_rates[topic] = 1.0 / time_diff

    def update_plot(self, frame):
        print("Updating plot...")
        current_time = self.get_clock().now().nanoseconds * 1e-9

        with self.lock:
            # Adjust window size based on max publish rate
            if self.publish_rates:
                max_publish_rate = max(self.publish_rates.values())
                new_window_size = max(1.0, min(10.0, 50.0 / max_publish_rate))
                self.window_size = new_window_size
            else:
                self.window_size = 10.0  # Default window size when no data

            for topic, sc in self.scatter_plots.items():
                timestamps = [t for t in self.timestamps[topic] if t >= current_time - self.window_size]
                if timestamps:
                    x_data = [datetime.datetime.fromtimestamp(t) for t in timestamps]
                    y_data = [self.topic_positions[topic]] * len(x_data)
                    offsets = np.column_stack((mdates.date2num(x_data), y_data))
                    sc.set_offsets(offsets)
                else:
                    sc.set_offsets(np.empty((0, 2)))  # Pass empty array with shape (0, 2)

            for topic in self.topics:
                bar = self.rate_bars[topic]
                bar.set_height(self.publish_rates[topic])
                label = self.rate_labels[topic]
                label.set_text(f"{self.publish_rates[topic]:.2f} Hz")
                # Update label position if needed
                label.xy = (bar.get_x() + bar.get_width() / 2, 0)

        # Update x-axis limits
        start_time = datetime.datetime.fromtimestamp(current_time - self.window_size)
        end_time = datetime.datetime.fromtimestamp(current_time)
        self.ax1.set_xlim(start_time, end_time)

        # Adjust x-axis ticks and grid lines
        self.set_xaxis_ticks()

        self.ax2.relim()
        self.ax2.autoscale_view()

        return list(self.scatter_plots.values()) + [self.rate_bars[topic] for topic in self.topics]

    def set_xaxis_ticks(self):
        # Adjust the x-axis ticks based on window size
        if self.window_size > 5:
            # Major ticks every second
            self.ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
            self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.ax1.xaxis.set_minor_locator(mdates.SecondLocator(interval=1))
        else:
            # Major ticks at tenths of seconds
            self.ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S.%f'))
            # Minor ticks every 0.1 seconds
            self.ax1.xaxis.set_minor_locator(mdates.AutoDateLocator())

        # Enable grid lines
        self.ax1.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

def main(args=None):
    rclpy.init(args=args)

    timestamp_visualizer = TimestampVisualizer()

    executor = MultiThreadedExecutor()
    executor.add_node(timestamp_visualizer)

    # Start the executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    try:
        # Start the Matplotlib event loop on the main thread
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the animation
        timestamp_visualizer.ani.event_source.stop()
        # Shutdown the node and executor
        timestamp_visualizer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()
        executor_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
