"""
Kalman Filter for Bounding Box Tracking.

Provides smooth trajectory estimation for tracked objects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KalmanBoxTracker:
    """
    Kalman filter for tracking bounding boxes.
    
    State vector: [x_center, y_center, area, aspect_ratio, vx, vy, va]
    Measurement: [x_min, y_min, x_max, y_max]
    """
    
    def __init__(
        self,
        bbox: NDArray[np.float64],
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ):
        """
        Initialize Kalman tracker with initial bounding box.
        
        Args:
            bbox: [x_min, y_min, x_max, y_max]
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        # State dimension: 7 (x, y, area, aspect_ratio, vx, vy, va)
        # Measurement dimension: 4 (x_min, y_min, x_max, y_max)
        
        self.dim_x = 7
        self.dim_z = 4
        
        # State transition matrix
        self.F = np.eye(self.dim_x)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # area += va
        
        # Measurement matrix (converts state to measurement)
        self.H = np.zeros((self.dim_z, self.dim_x))
        
        # Process noise covariance
        self.Q = np.eye(self.dim_x) * process_noise
        self.Q[4:, 4:] *= 10  # Higher noise for velocity components
        
        # Measurement noise covariance
        self.R = np.eye(self.dim_z) * measurement_noise
        
        # Initial state covariance
        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000  # High uncertainty for initial velocities
        
        # Initialize state from first bbox
        self.x = np.zeros(self.dim_x)
        self._bbox_to_state(bbox)
        
        # Track history
        self.history = []
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
    
    def _bbox_to_state(self, bbox: NDArray[np.float64]):
        """Convert bounding box to state vector."""
        x_min, y_min, x_max, y_max = bbox
        
        w = x_max - x_min
        h = y_max - y_min
        
        self.x[0] = (x_min + x_max) / 2  # x_center
        self.x[1] = (y_min + y_max) / 2  # y_center
        self.x[2] = w * h                 # area
        self.x[3] = w / (h + 1e-6)        # aspect ratio
        # velocities stay at 0 initially
    
    def _state_to_bbox(self) -> NDArray[np.float64]:
        """Convert state vector to bounding box."""
        x_center = self.x[0]
        y_center = self.x[1]
        area = max(self.x[2], 1)  # Prevent negative area
        aspect_ratio = max(self.x[3], 0.1)  # Prevent invalid aspect ratio
        
        w = np.sqrt(area * aspect_ratio)
        h = area / (w + 1e-6)
        
        return np.array([
            x_center - w / 2,
            y_center - h / 2,
            x_center + w / 2,
            y_center + h / 2,
        ])
    
    def predict(self) -> NDArray[np.float64]:
        """
        Predict next state.
        
        Returns:
            Predicted bounding box
        """
        # Prevent negative area
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        
        # State prediction
        self.x = self.F @ self.x
        
        # Covariance prediction
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.age += 1
        self.time_since_update += 1
        
        # Store prediction
        bbox = self._state_to_bbox()
        self.history.append(bbox)
        
        return bbox
    
    def update(self, bbox: NDArray[np.float64]):
        """
        Update state with measurement.
        
        Args:
            bbox: Measured bounding box [x_min, y_min, x_max, y_max]
        """
        self.time_since_update = 0
        self.hits += 1
        
        # Convert bbox to measurement
        z = bbox
        
        # Convert current state to measurement space for comparison
        z_pred = self._state_to_bbox()
        
        # Update H matrix based on current state
        # This is a linearization around current state
        self._update_measurement_matrix()
        
        # Innovation (measurement residual)
        y = z - z_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update
        I = np.eye(self.dim_x)
        self.P = (I - K @ self.H) @ self.P
    
    def _update_measurement_matrix(self):
        """Update measurement Jacobian based on current state."""
        # Linearization of state_to_bbox around current state
        x_c, y_c, area, ar = self.x[:4]
        
        area = max(area, 1)
        ar = max(ar, 0.1)
        
        w = np.sqrt(area * ar)
        h = area / (w + 1e-6)
        
        # Partial derivatives
        dw_darea = 0.5 * np.sqrt(ar / (area + 1e-6))
        dw_dar = 0.5 * np.sqrt(area / (ar + 1e-6))
        dh_darea = 1 / (w + 1e-6) - area * dw_darea / (w**2 + 1e-6)
        dh_dar = -area * dw_dar / (w**2 + 1e-6)
        
        # Jacobian: d[x_min, y_min, x_max, y_max] / d[x_c, y_c, area, ar, ...]
        self.H = np.zeros((self.dim_z, self.dim_x))
        
        # d(x_min)/d(x_c) = 1, d(x_min)/d(w) = -0.5
        self.H[0, 0] = 1
        self.H[0, 2] = -0.5 * dw_darea
        self.H[0, 3] = -0.5 * dw_dar
        
        # d(y_min)/d(y_c) = 1, d(y_min)/d(h) = -0.5
        self.H[1, 1] = 1
        self.H[1, 2] = -0.5 * dh_darea
        self.H[1, 3] = -0.5 * dh_dar
        
        # d(x_max)/d(x_c) = 1, d(x_max)/d(w) = 0.5
        self.H[2, 0] = 1
        self.H[2, 2] = 0.5 * dw_darea
        self.H[2, 3] = 0.5 * dw_dar
        
        # d(y_max)/d(y_c) = 1, d(y_max)/d(h) = 0.5
        self.H[3, 1] = 1
        self.H[3, 2] = 0.5 * dh_darea
        self.H[3, 3] = 0.5 * dh_dar
    
    def get_state(self) -> NDArray[np.float64]:
        """Get current state as bounding box."""
        return self._state_to_bbox()
    
    def get_velocity(self) -> tuple[float, float]:
        """Get current velocity estimate (vx, vy)."""
        return float(self.x[4]), float(self.x[5])


