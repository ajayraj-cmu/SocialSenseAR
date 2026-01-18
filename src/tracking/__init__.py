"""
Object Tracking Module.

Responsibilities:
- Persistent object ID assignment
- Temporal identity tracking
- Kalman filtering for smooth trajectories
- ID reassignment logging
"""

from .object_tracker import ObjectTracker
from .kalman_tracker import KalmanBoxTracker


