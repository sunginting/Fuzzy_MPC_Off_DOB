#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Logger Node for LOS Guidance Trajectory Tracking (ROS1 version)

Sama seperti data_logger_mavros_traj.py, tetapi:
- target_pos subscribe ke /trajectory/desired_pose (desired path asli, bukan LOS lookahead)
- Untuk menghitung error terhadap path yang sebenarnya

Fitur:
- Logging → CSV (append) dengan urutan kolom tetap
- Waktu 'time' konsisten relatif t0 (sampel pertama), semua batch
- Saat ROS shutdown: flush CSV terakhir → build Excel (Summary + sheets)
- Menggunakan topik MAVROS + trajectory:
  - /mavros/local_position/pose
  - /mavros/local_position/velocity_local
  - /trajectory/desired_pose        (🔁 DESIRED PATH - titik proyeksi pada path asli)
  - /trajectory/ref_pose            (LOS lookahead point - untuk referensi)
  - /trajectory/ref_vel             (🔁 trajectory reference velocity)
  - /control/trajectory_setpoint    (output posisi dari MPC)
  - /control/velocity_setpoint      (output kecepatan dari MPC)
  - /control/mpc_acceleration       (output acceleration dari MPC sebelum konversi ke attitude)
  - /mavros/setpoint_raw/target_attitude  (attitude target yang dikirim ke PX4)
"""

import os
import csv
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from mavros_msgs.msg import AttitudeTarget
from std_msgs.msg import String, Float64MultiArray, Float32MultiArray


class DataLoggerLOSGuidance(object):
    def __init__(self):
        # =======================
        # PARAMETER
        # =======================
        home_dir = os.path.expanduser('~')
        default_log_dir = os.path.join(home_dir, 'flight_logs')

        self.output_dir = rospy.get_param('~output_dir', default_log_dir)
        self.log_rate = float(rospy.get_param('~log_rate_hz', 20.0))          # Hz
        self.auto_save_interval = float(rospy.get_param('~auto_save_interval', 30.0))  # detik
        self.auto_stop_on_finish = rospy.get_param('~auto_stop_on_finish', True)  # Auto stop saat trajectory selesai

        self.trajectory_finished = False

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = f'flight_log_los_{timestamp}'
        self.csv_filename = os.path.join(self.output_dir, base + '.csv')
        self.excel_filename = os.path.join(self.output_dir, base + '.xlsx')

        # clock steady (pakai time.monotonic)
        self._t0 = None               # t0 monotonic
        self.last_ref_time = None     # waktu terakhir trajectory ref pose diterima

        # =======================
        # DATA STORAGE
        # =======================
        self.data_buffer = []
        self._csv_header_written = False
        self._buf_flush_every = 200

        # urutan kolom fix
        self._columns = [
            'time',
            'pos_x', 'pos_y', 'pos_z',
            'vel_x', 'vel_y', 'vel_z', 'vel_mag',
            'roll_deg', 'pitch_deg', 'yaw_deg',
            # Target = desired path (proyeksi pada path asli)
            'target_pos_x', 'target_pos_y', 'target_pos_z',
            'target_vel_x', 'target_vel_y', 'target_vel_z', 'target_vel_mag',
            'target_yaw_deg',
            # LOS reference (lookahead point) - untuk analisis tambahan
            'los_ref_x', 'los_ref_y', 'los_ref_z', 'los_ref_yaw_deg',
            # Control output dari MPC
            'control_pos_x', 'control_pos_y', 'control_pos_z',
            'control_vel_x', 'control_vel_y', 'control_vel_z', 'control_vel_mag',
            'control_yaw_deg',
            'mpc_accel_x', 'mpc_accel_y', 'mpc_accel_z', 'mpc_accel_mag',  # MPC acceleration output (NED)
            'attitude_target_roll_deg', 'attitude_target_pitch_deg', 'attitude_target_yaw_deg',
            'attitude_target_thrust',
            # Error terhadap desired path (path asli)
            'est_xdist_x', 'est_xdist_y', 'est_xdist_z',
            'est_xdist_vx', 'est_xdist_vy', 'est_xdist_vz',
            'est_ydist_x', 'est_ydist_y', 'est_ydist_z',
            'est_ydist_vx', 'est_ydist_vy', 'est_ydist_vz',
            'error_x', 'error_y', 'error_z', 'error_mag',
            'de_x', 'de_y', 'de_z',
            'error_vel_x', 'error_vel_y', 'error_vel_z', 'error_vel_mag',
            'de_vx', 'de_vy', 'de_vz',
            'error_yaw_deg',
            'error_roll_deg', 'error_pitch_deg',
            # Error terhadap LOS reference (lookahead point)
            'error_los_x', 'error_los_y', 'error_los_z', 'error_los_mag',
            'waypoint_fresh',   # sekarang artinya "ref point fresh" (trajectory point baru < 0.5 s)
            'trajectory_status',  # status: WAIT_ALT, TRACK, HOLD, FINISHED
            # FLC weights (MPC Q and R matrices)
            'flc_q_x', 'flc_q_y', 'flc_q_z', 'flc_q_vx', 'flc_q_vy', 'flc_q_vz',
            'flc_r_x', 'flc_r_y', 'flc_r_z',
            'avg_error_pose', 'avg_error_att', 'avg_error_velo', 'avg_error_vz'
        ]

        # state sekarang (ENU dari MAVROS)
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_euler = np.zeros(3)  # roll, pitch, yaw (rad)

        # desired path (target) - titik proyeksi pada path asli
        self.target_pos = np.zeros(3)
        self.target_vel = np.zeros(3)
        self.target_yaw = 0.0

        # LOS reference (lookahead point)
        self.los_ref_pos = np.zeros(3)
        self.los_ref_yaw = 0.0

        # control output (dari controller/MPC)
        self.control_pos = np.zeros(3)
        self.control_vel = np.zeros(3)
        self.control_yaw = 0.0

        # MPC acceleration output (NED frame)
        self.mpc_acceleration = np.zeros(3)

        # attitude target (dari controller via MAVROS)
        self.attitude_target_euler = np.zeros(3)
        self.attitude_target_thrust = 0.0

        # flags
        self.position_received = False
        self.velocity_received = False
        self.desired_ref_received = False
        self.los_ref_received = False
        self.trajectory_output_received = False

        # trajectory status (WAIT_ALT, TRACK, HOLD, FINISHED)
        self.current_trajectory_status = "UNKNOWN"

        # FLC weights (MPC Q and R matrices)
        self.flc_weights = np.zeros(9)  # [q_x, q_y, q_z, q_vx, q_vy, q_vz, r_x, r_y, r_z]

        # AVG ERR
        self.previous_time = 0.0
        self.current_time = 0.0

        self.d_error_x = 0.0
        self.d_error_y = 0.0
        self.d_error_z = 0.0

        self.d_error_vx = 0.0
        self.d_error_vy = 0.0
        self.d_error_vz = 0.0

        self.dist = np.zeros(12)

        self.avg_error = np.zeros(4)
        self.previous_pos_error = np.zeros(3)
        self.previous_vel_error = np.zeros(3)

        # info controller (untuk sheet Summary)
        self.controller_type = 'MPC_LOS_GUIDANCE'

        # graceful shutdown
        self._shutting_down = False
        self._io_lock = threading.RLock()

        # =======================
        # SUBSCRIBERS
        # =======================
        queue_size = 10

        # Posisi & kecepatan aktual dari MAVROS (ENU)
        self.position_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.position_callback, queue_size=queue_size)
        self.velocity_sub = rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.velocity_callback, queue_size=queue_size)

        # DESIRED PATH - titik proyeksi pada path asli (untuk hitung error)
        self.desired_pose_sub = rospy.Subscriber('/trajectory/desired_pose', PoseStamped, self.desired_pose_callback, queue_size=queue_size)

        # LOS reference (lookahead point) - untuk referensi/analisis
        self.los_ref_pose_sub = rospy.Subscriber('/trajectory/ref_pose', PoseStamped, self.los_ref_pose_callback, queue_size=queue_size)

        # Trajectory reference velocity
        self.traj_vel_sub = rospy.Subscriber('/trajectory/ref_vel',TwistStamped,self.trajectory_vel_callback,queue_size=queue_size)

        # Output MPC (posisi & kecepatan setpoint yang dikirim ke PX4)
        self.trajectory_sub = rospy.Subscriber('/control/trajectory_setpoint',PoseStamped,self.trajectory_output_callback,queue_size=queue_size)

        self.control_vel_sub = rospy.Subscriber('/control/velocity_setpoint',TwistStamped,self.control_velocity_callback,queue_size=queue_size)

        # MPC acceleration output (NED frame, sebelum dikonversi ke attitude)
        self.mpc_accel_sub = rospy.Subscriber('/control/mpc_acceleration',Vector3Stamped,self.mpc_acceleration_callback,queue_size=queue_size)

        # Attitude target yang dikirim ke PX4
        self.attitude_target_sub = rospy.Subscriber('/mavros/setpoint_raw/target_attitude',AttitudeTarget,self.attitude_target_callback,queue_size=queue_size)

        # Subscribe ke trajectory status untuk auto-stop
        self.trajectory_status_sub = rospy.Subscriber('/trajectory/status',String,self.trajectory_status_callback,queue_size=10)

        # FLC MPC weights (Q and R matrices from fuzzy logic controller)
        self.weights_MPC = rospy.Subscriber('/mpc/weights', Float64MultiArray, self.flc_weights_callback, queue_size=10)

        # AVG ERROR
        self.error_avg_sub = rospy.Subscriber('/flc/error_avg', Float64MultiArray, self.avg_error_callback, queue_size=10)

        # Estimate callbacl
        self.est_sub = rospy.Subscriber('/mpc/dist_estimate', Float32MultiArray, self.est_callback,queue_size=10)

        # =======================
        # TIMERS
        # =======================
        self.log_timer = rospy.Timer(
            rospy.Duration(1.0 / self.log_rate),
            self.log_data
        )
        self.save_timer = rospy.Timer(
            rospy.Duration(self.auto_save_interval),
            self.auto_flush_csv
        )

        # info awal
        rospy.loginfo('=' * 70)
        rospy.loginfo('📊 DATA LOGGER LOS GUIDANCE (ROS1) STARTED')
        rospy.loginfo('=' * 70)
        rospy.loginfo('Output dir : %s', self.output_dir)
        rospy.loginfo('CSV file   : %s', self.csv_filename)
        rospy.loginfo('Excel file : %s (dibuat saat shutdown)', self.excel_filename)
        rospy.loginfo('Log rate   : %.1f Hz', self.log_rate)
        rospy.loginfo('Flush intv : %.1f s', self.auto_save_interval)
        rospy.loginfo('=' * 70)
        rospy.loginfo('Target reference: /trajectory/desired_pose (desired path asli)')
        rospy.loginfo('LOS reference   : /trajectory/ref_pose (lookahead point)')
        rospy.loginfo('=' * 70)
        rospy.loginfo('Waiting for MAVROS & trajectory data...')

        # ROS shutdown hook + atexit
        rospy.on_shutdown(self.on_shutdown)

    # =======================
    # CALLBACKS
    # =======================
    def position_callback(self, msg: PoseStamped):
        """MAVROS local_position/pose (ENU)"""
        if self._shutting_down:
            return
        
        self.current_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        self.current_euler = self.quaternion_to_euler(q.w, q.x, q.y, q.z)

        if not self.position_received:
            self.position_received = True
            rospy.loginfo('✅ Position data received (MAVROS /mavros/local_position/pose)')

    def velocity_callback(self, msg: TwistStamped):
        """MAVROS local_position/velocity_local (ENU)"""
        if self._shutting_down:
            return

        self.current_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

        if not self.velocity_received:
            self.velocity_received = True
            rospy.loginfo('✅ Velocity data received (MAVROS /mavros/local_position/velocity_local)')

    def desired_pose_callback(self, msg: PoseStamped):
        """
        DESIRED PATH - titik proyeksi pada path asli (ENU).
        Ini adalah lintasan referensi yang sebenarnya untuk menghitung trajectory error.
        """
        if self._shutting_down:
            return

        self.target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        _, _, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.target_yaw = yaw

        # untuk flag "fresh point"
        self.last_ref_time = time.monotonic()

        if not self.desired_ref_received:
            self.desired_ref_received = True
            rospy.loginfo('✅ Desired path reference received (/trajectory/desired_pose)')

    def los_ref_pose_callback(self, msg: PoseStamped):
        """
        LOS reference (lookahead point) - untuk referensi/analisis.
        Ini adalah titik yang sebenarnya di-track oleh controller.
        """
        if self._shutting_down:
            return

        self.los_ref_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        _, _, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.los_ref_yaw = yaw

        if not self.los_ref_received:
            self.los_ref_received = True
            rospy.loginfo('✅ LOS reference pose received (/trajectory/ref_pose)')

    def trajectory_vel_callback(self, msg: TwistStamped):
        """Trajectory reference velocity (ENU) → disimpan sebagai target_vel."""
        if self._shutting_down:
            return

        self.target_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

    def trajectory_output_callback(self, msg: PoseStamped):
        """
        Control trajectory setpoint (posisi) yang dikirim oleh controller (MPC) ke PX4.
        Disimpan sebagai control_pos + control_yaw, hanya untuk logging/analisis.
        """
        if self._shutting_down:
            return

        self.control_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

        q = msg.pose.orientation
        _, _, yaw = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.control_yaw = yaw

        if not self.trajectory_output_received:
            self.trajectory_output_received = True
            rospy.loginfo('✅ Trajectory setpoint output received (/control/trajectory_setpoint)')

    def control_velocity_callback(self, msg: TwistStamped):
        """Control velocity setpoint dari controller."""
        if self._shutting_down:
            return

        self.control_vel = np.array([
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ])

    def attitude_target_callback(self, msg: AttitudeTarget):
        """Attitude target dari controller (via MAVROS)."""
        if self._shutting_down:
            return

        q = msg.orientation
        self.attitude_target_euler = self.quaternion_to_euler(q.w, q.x, q.y, q.z)
        self.attitude_target_thrust = msg.thrust

    def mpc_acceleration_callback(self, msg: Vector3Stamped):
        """MPC acceleration output (NED frame) sebelum dikonversi ke attitude."""
        if self._shutting_down:
            return
        
        self.mpc_acceleration = np.array([
            msg.vector.x,  # ax (North)
            msg.vector.y,  # ay (East)
            msg.vector.z   # az (Down)
        ])

    def trajectory_status_callback(self, msg: String):
        """Callback untuk status trajectory - simpan status dan auto stop saat selesai"""
        # Simpan status saat ini untuk logging
        status_text = msg.data.upper()
        
        # Parse status dari pesan (bisa berupa "WAIT_ALT", "TRACK", "HOLD", "FINISHED", dll)
        if "FINISHED" in status_text:
            self.current_trajectory_status = "FINISHED"
        elif "HOLD" in status_text:
            self.current_trajectory_status = "HOLD"
        elif "TRACK" in status_text:
            self.current_trajectory_status = "TRACK"
        elif "WAIT" in status_text or "ALT" in status_text:
            self.current_trajectory_status = "WAIT_ALT"
        else:
            self.current_trajectory_status = "UNKNOWN"  # Truncate jika terlalu panjang
        
        if msg.data == "FINISHED" and not self.trajectory_finished:
            self.trajectory_finished = True
            rospy.loginfo("\n" + "="*60)
            rospy.loginfo("🏁 Trajectory FINISHED detected!")
            rospy.loginfo("="*60)
            
            if self.auto_stop_on_finish:
                rospy.loginfo("Auto-stopping data logger in 2 seconds...")
                rospy.loginfo("Shutting down data logger node.")
                rospy.signal_shutdown("Trajectory finished - auto stop")

    def flc_weights_callback(self, msg: Float64MultiArray):
        """
        Callback untuk menerima weights MPC dari FLC (Fuzzy Logic Controller).
        Data berisi Q dan R matrices yang di-flatten.
        """
        if self._shutting_down:
            return
        
        # msg.data = [q_x, q_y, q_z, q_vx, q_vy, q_vz, r_x, r_y, r_z]
        if len(msg.data) >= 9:
            self.flc_weights = np.array(msg.data[:9])

    def avg_error_callback(self, msg:Float64MultiArray):
        if self._shutting_down:
            return
        
        if len(msg.data) >= 4:
            self.avg_error = np.array(msg.data[:4])

    def est_callback(self, msg:Float32MultiArray):
        if self._shutting_down:
            return
        
        if len(msg.data) >= 12:
            self.dist = np.array(msg.data[:12])

    # =======================
    # UTIL
    # =======================
    def quaternion_to_euler(self, w, x, y, z):
        """Konversi quaternion → euler (roll, pitch, yaw) rad."""
        # roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # pitch
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)

        # yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    # =======================
    # LOGGING
    # =======================
    def log_data(self, event):
        if self._shutting_down:
            return

        if not self.position_received or not self.velocity_received:
            return

        now = time.monotonic()
        timestamp_sec = now

        if self._t0 is None:
            self._t0 = timestamp_sec

        # flag waypoint_fresh (di sini artinya "trajectory ref fresh")
        waypoint_fresh = 0
        if self.last_ref_time is not None:
            dt_wp = now - self.last_ref_time
            waypoint_fresh = 1 if dt_wp < 0.5 else 0

        # trajectory tracking error (terhadap desired path asli)
        pos_error = self.current_pos - self.target_pos
        vel_error = self.current_vel - self.target_vel

        self.current_time = rospy.Time.now().to_sec()
        if self.previous_time == 0.0:
            self.previous_time = self.current_time
            return
        
        dt = self.current_time - self.previous_time
        if dt <= 0:
            return
        if dt > 0:
            self.d_error_x = (pos_error[0] - self.previous_pos_error[0])/dt
            self.d_error_y = (pos_error[1] - self.previous_pos_error[1])/dt
            self.d_error_z = (pos_error[2] - self.previous_pos_error[2])/dt

            self.d_error_vx = (vel_error[0] - self.previous_vel_error[0])/dt
            self.d_error_vy = (vel_error[1] - self.previous_vel_error[1])/dt
            self.d_error_vz = (vel_error[2] - self.previous_vel_error[2])/dt

        self.previous_pos_error = pos_error.copy()
        self.previous_vel_error = vel_error.copy()
        self.previous_time = self.current_time

        yaw_error = self.current_euler[2] - self.target_yaw
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # error terhadap LOS reference (lookahead point)
        los_error = self.current_pos - self.los_ref_pos

        # attitude error
        roll_error = self.current_euler[0] - self.attitude_target_euler[0]
        pitch_error = self.current_euler[1] - self.attitude_target_euler[1]
        roll_error = np.arctan2(np.sin(roll_error), np.cos(roll_error))
        pitch_error = np.arctan2(np.sin(pitch_error), np.cos(pitch_error))

        entry = {
            'time': timestamp_sec,

            'pos_x': float(self.current_pos[0]),
            'pos_y': float(self.current_pos[1]),
            'pos_z': float(self.current_pos[2]),

            'vel_x': self.current_vel[0],
            'vel_y': self.current_vel[1],
            'vel_z': self.current_vel[2],
            'vel_mag': float(np.linalg.norm(self.current_vel)),

            'roll_deg': float(np.degrees(self.current_euler[0])),
            'pitch_deg': float(np.degrees(self.current_euler[1])),
            'yaw_deg': float(np.degrees(self.current_euler[2])),

            # Target = desired path (proyeksi pada path asli)
            'target_pos_x': float(self.target_pos[0]),
            'target_pos_y': float(self.target_pos[1]),
            'target_pos_z': float(self.target_pos[2]),

            'target_vel_x': self.target_vel[0],
            'target_vel_y': self.target_vel[1],
            'target_vel_z': self.target_vel[2],
            'target_vel_mag': float(np.linalg.norm(self.target_vel)),

            'target_yaw_deg': float(np.degrees(self.target_yaw)),

            # LOS reference (lookahead point)
            'los_ref_x': self.los_ref_pos[0],
            'los_ref_y': self.los_ref_pos[1],
            'los_ref_z': self.los_ref_pos[2],
            'los_ref_yaw_deg': float(np.degrees(self.los_ref_yaw)),

            # Control output dari MPC
            'control_pos_x': self.control_pos[0],
            'control_pos_y': self.control_pos[1],
            'control_pos_z': self.control_pos[2],

            'control_vel_x': self.control_vel[0],
            'control_vel_y': self.control_vel[1],
            'control_vel_z': self.control_vel[2],
            'control_vel_mag': float(np.linalg.norm(self.control_vel)),

            'control_yaw_deg': float(np.degrees(self.control_yaw)),

            # MPC acceleration output (NED frame)
            'mpc_accel_x': self.mpc_acceleration[0],
            'mpc_accel_y': self.mpc_acceleration[1],
            'mpc_accel_z': self.mpc_acceleration[2],
            'mpc_accel_mag': float(np.linalg.norm(self.mpc_acceleration)),

            'attitude_target_roll_deg': float(np.degrees(self.attitude_target_euler[0])),
            'attitude_target_pitch_deg': float(np.degrees(self.attitude_target_euler[1])),
            'attitude_target_yaw_deg': float(np.degrees(self.attitude_target_euler[2])),
            'attitude_target_thrust': float(self.attitude_target_thrust),

            'est_xdist_x': float(self.dist[0]),
            'est_xdist_y': float(self.dist[1]),
            'est_xdist_z': float(self.dist[2]),
            'est_xdist_vx': float(self.dist[3]),
            'est_xdist_vy': float(self.dist[4]),
            'est_xdist_vz': float(self.dist[5]),
            'est_ydist_x': float(self.dist[6]),
            'est_ydist_y': float(self.dist[7]),
            'est_ydist_z': float(self.dist[8]),
            'est_ydist_vx': float(self.dist[9]),
            'est_ydist_vy': float(self.dist[10]),
            'est_ydist_vz': float(self.dist[11]),

            # Error terhadap desired path (path asli)
            'error_x': pos_error[0],
            'error_y': pos_error[1],
            'error_z': pos_error[2],
            'error_mag': float(np.linalg.norm(pos_error)),

            'de_x': float(self.d_error_x),
            'de_y': float(self.d_error_y),
            'de_z': float(self.d_error_z),

            'error_vel_x': vel_error[0],
            'error_vel_y': vel_error[1],
            'error_vel_z': vel_error[2],
            'error_vel_mag': float(np.linalg.norm(vel_error)),

            'de_vx': float(self.d_error_vx),
            'de_vy': float(self.d_error_vy),
            'de_vz': float(self.d_error_vz),

            'error_yaw_deg': float(np.degrees(yaw_error)),
            'error_roll_deg': float(np.degrees(roll_error)),
            'error_pitch_deg': float(np.degrees(pitch_error)),

            # Error terhadap LOS reference (lookahead point)
            'error_los_x': los_error[0],
            'error_los_y': los_error[1],
            'error_los_z': los_error[2],
            'error_los_mag': float(np.linalg.norm(los_error)),

            'waypoint_fresh': int(waypoint_fresh),
            'trajectory_status': self.current_trajectory_status,

            # FLC weights (MPC Q and R matrices)
            'flc_q_x': float(self.flc_weights[0]),
            'flc_q_y': float(self.flc_weights[1]),
            'flc_q_z': float(self.flc_weights[2]),
            'flc_q_vx': float(self.flc_weights[3]),
            'flc_q_vy': float(self.flc_weights[4]),
            'flc_q_vz': float(self.flc_weights[5]),
            'flc_r_x': float(self.flc_weights[6]),
            'flc_r_y': float(self.flc_weights[7]),
            'flc_r_z': float(self.flc_weights[8]),

            'avg_error_pose': float(self.avg_error[0]),
            'avg_error_att': float(self.avg_error[1]),
            'avg_error_velo': float(self.avg_error[2]),
            'avg_error_vz': float(self.avg_error[3]),
        }

        self.data_buffer.append(entry)

        if len(self.data_buffer) >= self._buf_flush_every:
            self.flush_to_csv()

    def auto_flush_csv(self, event):
        if self._shutting_down:
            return
        self.flush_to_csv()

    def flush_to_csv(self):
        """Tulis buffer → CSV (append) dengan urutan kolom fix, time relatif t0."""
        if not self.data_buffer:
            return

        with self._io_lock:
            if not self.data_buffer:
                return

            try:
                df = pd.DataFrame(self.data_buffer, columns=self._columns)

                t0 = self._t0 if self._t0 is not None else df['time'].iloc[0]
                df['time'] = df['time'] - t0

                mode = 'a' if self._csv_header_written else 'w'
                header = not self._csv_header_written

                df.to_csv(
                    self.csv_filename,
                    mode=mode,
                    header=header,
                    index=False,
                    quoting=csv.QUOTE_NONNUMERIC
                )

                self._csv_header_written = True
                self.data_buffer.clear()
                rospy.loginfo('💾 Flushed → %s', Path(self.csv_filename).name)

            except Exception as e:
                rospy.logerr('❌ CSV flush failed: %s', str(e))

    # =======================
    # SHUTDOWN
    # =======================
    def on_shutdown(self):
        if self._shutting_down:
            return
        
        self._shutting_down = True
        rospy.loginfo("Stopping timers...")
        try:
            self.log_timer.shutdown()
        except:
            pass

        try:
            self.save_timer.shutdown()
        except:
            pass

        time.sleep(1.0)
        rospy.loginfo("Saving final CSV...")
        try:
            self.flush_to_csv()
        except Exception as e:
            rospy.logerr(str(e))

        rospy.loginfo("CSV saved successfully.")


def build_excel_from_csv(csv_path, excel_path):

    import pandas as pd
    import numpy as np

    print("Loading CSV...")
    df = pd.read_csv(csv_path)

    print("Building Excel...")

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        df.to_excel(writer, sheet_name='Flight Data', index=False)

        summary = pd.DataFrame({
            'Metric': ['Samples'],
            'Value': [len(df)]
        })

        summary.to_excel(writer, sheet_name='Summary', index=False)

    print("Excel complete.")

def main():
    rospy.init_node('data_logger_los_guidance', anonymous=False)
    node = DataLoggerLOSGuidance()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('🔻 KeyboardInterrupt received.')
    # on_shutdown akan otomatis dipanggil oleh ROS


if __name__ == '__main__':
    main()