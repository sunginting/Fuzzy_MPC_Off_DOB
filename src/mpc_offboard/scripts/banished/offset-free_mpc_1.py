#!/usr/bin/env python3
import rospy
import numpy as np
import quadprog

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from mavros_msgs.msg import State, AttitudeTarget
from std_msgs.msg import Float64MultiArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PositionMPC:
    def __init__(self, dt = 0.1, y_hor = 4, c_hor = 3):
        self.nx = 6
        self.nu = 3
        self.y_hor = y_hor
        self.c_hor = c_hor

        self.A_model = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        self.B_model = np.array([
            [0.5*(dt**2), 0, 0],
            [0, 0.5*(dt**2), 0],
            [0, 0, 0.5*(dt**2)],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]
        ])

        self.C_model = np.eye(self.nx)

        self.B_dist = np.eye(self.nx)
        self.C_dist = np.eye(self.nx)

        self._check_dimension(self.A_model, self.B_model, self.C_model)

        self.d_model = np.zeros((6,1))
        self.x_model = np.zeros((self.nx, 1))
        self.y_model = np.zeros((self.nx, 1))
        self.u_model = np.zeros((self.nu, 1))

        # Cost matrices
        self.Q= np.diag([
            40.0, 40.0, 100.0,   # posisi
            20.0, 20.0, 12.0     # kecepatan
        ])

        self.R = np.diag([1.0, 1.0, 0.10])     # penalti besarnya u
        self.R_delta = np.diag([0.2, 0.2, 0.06])  # penalti perubahan u

        self.u_prev = np.zeros(self.nu)
        self.x_prev = np.zeros(self.nx)
        self.a_max = 5.0  # m/s^2
        self.del_u_max = 0.5  # m/s^2 per control step

        self._augment_matrix()

        self.weight_subs = rospy.Subscriber('/flc/mpc_weights', Float64MultiArray, self.weights_callback, queue_size=10)
        self.weight_pub = rospy.Publisher('/mpc/weights', Float64MultiArray, queue_size=10)

    def _check_dimension(self, A_model, B_model, C_model):
        # A matrix check (n,n)        
        if A_model.shape[0] != A_model.shape[1]:
            raise ValueError("A matrix must be square")
        if A_model.shape[0] != self.nx:
            raise ValueError(f"A rows must equal number of states ({self.nx})")

        # B matrix check (n,m)
        if B_model.shape[0] != self.nx:
            raise ValueError(f"B rows must equal number of states ({self.nx})")
        if B_model.shape[1] != self.nu:
            raise ValueError(f"B collums must equal number of inputs ({self.nu})")

        # C matrix check (n,n)
        if C_model.shape[0] != C_model.shape[1]:
            raise ValueError("C matrix must be square for full state measurement")
        if C_model.shape[0] != self.nx:
            raise ValueError(f"C rows must equal number of states ({self.nx})")

    def _augment_matrix(self):
        n = self.A_model.shape[0]
        m = self.B_model.shape[1]
        nd = self.B_dist.shape[1]    

        A_aug = np.block([
            [self.A_model,      self.B_dist],
            [np.zeros((nd, n)),  np.eye(nd)]
        ])

        B_aug = np.vstack((
            self.B_model,
            np.zeros((nd, m))
        ))

        C_aug = np.hstack((
            self.C_model, self.C_dist))

        self.A_aug = A_aug
        self.B_aug = B_aug
        self.C_aug = C_aug

        self._prediction_matrix()
    
    def _prediction_matrix(self):
        q = self.C_aug.shape[0]
        m = self.B_aug.shape[1]
        
        P_matrix = []
        for i in range (1,self.y_hor+1):
            P_matrix.append(self.C_aug @ np.linalg.matrix_power(self.A_aug,i))
        P_matrix = np.vstack(P_matrix)

        H_matrix = np.zeros((self.y_hor*q, self.c_hor*m))
        for i in range(self.y_hor):
            for j in range(self.c_hor):
                if i >= j:
                    H_matrix[i*q:(i+1)*q,j*m:(j+1)*m] = self.C_aug @ np.linalg.matrix_power(self.A_aug,i-j) @ self.B_aug
        
        self.P_matrix = P_matrix # State Prediction
        self.H_matrix = H_matrix # Control Prediction

        self._qp_matrices()

    def _qp_matrices(self):
        Q_bar = np.kron(np.eye(self.y_hor), self.Q)
        R_bar = np.kron(np.eye(self.c_hor), self.R)  
        R_delta_bar = np.kron(np.eye(self.c_hor), self.R_delta)

        H = self.H_matrix.T @ Q_bar @ self.H_matrix + R_bar + R_delta_bar
        self.H = (H + H.T) / 2.0

        self.Q_bar = Q_bar
        self.R_delta_bar = R_delta_bar

    def compute_control(self, x_meas, xref):
        ref = np.tile(xref, self.y_hor)
        pred_err = 
        
        try:
            del_u_opt = quadprog.solve_qp(self.H, -f, C.T, b, meq=0)[0]
            del_u = del_u_opt[:self.nu]

            u = self.u_prev + del_u
            u = np.clip(u, -self.a_max, self.a_max)

            self.x_prev = x_meas.copy()
            self.u_prev = u.copy()
            return u

        except Exception as e:
            rospy.logwarn(f"MPC QP failed: {e}")
            return self.u_prev.copy()


    def weights_callback(self, msg: Float64MultiArray):
        if len(msg.data) != 9:
            rospy.logwarn(f"Received weights array of length {len(msg.data)}, expected 9. Ignoring.")
            return
        weights_recieved = msg.data
        self.Q_flc = np.diag(weights_recieved[0:6])
        self.R_flc = np.diag(weights_recieved[6:9])
        self._qp_matrices()

def acceleration_to_thrust(accel, yaw_ref, hover_thrust=0.35, gravity=9.81):
    ax, ay, az = accel
    ax = np.clip(ax, -8.0, 8.0)
    ay = np.clip(ay, -8.0, 8.0)
    az = np.clip(az, -8.0, 8.0)

    specific_force = np.array([ax, ay, az - gravity])

    thrust_mag = np.linalg.norm(specific_force)
    if thrust_mag < gravity:
        thrust_mag = gravity

    body_z = -specific_force / np.linalg.norm(specific_force)

    thrust_norm = (thrust_mag / gravity) * hover_thrust
    thrust_norm = np.clip(thrust_norm, hover_thrust*0.5, 1.0)

    if np.linalg.norm(body_z) < 1e-8:
        body_z = np.array([0.0, 0.0, 1.0])
    body_z = body_z / np.linalg.norm(body_z)

    y_C = np.array([-np.sin(yaw_ref), np.cos(yaw_ref), 0.0])
    body_x = np.cross(y_C, body_z)

    if body_z[2] < 0.0:
        body_x = -body_x

    if abs(body_z[2]) < 1e-6:
        body_x = np.array([0.0, 0.0, 1.0])

    body_x = body_x / np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)

    R = np.column_stack([body_x, body_y, body_z])

    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    max_tilt = np.radians(30.0)
    roll = np.clip(roll, -max_tilt, max_tilt)
    pitch = np.clip(pitch, -max_tilt, max_tilt)

    return roll, pitch, yaw, thrust_norm, R

class TrajectoryTrackingMPC:
    def __init__(self):
        self.node_name = "mpc_trajectory_follower_manual"
        rospy.loginfo("="*60)
        rospy.loginfo("MPC Trajectory Tracking (ROS1 + MAVROS) - MANUAL OFFBOARD")
        rospy.loginfo("Input ref: /trajectory/ref_pose & /trajectory/ref_vel")
        rospy.loginfo("Output   : /mavros/setpoint_raw/attitude")
        rospy.loginfo("Mode & ARM dikendalikan manual dari QGC/RC")
        rospy.loginfo("MPC Optimization: 10 Hz | Control Output: 50 Hz")
        rospy.loginfo("="*60)

        # MPC core
        self.mpc = PositionMPC(dt=0.1, y_hor = 10, c_hor = 3)
        rospy.loginfo("MPC Controller initialized: dt = 0.1s, p_hor = 10, c_hor = 3")

        # State MAVROS
        self.current_state = State()
        self.armed = False
        self.offboard_mode = False

        # State UAV (NED)
        self.current_position = np.zeros(3)      # [N,E,D]
        self.current_velocity = np.zeros(3)      # [vN,vE,vD]
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z

        # Trajectory reference (NED)
        self.ref_position = np.array([0.0, 0.0, -2.5])
        self.ref_velocity = np.zeros(3)
        self.ref_yaw = 0.0

        self.ref_pose_received = False
        self.ref_vel_received = False

        # Output MPC (aks NED) + attitude hasil konversi
        self.mpc_acceleration = np.zeros(3)
        self.attitude_roll = 0.0
        self.attitude_pitch = 0.0
        self.attitude_yaw = 0.0
        self.attitude_thrust = 0.35
        self.attitude_R_matrix = np.eye(3)

        self.setpoint_counter = 0

        # ===================== SUBSCRIBERS =====================
        self.state_sub = rospy.Subscriber(
            "/mavros/state", State, self.state_callback, queue_size=10
        )
        self.local_pose_sub = rospy.Subscriber(
            "/mavros/local_position/pose", PoseStamped, self.local_pose_callback, queue_size=10
        )
        self.local_vel_sub = rospy.Subscriber(
            "/mavros/local_position/velocity_local", TwistStamped, self.local_vel_callback, queue_size=10
        )

        # Trajectory references (ENU)
        self.traj_pose_sub = rospy.Subscriber(
            "/trajectory/ref_pose", PoseStamped, self.traj_pose_callback, queue_size=10
        )
        self.traj_vel_sub = rospy.Subscriber(
            "/trajectory/ref_vel", TwistStamped, self.traj_vel_callback, queue_size=10
        )

        # ===================== PUBLISHERS =====================
        self.attitude_pub = rospy.Publisher(
            "/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=20
        )
        
        # Publisher untuk MPC acceleration output (NED frame)
        self.accel_pub = rospy.Publisher(
            "/control/mpc_acceleration", Vector3Stamped, queue_size=20
        )

        # ===================== TIMERS =====================
        self.mpc_timer = rospy.Timer(rospy.Duration(0.1), self.mpc_callback)    # 10 Hz
        self.ctrl_timer = rospy.Timer(rospy.Duration(0.02), self.control_loop)  # 50 Hz
        self.sm_timer = rospy.Timer(rospy.Duration(0.5), self.state_monitor_callback)  # 2 Hz

         # ===================== Callbacks =====================
    def state_callback(self, msg: State):
        prev_armed = self.armed
        prev_offboard = self.offboard_mode

        self.current_state = msg
        self.armed = msg.armed
        self.offboard_mode = (msg.mode == "OFFBOARD")

        if prev_armed != self.armed:
            if self.armed:
                rospy.loginfo("✓ ARMED (dari pilot/QGC)")
            else:
                rospy.logwarn("✗ DISARMED")

        if prev_offboard != self.offboard_mode:
            if self.offboard_mode:
                rospy.loginfo("✓ OFFBOARD MODE ACTIVE (set manual)")
            else:
                rospy.logwarn("✗ OFFBOARD MODE INACTIVE")

    def local_pose_callback(self, msg: PoseStamped):
        # ENU → NED
        self.current_position[0] = msg.pose.position.y      # N
        self.current_position[1] = msg.pose.position.x      # E
        self.current_position[2] = -msg.pose.position.z     # D

        self.current_orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])

    def local_vel_callback(self, msg: TwistStamped):
        # ENU → NED
        self.current_velocity[0] = msg.twist.linear.y
        self.current_velocity[1] = msg.twist.linear.x
        self.current_velocity[2] = -msg.twist.linear.z

    def traj_pose_callback(self, msg: PoseStamped):
        """
        Pose referensi dari trajectory publisher (ENU) → simpan di NED
        """
        # Posisi ENU → NED
        n = msg.pose.position.y
        e = msg.pose.position.x
        d = -msg.pose.position.z
        self.ref_position = np.array([n, e, d])

        # Yaw dari quaternion
        q = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ]
        _, _, yaw = euler_from_quaternion(q)
        self.ref_yaw = yaw

        if not self.ref_pose_received:
            self.ref_pose_received = True
            rospy.loginfo("✅ Trajectory pose reference received.")

    def traj_vel_callback(self, msg: TwistStamped):
        """
        Velocity referensi dari trajectory publisher (ENU) → simpan di NED
        """
        # ENU vel: [vx, vy, vz] → NED: [vn, ve, vd]
        vn = msg.twist.linear.y
        ve = msg.twist.linear.x
        vd = -msg.twist.linear.z
        self.ref_velocity = np.array([vn, ve, vd])

        if not self.ref_vel_received:
            self.ref_vel_received = True
            rospy.loginfo("✅ Trajectory velocity reference received.")

    # ===================== MPC timer =====================

    def mpc_callback(self, event):
        # MPC hanya aktif jika mode OFFBOARD & armed
        if not (self.offboard_mode and self.armed):
            rospy.logwarn_throttle(1.0,f"MPC BLOCKED | offboard={self.offboard_mode} | armed={self.armed} | mode={self.current_state.mode}")
            return
        rospy.loginfo_throttle(1.0, "MPC CALLBACK RUNNING")

        # Jika belum ada trajectory ref, tahan dekat posisi sekarang (hover pada z=-5m)
        if not self.ref_pose_received:
            self.ref_position[0] = self.current_position[0]
            self.ref_position[1] = self.current_position[1]
            self.ref_position[2] = -2.5
            self.ref_velocity[:] = 0.0
        else:
            if not self.ref_vel_received:
                self.ref_velocity[:] = 0.0

        # State & reference vector (NED)
        x = np.concatenate([self.current_position, self.current_velocity])
        x_ref = np.concatenate([self.ref_position, self.ref_velocity])

        acc = self.mpc.compute_control(x, x_ref)
        self.mpc_acceleration = acc
        
        # Publish acceleration output (NED frame)
        accel_msg = Vector3Stamped()
        accel_msg.header.stamp = rospy.Time.now()
        accel_msg.header.frame_id = "base_link_ned"
        accel_msg.vector.x = float(acc[0])  # ax (North)
        accel_msg.vector.y = float(acc[1])  # ay (East)
        accel_msg.vector.z = float(acc[2])  # az (Down)
        self.accel_pub.publish(accel_msg)

        roll, pitch, yaw, thrust, R = acceleration_to_thrust(
            acc, self.ref_yaw, hover_thrust=0.35, gravity=9.81
        )

        self.attitude_roll = roll
        self.attitude_pitch = pitch
        self.attitude_yaw = yaw
        self.attitude_thrust = thrust
        self.attitude_R_matrix = R

        pos_err = np.linalg.norm(self.ref_position - self.current_position)
        vel_err = np.linalg.norm(self.ref_velocity - self.current_velocity)

        rospy.loginfo_throttle(
            1.0,
            f"🎯 MPC TRAJ STATE:\n"
            f"   Pos: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] → "
            f"[{self.ref_position[0]:.1f}, {self.ref_position[1]:.1f}, {self.ref_position[2]:.1f}] | Err: {pos_err:.2f}m\n"
            f"   Vel: [{self.current_velocity[0]:.2f}, {self.current_velocity[1]:.2f}, {self.current_velocity[2]:.2f}] → "
            f"[{self.ref_velocity[0]:.2f}, {self.ref_velocity[1]:.2f}, {self.ref_velocity[2]:.2f}] m/s | Err: {vel_err:.2f}\n"
            f"   Acc: [{acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}] m/s² | Thrust: {thrust:.3f}"
        )

    # ===================== Attitude setpoint =====================

    def rotmat_nedfrd_to_quat_enuflu(self, R_ned_from_body_frd: np.ndarray):
        T_ENU_NED = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0,-1]
        ], dtype=float)

        T_FRD_FLU = np.diag([1, -1, -1]).astype(float)

        R_enu_from_body_flu = T_ENU_NED @ R_ned_from_body_frd @ T_FRD_FLU
        wxyz = self.rotation_matrix_to_quaternion(R_enu_from_body_flu)
        return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=float)

    def control_loop(self, event):
    
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.header.frame_id = "map"

        att.type_mask = (
            AttitudeTarget.IGNORE_ROLL_RATE |
            AttitudeTarget.IGNORE_PITCH_RATE |
            AttitudeTarget.IGNORE_YAW_RATE
        )

        q_xyzw = self.rotmat_nedfrd_to_quat_enuflu(self.attitude_R_matrix)
        att.orientation.x = float(q_xyzw[0])
        att.orientation.y = float(q_xyzw[1])
        att.orientation.z = float(q_xyzw[2])
        att.orientation.w = float(q_xyzw[3])

        att.thrust = float(np.clip(self.attitude_thrust, 0.0, 1.0))
        att.body_rate.x = 0.0
        att.body_rate.y = 0.0
        att.body_rate.z = 0.0

        self.attitude_pub.publish(att)
        self.setpoint_counter += 1

        if self.setpoint_counter % 50 == 0:
            rospy.loginfo(
                f"📤 AttitudeTarget ENU: thrust={att.thrust:.3f} "
                f"q=[{att.orientation.x:.3f},{att.orientation.y:.3f},"
                f"{att.orientation.z:.3f},{att.orientation.w:.3f}]"
            )

    # ===================== State monitor (tanpa command) =====================

    def state_monitor_callback(self, event):
       
        if self.offboard_mode and self.armed:
            err = np.linalg.norm(self.current_position - self.ref_position)
            vel_norm = np.linalg.norm(self.current_velocity)
            acc_norm = np.linalg.norm(self.mpc_acceleration)

            if self.ref_pose_received:
                rospy.loginfo(
                    f"[MPC ACTIVE] Pos: [{self.current_position[0]:.1f}, {self.current_position[1]:.1f}, {self.current_position[2]:.1f}] | "
                    f"Ref: [{self.ref_position[0]:.1f}, {self.ref_position[1]:.1f}, {self.ref_position[2]:.1f}] | "
                    f"Err: {err:.2f}m | Vel: {vel_norm:.2f} m/s | Acc: {acc_norm:.2f} m/s²"
                )
            else:
                rospy.loginfo(
                    f"[MPC ACTIVE] Hovering dekat posisi sekarang, belum ada trajectory reference."
                )
        else:
            rospy.loginfo_throttle(
                2.0,
                f"[MPC STANDBY] mode={self.current_state.mode}, armed={self.armed}, "
                f"ref_pose={'OK' if self.ref_pose_received else 'NO'}, "
                f"ref_vel={'OK' if self.ref_vel_received else 'NO'}"
            )

    # ===================== Helpers =====================

    def rotation_matrix_to_quaternion(self, R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])


def main():
    rospy.init_node("mpc_trajectory_follower_manual", anonymous=False)
    node = TrajectoryTrackingMPC()
    rospy.loginfo("MPC Trajectory Follower ROS1 (manual OFFBOARD) started.")
    rospy.spin()


if __name__ == "__main__":
    main()