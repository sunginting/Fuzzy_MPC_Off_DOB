#!/usr/bin/env python3
import rospy
import numpy as np
import quadprog

from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from mavros_msgs.msg import State, AttitudeTarget
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

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

        # Cost matrices
        self.Q  = np.diag([
            40.0, 40.0, 100.0,   # posisi
            20.0, 20.0, 12.0     # kecepatan
        ])

        self.R = np.diag([1.0, 1.0, 0.10])     # penalti besarnya u
        self.R_delta = np.diag([0.2, 0.2, 0.06])  # penalti perubahan u

        self.a_max = 5.0

        self.A_stack = np.zeros((self.y_hor*self.nx, self.nx))
        self.B_stack = np.zeros((self.y_hor*self.nx, self.c_hor*self.nu))
        self.R_delta_bar = np.zeros((self.c_hor*self.nu, self.c_hor*self.nu))
        self.H = np.zeros((self.c_hor*self.nu, self.c_hor*self.nu))

        self.K = np.eye(self.nx)

        self.x_est = np.zeros(self.nx)
        self.x_est_prev = np.zeros(self.nx)

        self.dist_x = np.zeros(self.nx)
        self.dist_y = np.zeros(self.nx)

        self.u = np.zeros(self.nu)
        self.u_prev = np.zeros(self.nu)

        self.y_star = np.zeros(self.nx)

        # SUBSCRIBERS
        # self.K_sub = rospy.Subscriber('/flc/disturbance_gain', Float32MultiArray, self.K_callback, queue_size=10) 

        # PUBLISHER
        self.est_pub = rospy.Publisher('/mpc/dist_estimate', Float32MultiArray, queue_size=10)

        self.build_prediction_matrix()
    
    # def K_callback(self,msg:Float32MultiArray):
    #     if len(msg.data) != 4:
    #         rospy.logwarn("Received K data with incorrect length")
    #         return
    #     self.K = np.array([
    #         [msg.data[0], 0, 0, 0, 0, 0],
    #         [0, msg.data[0], 0, 0, 0, 0],
    #         [0, 0, msg.data[1], 0, 0, 0],
    #         [0, 0, 0, msg.data[2], 0, 0],
    #         [0, 0, 0, 0, msg.data[2], 0],
    #         [0, 0, 0, 0, 0, msg.data[3]]
    #     ])
    #     if self.hurwitz_stability():
    #         pass
    #     else:
    #         rospy.logwarn("Rejecting K update — would make observer unstable")
    #         self.K = np.eye(self.nx)  # revert

    def hurwitz_stability(self):
        self.K = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        A_KCA = self.A_model - self.K @ self.C_model @ self.A_model
        eigenvalues = np.linalg.eigvals(A_KCA)
        is_stable = bool(np.all(np.abs(eigenvalues) < 1))

        rospy.loginfo(f"Eigenvalues: {np.round(np.abs(eigenvalues), 4)}")
        rospy.loginfo(f"Hurwitz: {is_stable}")
        return is_stable  # ✅ actually return the result
    
    def disturbance_observer(self, y_meas):
        # Current Observer
        self.y_star = self.C_model @ (self.A_model @ self.x_est_prev + self.B_model @ self.u_prev)

        self.x_est = (self.A_model@self.x_est_prev) + (self.B_model@self.u_prev) + self.K@(y_meas - self.y_star)
        self.x_est_prev = self.x_est.copy()

        # State Disturbance
        dist_x = self.K @ (y_meas - self.y_star)
        dist_y = y_meas - self.C_model @ self.x_est

        return dist_x, dist_y

    def build_prediction_matrix(self):
        for i in range(self.y_hor):
            self.A_stack[i*self.nx:(1+i)*self.nx, :] = np.linalg.matrix_power(self.A_model,i)

        for j in range(self.y_hor):
            for k in range(min(j+1, self.c_hor)):
                A_power = np.eye(self.nx)
                for _ in range(j-k):
                    A_power = A_power @ self.A_model
                self.B_stack[j*self.nx:(j+1)*self.nx, k*self.nu:(k+1)*self.nu] = A_power @ self.B_model
        self.qp_matrices()

    def qp_matrices(self):
        Q_bar = np.kron(np.eye(self.y_hor), self.Q)
        R_bar = np.kron(np.eye(self.c_hor), self.R)

        R_delta_bar = np.zeros((self.c_hor*self.nu, self.c_hor*self.nu))
        for i in range(self.c_hor):
            # Diagonal: R_delta for endpoints, 2*R_delta for middle terms
            if i == 0 or i == self.c_hor - 1:
                R_delta_bar[i*self.nu:(i+1)*self.nu,
                            i*self.nu:(i+1)*self.nu] = self.R_delta
            else:
                R_delta_bar[i*self.nu:(i+1)*self.nu,
                            i*self.nu:(i+1)*self.nu] = 2 * self.R_delta
            # Off-diagonals
            if i > 0:
                R_delta_bar[i*self.nu:(i+1)*self.nu,
                            (i-1)*self.nu:i*self.nu] = -self.R_delta
                R_delta_bar[(i-1)*self.nu:i*self.nu,
                            i*self.nu:(i+1)*self.nu] = -self.R_delta

        H_mat = self.B_stack.T @ Q_bar @ self.B_stack + R_bar + R_delta_bar
        self.H = (H_mat + H_mat.T) / 2.0

        # Verify positive definiteness
        try:
            np.linalg.cholesky(self.H)
            rospy.loginfo("H matrix is positive definite ✅")
        except np.linalg.LinAlgError:
            rospy.logwarn("H matrix is NOT positive definite ❌")

        self.Q_bar = Q_bar
        self.R_delta_bar = R_delta_bar

    def compute_control(self, x_meas, x_ref):
        ref = np.tile(x_ref, self.y_hor)
        y_meas = self.C_model@x_meas
        self.dist_x, self.dist_y = self.disturbance_observer(y_meas)
        dist_est_msg = Float32MultiArray()
        dist_est_msg.data = (self.dist_x.flatten().tolist() + self.dist_y.flatten().tolist())
        self.est_pub.publish(dist_est_msg)

        dist_x_stack = np.tile(self.dist_x, self.y_hor)
        dist_y_stack = np.tile(self.dist_y, self.y_hor)

        err_pred = ref - self.A_stack@x_meas - dist_x_stack - dist_y_stack

        u_prev_ext = np.tile(self.u_prev, self.c_hor)

        f_tracking = self.B_stack.T @ self.Q_bar @ err_pred
        f_rate = -self.R_delta_bar @ u_prev_ext
        f = f_tracking + f_rate

        C = np.vstack([
            np.eye(self.c_hor*self.nu),
            -np.eye(self.c_hor*self.nu)
        ])
        b = np.hstack([
            np.full(self.c_hor*self.nu, -self.a_max),
            np.full(self.c_hor*self.nu, -self.a_max)
        ])

        try:
            u_opt = quadprog.solve_qp(self.H, -f, C.T, b, meq=0)[0]
            u = u_opt[:self.nu]

            z_err = abs(x_meas[2] - x_ref[2])
            if z_err > 0.3:
                lateral_reduction = np.clip(1.0 - (z_err - 0.3)*2.0, 0.3, 1.0)
                u[0] *= lateral_reduction
                u[1] *= lateral_reduction

            u = np.clip(u, -self.a_max, self.a_max)
            self.u_prev = u.copy()
            return u

        except Exception as e:
            rospy.logwarn(f"MPC QP failed: {e}")
            return np.zeros(self.nu)
        
def acceleration_to_attitude_thrust_px4(accel_ned, yaw_desired, hover_thrust=0.35, gravity=9.81):
    ax, ay, az = accel_ned

    ax = np.clip(ax, -8.0, 8.0)
    ay = np.clip(ay, -8.0, 8.0)
    az = np.clip(az, -8.0, 8.0)

    # In NED: gravity = +9.81 in D direction
    # Thrust needed = commanded acceleration - gravity (gravity already pulls down)
    # To hover: az_cmd=0, thrust must cancel gravity → specific_force_z = 0 - (-9.81) = +9.81... 
    
    # Simpler: treat az as the NET vertical command
    # Body needs to produce: [ax, ay, az] + [0, 0, g] to get net [ax, ay, az]
    specific_force = np.array([ax, ay, az + gravity])  # gravity compensation in NED

    thrust_mag = np.linalg.norm(specific_force)
    thrust_mag = max(thrust_mag, 0.1 * gravity)

    body_z = specific_force / np.linalg.norm(specific_force)
    thrust_norm = (thrust_mag / gravity) * hover_thrust
    thrust_norm = np.clip(thrust_norm, 0.1, 1.0)

    if np.linalg.norm(body_z) < 1e-8:
        body_z = np.array([0.0, 0.0, 1.0])
    body_z = body_z / np.linalg.norm(body_z)

    y_C = np.array([-np.sin(yaw_desired), np.cos(yaw_desired), 0.0])
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

class MPCTrajectoryFollowerManualROS1:

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
        self.mpc = PositionMPC(dt=0.1, y_hor=10, c_hor=3)
        rospy.loginfo("MPC Controller initialized: dt=0.1s, Np=10, Nc=3")

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
            return

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

        roll, pitch, yaw, thrust, R = acceleration_to_attitude_thrust_px4(
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
    node = MPCTrajectoryFollowerManualROS1()
    rospy.loginfo("MPC Trajectory Follower ROS1 (manual OFFBOARD) started.")
    rospy.spin()


if __name__ == "__main__":
    main()