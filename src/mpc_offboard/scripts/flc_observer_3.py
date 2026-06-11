#!/home/sundy/venv_hexacopter/bin/python
import rospy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

class FuzzyLogicControl:
    flc_labels = ['NB','NS','Z', 'PS', 'PB']
    rule_table = np.array([
        ['B', 'B', 'B', 'B', 'M'],
        ['B', 'B', 'B', 'M', 'S'],
        ['B', 'M', 'S', 'M', 'B'],
        ['S', 'M', 'B', 'B', 'B'],
        ['M', 'B', 'B', 'B', 'B']
        ])

    def __init__(self):
        self.position_desired = np.zeros(3)
        self.position_actual = np.zeros(3)
        self.velocity_desired = np.zeros(3)
        self.velocity_actual = np.zeros(3)

        self.sim = self._build_flc_pose()

        self.Q_x = 40
        self.Q_y = 40
        self.Q_alt = 100
        self.Q_vx = 20
        self.Q_vy = 20
        self.Q_vz = 12

        self.R_delta_x = 0.12
        self.R_delta_y = 0.12
        self.R_delta_z = 0.06

        self.K_x = 0.9
        self.K_y = 0.9
        self.K_alt = 0.95
        self.K_vx = 0.9
        self.K_vy = 0.9
        self.K_vz = 0.95

        self.Q_max = np.array([75.0, 75.0, 130.0, 10.0, 10.0, 12.0])
        self.Q_min = np.array([35.0, 35.0, 100.0, 35.0, 35.0, 7.0])
        self.R_delta_max = np.array([0.12, 0.12, 0.06])
        self.R_delta_min = np.array([0.08, 0.08, 0.02])

        self.K_max = np.array([0.85, 0.85, 0.9, 0.7, 0.7, 0.6])
        self.K_min = np.array([0.2, 0.2, 0.2, 0.0, 0.0, 0.0])

        self.d_error_x = 0.0
        self.d_error_y = 0.0
        self.d_error_alt = 0.0

        self.previous_x_error = 0.0
        self.previous_y_error = 0.0
        self.previous_alt_error = 0.0

        self.previous_time = 0.0
        self.current_time = 0.0

        # PUBLISH
        self.dist_gain_pub = rospy.Publisher('/flc/disturbance_gain', Float32MultiArray, queue_size=10)
        self.weights_MPC_pub = rospy.Publisher('/flc/mpc_weights', Float32MultiArray, queue_size=10)

        # SUBSCRIBE
        self.pose_actual_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_actual_callback, queue_size=10)
        self.pose_desired_sub = rospy.Subscriber("/trajectory/ref_pose", PoseStamped, self.pose_desired_callback, queue_size=10)
        self.velo_actual_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velo_actual_callback,queue_size=10)
        self.velo_desired_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_desired_callback,queue_size=10)

        # TIMER 20 Hz
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def timer_callback(self, event):
        self.compute_flc()

    def pose_actual_callback(self, msg:PoseStamped):
        # ENU -> NED
        self.position_actual = np.array([msg.pose.position.y, msg.pose.position.x, -msg.pose.position.z])
        self.orientation_actual_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_actual_euler = euler_from_quaternion(self.orientation_actual_quat)

    def pose_desired_callback(self, msg:PoseStamped):
        # ENU -> NED
        self.position_desired = np.array([msg.pose.position.y, msg.pose.position.x, -msg.pose.position.z])
        self.orientation_desired_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_desired_euler = euler_from_quaternion(self.orientation_desired_quat)

    def velo_actual_callback(self, msg:TwistStamped):
        # ENU -> NED
        self.velocity_actual = np.array([msg.twist.linear.y, msg.twist.linear.x, -msg.twist.linear.z])

    def velo_desired_callback(self, msg:TwistStamped):
        # ENU -> NED
        self.velocity_desired = np.array([msg.twist.linear.y, msg.twist.linear.x, -msg.twist.linear.z])

    def generate_rules(self, error_ante, d_error_ante, out_mem, table):
        rules = []
        for i, e_label in enumerate(self.flc_labels):
            for j, de_label in enumerate(self.flc_labels):
                out_table = table[i][j]
                rule = ctrl.Rule(error_ante[e_label] & d_error_ante[de_label],out_mem[out_table])
                rules.append(rule)
        return rules

    def _build_flc_pose(self):
        input_min = -1.0
        input_max = 1.0

        alpha_min = 0.0
        alpha_max = 1.0
        # ANTECEDANT
        input_1 = ctrl.Antecedent(np.linspace(input_min, input_max, 200), 'input_1')
        input_2 = ctrl.Antecedent(np.linspace(input_min, input_max, 200), 'input_2')

        # CONSEQUENT
        alpha = ctrl.Consequent(np.linspace(alpha_min, alpha_max, 200), 'alpha')

        # MEMBERSHIP FUNCTION
        input_1['NB'] = fuzz.trimf(input_1.universe, [input_min, input_min, (1/2)*input_min])
        input_1['NS'] = fuzz.trimf(input_1.universe, [input_min, (1/2)*input_min, 0.0])
        input_1['Z'] = fuzz.trimf(input_1.universe, [(1/2)*input_min, 0.0, (1/2)*input_max])
        input_1['PS'] = fuzz.trimf(input_1.universe, [0.0, (1/2)*input_max, input_max])
        input_1['PB'] = fuzz.trimf(input_1.universe, [(1/2)*input_max, input_max, input_max])

        input_2['NB'] = fuzz.trimf(input_2.universe, [input_min, input_min, (1/2)*input_min])
        input_2['NS'] = fuzz.trimf(input_2.universe, [input_min, (1/2)*input_min, 0.0])
        input_2['Z'] = fuzz.trimf(input_2.universe, [(1/2)*input_min, 0.0, (1/2)*input_max])
        input_2['PS'] = fuzz.trimf(input_2.universe, [0.0, (1/2)*input_max, input_max])
        input_2['PB'] = fuzz.trimf(input_2.universe, [(1/2)*input_max, input_max, input_max])

        alpha['S'] = fuzz.trimf(alpha.universe, [alpha_min, alpha_min, (1/2)*alpha_max])
        alpha['M'] = fuzz.trimf(alpha.universe, [0.0, (1/2)*alpha_max, alpha_max])
        alpha['B'] = fuzz.trimf(alpha.universe, [(1/2)*alpha_max, alpha_max, alpha_max])

        rules = self.generate_rules(input_1, input_2, alpha, self.rule_table)
        self.sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
        return self.sim
    
    def _run_flc(self, input_1, input_2):
        self.sim.input['input_1'] = np.clip(input_1, -1.0, 1.0)
        self.sim.input['input_2'] = np.clip(input_2, -1.0, 1.0)
        self.sim.compute()

        alpha = self.sim.output['alpha']
        return alpha
    
    def compute_flc(self):
        self.current_time = rospy.Time.now().to_sec()
        if self.previous_time == 0.0:
            self.previous_time = self.current_time
            return

        error_x = self.position_desired[0]-self.position_actual[0]
        error_y = self.position_desired[1]-self.position_actual[1]
        error_z = self.position_desired[2]-self.position_actual[2]

        x_error_norm = np.clip(error_x/1, -1.0, 1.0)
        y_error_norm = np.clip(error_y/1, -1.0, 1.0)
        alt_error_norm = np.clip(error_z/0.15, -1.0, 1.0)

        dt = self.current_time - self.previous_time
        if dt <= 0:
            return
        else:
            self.d_error_x = (error_x - self.previous_x_error) / dt
            self.d_error_y = (error_y - self.previous_y_error) / dt
            self.d_error_alt = (error_z - self.previous_alt_error) / dt

        d_err_x_norm = np.clip(self.d_error_x/0.55, -1.0, 1.0)
        d_err_y_norm = np.clip(self.d_error_y/0.55, -1.0, 1.0)
        d_err_alt_norm = np.clip(self.d_error_alt/0.1, -1.0, 1.0)

        alpha_x = self._run_flc(x_error_norm, d_err_x_norm)
        alpha_y = self._run_flc(y_error_norm, d_err_y_norm)
        alpha_z = self._run_flc(alt_error_norm, d_err_alt_norm)

        self.K_x = self.K_min[0] + alpha_x*(self.K_max[0] - self.K_min[0])
        self.K_y = self.K_min[1] + alpha_y*(self.K_max[1] - self.K_min[1])
        self.K_alt = self.K_min[2] + alpha_z*(self.K_max[2] - self.K_min[2])
        self.K_vx = self.K_max[3] - alpha_x*(self.K_max[3] - self.K_min[3])
        self.K_vy = self.K_max[4] - alpha_y*(self.K_max[4] - self.K_min[4])
        self.K_vz = self.K_max[5] - alpha_z*(self.K_max[5] - self.K_min[5])

        self.Q_x = self.Q_min[0] + alpha_x*(self.Q_max[0] - self.Q_min[0])
        self.Q_y = self.Q_min[1] + alpha_y*(self.Q_max[1] - self.Q_min[1])
        self.Q_alt = self.Q_min[2] + alpha_z*(self.Q_max[2] - self.Q_min[2])
        self.Q_vx = self.Q_max[3] - alpha_x*(self.Q_max[3] - self.Q_min[3])
        self.Q_vy = self.Q_max[4] - alpha_y*(self.Q_max[4] - self.Q_min[4])
        self.Q_vz = self.Q_max[5] - alpha_z*(self.Q_max[5] - self.Q_min[5])

        self.R_delta_x = self.R_delta_max[0] - alpha_x*(self.R_delta_max[0] - self.R_delta_min[0])
        self.R_delta_y = self.R_delta_max[1] - alpha_y*(self.R_delta_max[1] - self.R_delta_min[1])
        self.R_delta_alt = self.R_delta_max[2] - alpha_z*(self.R_delta_max[2] - self.R_delta_min[2])

        dist_gain_msg = Float32MultiArray()
        dist_gain_msg.data = [self.K_x, self.K_y, self.K_alt, self.K_vx, self.K_vy, self.K_vz]
        self.dist_gain_pub.publish(dist_gain_msg)

        weights_msg = Float32MultiArray()
        weights_msg.data = [self.Q_x, self.Q_y, self.Q_alt, self.Q_vx, self.Q_vy, self.Q_vz, self.R_delta_x, self.R_delta_y, self.R_delta_z]
        self.weights_MPC_pub.publish(weights_msg)

        self.previous_x_error = error_x
        self.previous_y_error = error_y
        self.previous_alt_error = error_z

        self.previous_time = self.current_time

if __name__ == "__main__":
    rospy.init_node("flc_hexacopter")
    FuzzyLogicControl()
    rospy.spin()