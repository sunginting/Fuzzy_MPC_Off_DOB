#!/home/sundy/venv_hexacopter/bin/python
import rospy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

class FuzzyLogicControl:
    label_1 = ['Z', 'S', 'B']
    label_2 = ['NB','NS','Z', 'PS', 'PB']
    agg_table = np.array([
        ['B', 'M', 'S', 'M', 'B'],
        ['S', 'M', 'B', 'B', 'B'],
        ['M', 'B', 'B', 'B', 'B']
        ])
    cons_table = np.array([
        ['M', 'S', 'S', 'S', 'M'],
        ['S', 'S', 'M', 'B', 'B'],
        ['S', 'M', 'B', 'B', 'B']
        ])

    def __init__(self):
        self.position_desired = np.zeros(3)
        self.position_actual = np.zeros(3)
        self.velocity_desired = np.zeros(3)
        self.velocity_actual = np.zeros(3)

        self.sim_pose = None
        self.sim_velo = None

        self.Q_x = 50
        self.Q_y = 50
        self.Q_alt = 110
        self.Q_vx = 20
        self.Q_vy = 20
        self.Q_vz = 20

        self.R_x = 0.6
        self.R_y = 0.6
        self.R_alt = 0.06

        self.Q_max = np.array([70.0, 70.0, 150.0, 60.0, 60.0, 52.0])
        self.Q_min = np.array([30.0, 30.0, 100.0, 35.0, 35.0, 12.0])
        self.R_max = np.array([0.8, 0.8, 0.10])
        self.R_min = np.array([0.1, 0.1, 0.03])

        self.d_error_x = 0.0
        self.d_error_y = 0.0
        self.d_error_alt = 0.0
        self.d_error_vx = 0.0
        self.d_error_vy = 0.0
        self.d_error_vz = 0.0

        self.previous_x_error = 0.0
        self.previous_y_error = 0.0
        self.previous_alt_error = 0.0
        self.previous_vx_error = 0.0
        self.previous_vy_error = 0.0
        self.previous_vz_error = 0.0

        self.previous_time = 0.0
        self.current_time = 0.0

        # PUBLISH
        self.weights_MPC_pub = rospy.Publisher('/flc/mpc_weights', Float32MultiArray, queue_size=10)

        # SUBSCRIBE
        self.pose_actual_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_actual_callback, queue_size=10)
        self.pose_desired_sub = rospy.Subscriber("/trajectory/ref_pose", PoseStamped, self.pose_desired_callback, queue_size=10)
        self.velo_actual_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velo_actual_callback,queue_size=10)
        self.velo_desired_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_desired_callback,queue_size=10)

        # TIMER 20 Hz
        self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)

    def timer_callback(self, event):
        self._build_flc_pose()
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
        for i, e_label in enumerate(self.label_1):
            for j, de_label in enumerate(self.label_2):
                out_table = table[i][j]
                rule = ctrl.Rule(error_ante[e_label] & d_error_ante[de_label],out_mem[out_table])
                rules.append(rule)
        return rules

    def _build_flc_pose(self):
        input_min_1 = 0.0
        input_max_1 = 1.0

        input_min_2 = -1.0
        input_max_2 = 1.0

        alpha_min = 0.0
        alpha_max = 1.0
        # ANTECEDANT
        input_1 = ctrl.Antecedent(np.linspace(input_min_1, input_max_1, 200), 'input_1')
        input_2 = ctrl.Antecedent(np.linspace(input_min_2, input_max_2, 200), 'input_2')

        # CONSEQUENT
        alpha = ctrl.Consequent(np.linspace(alpha_min, alpha_max, 200), 'alpha')

        # MEMBERSHIP FUNCTION
        input_1['Z'] = fuzz.trimf(input_1.universe, [input_min_1, input_min_1,(1/2)*input_max_1])
        input_1['S'] = fuzz.trimf(input_1.universe, [input_min_1, (1/2)*input_max_1, input_max_1])
        input_1['B'] = fuzz.trimf(input_1.universe, [(1/2)*input_max_1, input_max_1, input_max_1])

        input_2['NB'] = fuzz.trimf(input_2.universe, [input_min_2, input_min_2, (1/2)*input_min_2])
        input_2['NS'] = fuzz.trimf(input_2.universe, [input_min_2, (1/2)*input_min_2, 0.0])
        input_2['Z'] = fuzz.trimf(input_2.universe, [(1/2)*input_min_2, 0.0, (1/2)*input_max_2])
        input_2['PS'] = fuzz.trimf(input_2.universe, [0.0, (1/2)*input_max_2, input_max_2])
        input_2['PB'] = fuzz.trimf(input_2.universe, [(1/2)*input_max_2, input_max_2, input_max_2])

        alpha['S'] = fuzz.trimf(alpha.universe, [alpha_min, alpha_min, (1/2)*alpha_max])
        alpha['M'] = fuzz.trimf(alpha.universe, [0.0, (1/2)*alpha_max, alpha_max])
        alpha['B'] = fuzz.trimf(alpha.universe, [(1/2)*alpha_max, alpha_max, alpha_max])

        rule_agg = self.generate_rules(input_1, input_2, alpha, self.agg_table)
        rule_cons = self.generate_rules(input_1, input_2, alpha, self.cons_table)
        self.sim_agg = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rule_agg))
        self.sim_cons = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rule_cons))
    
    def _run_flc(self, sim, input_1, input_2):
        sim.input['input_1'] = np.clip(input_1, 0.0, 1.0)
        sim.input['input_2'] = np.clip(input_2, -1.0, 1.0)
        sim.compute()

        alpha = sim.output['alpha']
        return alpha
    
    def compute_flc(self):
        self.current_time = rospy.Time.now().to_sec()
        if self.previous_time == 0.0:
            self.previous_time = self.current_time
            return

        error_x = abs(self.position_desired[0]-self.position_actual[0])
        error_y = abs(self.position_desired[1]-self.position_actual[1])
        error_z = abs(self.position_desired[2]-self.position_actual[2])
        error_vx = abs(self.velocity_desired[0]-self.velocity_actual[0])
        error_vy = abs(self.velocity_desired[1]-self.velocity_actual[1])
        error_vz = abs(self.velocity_desired[2]-self.velocity_actual[2])

        x_error_norm = np.clip(error_x/0.7, 0.0, 1.0)
        y_error_norm = np.clip(error_y/0.7, 0.0, 1.0)
        alt_error_norm = np.clip(error_z/0.15, 0.0, 1.0)
        vx_error_norm = np.clip(error_vx/0.75, 0.0, 1.0)
        vy_error_norm = np.clip(error_vy/0.75, 0.0, 1.0)
        vz_error_norm = np.clip(error_vz/0.2, 0.0, 1.0)

        dt = self.current_time - self.previous_time
        if dt <= 0:
            return
        else:
            self.d_error_x = (error_x - self.previous_x_error) / dt
            self.d_error_y = (error_y - self.previous_y_error) / dt
            self.d_error_alt = (error_z - self.previous_alt_error) / dt
            self.d_error_vx = (error_vx - self.previous_vx_error) / dt
            self.d_error_vy = (error_vy - self.previous_vy_error) / dt
            self.d_error_vz = (error_vz - self.previous_vz_error) / dt

        d_err_x_norm = np.clip(self.d_error_x/0.35, -1.0, 1.0)
        d_err_y_norm = np.clip(self.d_error_y/0.35, -1.0, 1.0)
        d_err_alt_norm = np.clip(self.d_error_alt/0.075, -1.0, 1.0)
        d_err_vx_norm = np.clip(self.d_error_vx/0.35, -1.0, 1.0)
        d_err_vy_norm = np.clip(self.d_error_vy/0.35, -1.0, 1.0)
        d_err_vz_norm = np.clip(self.d_error_vz/0.1, -1.0, 1.0)

        alpha_ax = self._run_flc(self.sim_agg, x_error_norm, d_err_x_norm)
        alpha_ay = self._run_flc(self.sim_agg, y_error_norm, d_err_y_norm)
        alpha_az = self._run_flc(self.sim_agg, alt_error_norm, d_err_alt_norm)
        alpha_cx = self._run_flc(self.sim_cons, vx_error_norm, d_err_vx_norm)
        alpha_cy = self._run_flc(self.sim_cons, vy_error_norm, d_err_vy_norm)
        alpha_cz = self._run_flc(self.sim_cons, vz_error_norm, d_err_vz_norm)

        self.Q_x = self.Q_min[0] + alpha_ax*(self.Q_max[0] - self.Q_min[0])
        self.Q_y = self.Q_min[1] + alpha_ay*(self.Q_max[1] - self.Q_min[1])
        self.Q_alt = self.Q_min[2] + alpha_az*(self.Q_max[2] - self.Q_min[2])

        if x_error_norm > vz_error_norm-0.1:
            self.Q_vx = self.Q_max[3] - alpha_cx*(self.Q_max[3] - self.Q_min[3])
        else:
            self.Q_vx = self.Q_min[3] + alpha_cx*(self.Q_max[3] - self.Q_min[3])

        if y_error_norm > vy_error_norm-0.1:
            self.Q_vy = self.Q_max[4] - alpha_cy*(self.Q_max[4] - self.Q_min[4])
        else:
            self.Q_vy = self.Q_min[4] + alpha_cy*(self.Q_max[4] - self.Q_min[4])
            
        if alt_error_norm > vz_error_norm-0.3:
            self.Q_vz = self.Q_max[5] - alpha_cz*(self.Q_max[5] - self.Q_min[5])
        else:
            self.Q_vz = self.Q_min[5] + alpha_cz*(self.Q_max[5] - self.Q_min[5])

        self.R_x = self.R_max[0] - alpha_ax*(self.R_max[0] - self.R_min[0])
        self.R_y = self.R_max[1] - alpha_ay*(self.R_max[1] - self.R_min[1])
        self.R_alt = self.R_max[2] - alpha_az*(self.R_max[2] - self.R_min[2])

        weights_msg = Float32MultiArray()
        weights_msg.data = [self.Q_x, self.Q_y, self.Q_alt, self.Q_vx, self.Q_vy, self.Q_vz, self.R_x, self.R_y, self.R_alt]
        self.weights_MPC_pub.publish(weights_msg)

        self.previous_x_error = error_x
        self.previous_y_error = error_y
        self.previous_alt_error = error_z
        self.previous_vx_error = error_vx
        self.previous_vy_error = error_vy
        self.previous_vz_error = error_vz

        self.previous_time = self.current_time

if __name__ == "__main__":
    rospy.init_node("flc_hexacopter")
    FuzzyLogicControl()
    rospy.spin()