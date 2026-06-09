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
        ['PB', 'PB', 'NB', 'PB', 'PB'],
        ['PB', 'PS', 'NS', 'PS', 'PB'],
        ['PB', 'PS', 'Z',  'PS', 'PB'],
        ['PB', 'PS', 'NS', 'PS', 'PB'],
        ['PB', 'PB', 'NB', 'PB', 'PB']
        ])

    def __init__(self):
        self.position_desired = np.zeros(3)
        self.position_actual = np.zeros(3)

        self.pose_sim = self._build_flc_pose()
        self.alt_sim = self._build_flc_alt()

        self.K_x = 0.9
        self.K_y = 0.9
        self.K_z = 0.95

        self.prev_K_x = 0.0
        self.prev_K_y = 0.0
        self.prev_K_alt = 0.0

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

        # SUBSCRIBE
        self.pose_actual_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_actual_callback, queue_size=10)
        self.pose_desired_sub = rospy.Subscriber("/trajectory/ref_pose", PoseStamped, self.pose_desired_callback, queue_size=10)
        self.velo_actual_sub = rospy.Subscriber("/mavros/local_position/velocity_local", TwistStamped, self.velo_actual_callback,queue_size=10)
        self.velo_desired_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_desired_callback,queue_size=10)

    def pose_actual_callback(self, msg:PoseStamped):
        # ENU -> NED
        self.position_actual = np.array([msg.pose.position.y, msg.pose.position.x, -msg.pose.position.z])
        self.orientation_actual_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_actual_euler = euler_from_quaternion(self.orientation_actual_quat)

        self.current_time = rospy.Time.now().to_sec()
        if self.previous_time == 0.0:
            self.previous_time = self.current_time
            return

        error_x = self.position_desired[0]-self.position_actual[0]
        error_y = self.position_desired[1]-self.position_actual[1]
        error_z = self.position_desired[2]-self.position_actual[2]

        x_error_norm = np.clip(error_x/1.5, -1.0, 1.0)
        y_error_norm = np.clip(error_y/1.5, -1.0, 1.0)
        alt_error_norm = np.clip(error_z/0.2, -1.0, 1.0)

        dt = self.current_time - self.previous_time
        if dt <= 0:
            return
        else:
            self.d_error_x = (error_x - self.previous_x_error) / dt
            self.d_error_y = (error_y - self.previous_y_error) / dt
            self.d_error_alt = (error_z - self.previous_alt_error) / dt

        d_err_x_norm = np.clip(self.d_error_x/0.75, -1.0, 1.0)
        d_err_y_norm = np.clip(self.d_error_y/0.75, -1.0, 1.0)
        d_err_alt_norm = np.clip(self.d_error_alt/0.1, -1.0, 0.5)

        del_K_x = self._run_flc_pose(x_error_norm, d_err_x_norm)
        del_K_y = self._run_flc_pose(y_error_norm, d_err_y_norm)
        del_K_alt = self._run_flc_alt(alt_error_norm, d_err_alt_norm)

        self.K_x = np.clip(self.prev_K_x + del_K_x, 0.75, 0.95)
        self.K_y = np.clip(self.prev_K_y + del_K_y, 0.75, 0.95)
        self.K_alt = np.clip(self.prev_K_alt + del_K_alt, 0.75, 0.95)

        dist_gain_msg = Float32MultiArray()
        dist_gain_msg.data = [self.K_x, self.K_y, self.K_alt]
        self.dist_gain_pub.publish(dist_gain_msg)

        self.previous_x_error = error_x.copy()
        self.previous_y_error = error_y.copy()
        self.previous_alt_error = error_z.copy()

        self.prev_K_x = self.K_x.copy()
        self.prev_K_y = self.K_y.copy()
        self.prev_K_alt = self.K_alt.copy()

        self.previous_time = self.current_time

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

    def generate_rules(self, error_ante, d_error_ante, out_mem, table=None):
        rules = []
        for i, e_label in enumerate(self.flc_labels):
            for j, de_label in enumerate(self.flc_labels):
                out_table = table[i][j]
                rule = ctrl.Rule(error_ante[e_label] & d_error_ante[de_label],out_mem[out_table])
                rules.append(rule)
        return rules

    def _build_flc_pose(self):
        error_min = -1.0
        error_max = 1.0

        k_min = -0.25
        k_max = 0.25
        # ANTECEDANT
        pose_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'pose_error')
        d_pose_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_pose_error')

        # CONSEQUENT
        K_pose = ctrl.Consequent(np.linspace(k_min, k_max, 200), 'K_pose')

        # MEMBERSHIP FUNCTION
        pose_error_ante['NB'] = fuzz.trimf(pose_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        pose_error_ante['NS'] = fuzz.trimf(pose_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        pose_error_ante['Z'] = fuzz.trimf(pose_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        pose_error_ante['PS'] = fuzz.trimf(pose_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        pose_error_ante['PB'] = fuzz.trimf(pose_error_ante.universe, [(1/2)*error_max, error_max, error_max])

        d_pose_error_ante['NB'] = fuzz.trimf(d_pose_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        d_pose_error_ante['NS'] = fuzz.trimf(d_pose_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        d_pose_error_ante['Z'] = fuzz.trimf(d_pose_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        d_pose_error_ante['PS'] = fuzz.trimf(d_pose_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        d_pose_error_ante['PB'] = fuzz.trimf(d_pose_error_ante.universe, [(1/2)*error_max, error_max, error_max])

        K_pose['NB'] = fuzz.trimf(K_pose.universe, [k_min, k_min, (1/2)*k_min])
        K_pose['NS'] = fuzz.trimf(K_pose.universe, [k_min, (1/2)*k_min, 0.0])
        K_pose['Z'] = fuzz.trimf(K_pose.universe, [(1/2)*k_min, 0.0, (1/2)*k_max])
        K_pose['PS'] = fuzz.trimf(K_pose.universe, [0.0, (1/2)*k_max, k_max])
        K_pose['PB'] = fuzz.trimf(K_pose.universe, [(1/2)*k_max, k_max, k_max])

        rules_pose = self.generate_rules(pose_error_ante, d_pose_error_ante, K_pose, self.rule_table)
        self.pose_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_pose))
        return self.pose_sim
    
    def _build_flc_alt(self):
        error_min = -1.0
        error_max = 1.0

        k_min = -0.25
        k_max = 0.25
        # ANTECEDANT
        alt_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'alt_error')
        d_alt_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_alt_error')

        # CONSEQUENT
        K_alt = ctrl.Consequent(np.linspace(k_min, k_max, 200), 'K_alt')

        # MEMBERSHIP FUNCTION
        alt_error_ante['NB'] = fuzz.trimf(alt_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        alt_error_ante['NS'] = fuzz.trimf(alt_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        alt_error_ante['Z'] = fuzz.trimf(alt_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        alt_error_ante['PS'] = fuzz.trimf(alt_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        alt_error_ante['PB'] = fuzz.trimf(alt_error_ante.universe, [(1/2)*error_max, error_max, error_max])
                                          
        d_alt_error_ante['NB'] = fuzz.trimf(d_alt_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        d_alt_error_ante['NS'] = fuzz.trimf(d_alt_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        d_alt_error_ante['Z'] = fuzz.trimf(d_alt_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        d_alt_error_ante['PS'] = fuzz.trimf(d_alt_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        d_alt_error_ante['PB'] = fuzz.trimf(d_alt_error_ante.universe, [(1/2)*error_max, error_max, error_max])

        K_alt['NB'] = fuzz.trimf(K_alt.universe, [k_min, k_min, (1/2)*k_min])
        K_alt['NS'] = fuzz.trimf(K_alt.universe, [k_min, (1/2)*k_min, 0.0])
        K_alt['Z'] = fuzz.trimf(K_alt.universe, [(1/2)*k_min, 0.0, (1/2)*k_max])
        K_alt['PS'] = fuzz.trimf(K_alt.universe, [0.0, (1/2)*k_max, k_max])
        K_alt['PB'] = fuzz.trimf(K_alt.universe, [(1/2)*k_max, k_max, k_max])

        rules_alt = self.generate_rules(alt_error_ante, d_alt_error_ante, K_alt, self.rule_table)
        self.alt_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_alt))
        return self.alt_sim
    
    def _run_flc_pose(self, err_pose, d_err_pose):
        self.pose_sim.input['pose_error'] = np.clip(err_pose, -1.0, 1.0)
        self.pose_sim.input['d_pose_error'] = np.clip(d_err_pose, -1.0, 1.0)
        self.pose_sim.compute()

        K_pose = self.pose_sim.output['K_pose']
        return K_pose
    
    def _run_flc_alt(self, err, d_err):
        self.alt_sim.input['alt_error'] = np.clip(err, -1.0, 1.0)
        self.alt_sim.input['d_alt_error'] = np.clip(d_err, -1.0, 1.0)
        self.alt_sim.compute()

        K_alt = self.alt_sim.output['K_alt']
        return K_alt

if __name__ == "__main__":
    rospy.init_node("flc_hexacopter")
    FuzzyLogicControl()
    rospy.spin()