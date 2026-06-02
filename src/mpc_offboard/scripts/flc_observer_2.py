#!/home/sundy/venv_hexacopter/bin/python
import rospy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

from collections import deque
import numpy as np

class OscillationDetector:
    def __init__(self, window=20):
        self.history = deque(maxlen=window)   # fixed-size sliding window
        self.eps = 1e-3

    def update(self, error):
        self.history.append(error)            # call EVERY step
        
        # Need minimum data to compute
        if len(self.history) < 5:
            return 0.0
        
        arr = np.array(self.history)
        
        sign_changes = np.sum(np.diff(np.sign(arr)) != 0)
        freq_score = sign_changes / len(arr)
        
        variance = np.var(arr)
        mean_mag = abs(np.mean(arr))
        spread_score = variance / (mean_mag + self.eps)
        
        osc_score = freq_score * np.tanh(spread_score)
        return np.clip(osc_score, 0.0, 1.0)

class FuzzyLogicControl:
    ante_labels = ['Z','VS','S', 'M', 'B']
    rule_table = np.array([
        ['Z', 'Z', 'Z', 'NS', 'NB'],
        ['PS', 'Z', 'Z', 'NS', 'NB'],
        ['PS', 'PS', 'Z', 'NS', 'NB'],
        ['PB', 'PS', 'PS', 'NS', 'NB'],
        ['PB', 'PB', 'PS', 'NS', 'NB']
        ])

    error_min = 0.0
    error_max = 1.0
    osc_min = 0.0
    osc_max = 1.0

    del_K_min = -0.4
    del_K_max = 0.4

    def __init__(self):
        self.osce_detect = OscillationDetector(window=20)
        self.position_desired = np.zeros(3)
        self.position_actual = np.zeros(3)

        # PUBLISH
        self.dist_gain_pub = rospy.Publisher('/flc/disturbance_gain', Float32MultiArray, queue_size=10)

        # SUBSCRIBE
        self.pose_actual_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_actual_callback, queue_size=10)
        self.pose_desired_sub = rospy.Subscriber("/trajectory/ref_pose", PoseStamped, self.pose_desired_callback, queue_size=10)
        self.velo_actual_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_actual_callback,queue_size=10)
        self.velo_desired_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_desired_callback,queue_size=10)

    def pose_actual_callback(self, msg:PoseStamped):
        self.position_actual = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.orientation_actual_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_actual_euler = euler_from_quaternion(self.orientation_actual_quat)

    def pose_desired_callback(self, msg:PoseStamped):
        self.position_desired = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.orientation_desired_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_desired_euler = euler_from_quaternion(self.orientation_desired_quat)

    def velo_actual_callback(self, msg:TwistStamped):
        self.velocity_actual = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def velo_desired_callback(self, msg:TwistStamped):
        self.velocity_desired = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])

    def generate_rules(self, ante_1, ante_2, out_mem, table=None):
        rules = []
        for i, label_1 in enumerate(self.ante_labels):
            for j, label_2 in enumerate(self.ante_labels):
                out_table = table[i][j]
                rule = ctrl.Rule(ante_1[label_1] & ante_2[label_2],out_mem[out_table])
                rules.append(rule)
        return rules

    def flc_orient(self):
        # Antecedent
        error = ctrl.Antecedent(np.linspace(self.error_min, self.error_max, 200), 'error')
        oscil = ctrl.Antecedent(np.linspace(self.osc_min, self.osc_max, 200), 'oscil')

        # Consequent
        del_K_orient= ctrl.Consequent(np.linspace(self.del_K_min, self.del_K_max, 200), 'del_K')

        # Membership function
        error['Z'] = fuzz.trimf(error.universe, [self.error_min, self.error_min, self.error_max*0.25])
        error['VS'] = fuzz.trimf(error.universe, [self.error_min, self.error_max*0.25, self.error_max*0.5])
        error['S'] = fuzz.trimf(error.universe, [self.error_max*0.25, self.error_max*0.5, self.error_max*0.75])
        error['M'] = fuzz.trimf(error.universe, [self.error_max*0.5, self.error_max*0.75, self.error_max])
        error['B'] = fuzz.trimf(error.universe, [self.error_max*0.75, self.error_max, self.error_max])

        oscil['Z'] = fuzz.trimf(oscil.universe, [self.osc_min, self.osc_min, self.osc_max*0.1])
        oscil['VS'] = fuzz.trimf(oscil.universe, [self.osc_min, self.osc_max*0.1, self.osc_max*0.3])
        oscil['S'] = fuzz.trimf(oscil.universe, [self.osc_max*0.1, self.osc_max*0.3, self.osc_max*0.5])
        oscil['M'] = fuzz.trimf(oscil.universe, [self.osc_max*0.3, self.osc_max*0.5, self.osc_max*0.7])
        oscil['B'] = fuzz.trapmf(oscil.universe, [self.osc_max*0.5, self.osc_max*0.7, self.osc_max, self.osc_max])

        del_K_orient['NB'] = fuzz.trimf(del_K_orient.universe, [self.del_K_min, self.del_K_min, self.del_K_min*0.5])
        del_K_orient['NS'] = fuzz.trimf(del_K_orient.universe, [self.del_K_min, self.del_K_min*0.5, 0])
        del_K_orient['Z'] = fuzz.trimf(del_K_orient.universe, [self.del_K_min*0.5, 0, self.del_K_max*0.5])
        del_K_orient['PS'] = fuzz.trimf(del_K_orient.universe, [0, self.del_K_max*0.5, self.del_K_max])
        del_K_orient['PB'] = fuzz.trimf(del_K_orient.universe, [self.del_K_max*0.5, self.del_K_max, self.del_K_max])

        rules_orient = self.generate_rules(error, oscil, del_K_orient, self.rule_table)
        self.orient_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_orient))
        return self.orient_sim
    
    def flc_alt(self):
        # Antecedent
        error = ctrl.Antecedent(np.linspace(self.error_min, self.error_max, 200), 'error')
        oscil = ctrl.Antecedent(np.linspace(self.osc_min, self.osc_max, 200), 'oscil')

        # Consequent
        del_K_alt = ctrl.Consequent(np.linspace(self.del_K_min, self.del_K_max, 200), 'del_K_alt')

        # Membership function
        error['Z'] = fuzz.trimf(error.universe, [self.error_min, self.error_min, self.error_max*0.25])
        error['VS'] = fuzz.trimf(error.universe, [self.error_min, self.error_max*0.25, self.error_max*0.5])
        error['S'] = fuzz.trimf(error.universe, [self.error_max*0.25, self.error_max*0.5, self.error_max*0.75])
        error['M'] = fuzz.trimf(error.universe, [self.error_max*0.5, self.error_max*0.75, self.error_max])
        error['B'] = fuzz.trimf(error.universe, [self.error_max*0.75, self.error_max, self.error_max])

        oscil['Z'] = fuzz.trimf(oscil.universe, [self.osc_min, self.osc_min, self.osc_max*0.1])
        oscil['VS'] = fuzz.trimf(oscil.universe, [self.osc_min, self.osc_max*0.1, self.osc_max*0.3])
        oscil['S'] = fuzz.trimf(oscil.universe, [self.osc_max*0.1, self.osc_max*0.3, self.osc_max*0.5])
        oscil['M'] = fuzz.trimf(oscil.universe, [self.osc_max*0.3, self.osc_max*0.5, self.osc_max*0.7])
        oscil['B'] = fuzz.trapmf(oscil.universe, [self.osc_max*0.5, self.osc_max*0.7, self.osc_max, self.osc_max])

        del_K_alt['NB'] = fuzz.trimf(del_K_alt.universe, [self.del_K_min, self.del_K_min, self.del_K_min*0.5])
        del_K_alt['NS'] = fuzz.trimf(del_K_alt.universe, [self.del_K_min, self.del_K_min*0.5, 0])
        del_K_alt['Z'] = fuzz.trimf(del_K_alt.universe, [self.del_K_min*0.5, 0, self.del_K_max*0.5])
        del_K_alt['PS'] = fuzz.trimf(del_K_alt.universe, [0, self.del_K_max*0.5, self.del_K_max])
        del_K_alt['PB'] = fuzz.trimf(del_K_alt.universe, [self.del_K_max*0.5, self.del_K_max, self.del_K_max])

        rules_alt = self.generate_rules(error, oscil, del_K_alt, self.rule_table)
        self.alt_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_alt))
        return self.alt_sim
    
    def run_flc(self, err_orient, oscil_orient, err_alt, oscil_alt):
        self.orient_sim.input['pose_error'] = np.clip(err_orient, 0.0, 1.0)
        self.orient_sim.input['d_pose_error'] = np.clip(oscil_orient, 0.0, 1.0)
        self.orient_sim.compute()

        self.alt_sim.input['alt_error'] = np.clip(err_alt, 0.0, 1.0)
        self.alt_sim.input['d_alt_error'] = np.clip(oscil_alt, 0.0, 1.0)
        self.alt_sim.compute()

        del_K_pose = self.orient_sim.output['K_pose']
        del_K_alt = self.alt_sim.output['K_alt']
        return del_K_pose, del_K_alt
    
    def compute_flc(self):
        error_roll = self.orientation_desired_euler[0] - self.orientation_actual_euler[0]
        error_pitch = self.orientation_desired_euler[1] - self.orientation_actual_euler[1]
        
        error_x = abs(self.position_desired[0] - self.position_actual[0])
        error_y = abs(self.position_desired[1] - self.position_actual[1])
        error_alt = self.position_desired[2] - self.position_actual[2]

if __name__ == "__main__":
    rospy.init_node("flc_hexacopter")
    FuzzyLogicControl()
    rospy.spin()

# flc_labels = ['VS','S','M', 'B', 'VB']
#     rule_table = np.array([
#         ['PB', 'PB', 'PB', 'PM', 'PS'],
#         ['PB', 'PB', 'PM', 'PS', 'Z'],
#         ['PM', 'PS', 'Z', 'PS', 'PM'],
#         ['Z', 'PS', 'PM', 'PB', 'PB'],
#         ['PS', 'PM', 'PB', 'PB', 'PB']
#         ])

# def _build_flc_alt(self):
#         error_min = 0.0
#         error_max = 1.0

#         k_min = -0.1
#         k_max = 0.1

#         # ANTECEDANT
#         alt_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'alt_error')
#         alt_avg_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_alt_error')

#         # CONSEQUENT
#         K_alt = ctrl.Consequent(np.linspace(k_min, k_max, 200), 'K_alt')

#         # MEMBERSHIP FUNCTION
#         alt_error_ante['VS'] = fuzz.trimf(alt_error_ante.universe, [error_min, error_min, error_max/4])
#         alt_error_ante['S'] = fuzz.trimf(alt_error_ante.universe, [error_min, error_max/4, (2/4)*error_max])
#         alt_error_ante['M'] = fuzz.trimf(alt_error_ante.universe, [error_max/4, (2/4)*error_max, (3/4)*error_max])
#         alt_error_ante['B'] = fuzz.trimf(alt_error_ante.universe, [(2/4)*error_max, (3/4)*error_max, error_max])
#         alt_error_ante['VB'] = fuzz.trimf(alt_error_ante.universe, [(3/4)*error_max, error_max, error_max])
                                          
#         alt_avg_error_ante['VS'] = fuzz.trimf(alt_avg_error_ante.universe, [error_min, error_min, error_max/4])
#         alt_avg_error_ante['S'] = fuzz.trimf(alt_avg_error_ante.universe, [error_min, error_max/4, (2/4)*error_max])
#         alt_avg_error_ante['M'] = fuzz.trimf(alt_avg_error_ante.universe, [error_max/4, (2/4)*error_max, (3/4)*error_max])
#         alt_avg_error_ante['B'] = fuzz.trimf(alt_avg_error_ante.universe, [(2/4)*error_max, (3/4)*error_max, error_max])
#         alt_avg_error_ante['VB'] = fuzz.trimf(alt_avg_error_ante.universe, [(3/4)*error_max, error_max, error_max])

#         K_alt['Z'] = fuzz.trapmf(K_alt.universe, [k_min, k_min, 0.15, 0.3])
#         K_alt['PS'] = fuzz.trimf(K_alt.universe, [0.3, 0.3, k_max/2])
#         K_alt['PM'] = fuzz.trimf(K_alt.universe, [0.3, k_max/2, k_max])
#         K_alt['PB'] = fuzz.trimf(K_alt.universe, [k_max/2, k_max, k_max])

#         rules_alt = self.generate_rules(alt_error_ante, alt_avg_error_ante, K_alt, self.rule_table)
#         self.alt_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_alt))
#         return self.alt_sim