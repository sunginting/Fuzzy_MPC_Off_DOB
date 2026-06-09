#!/home/sundy/venv_hexacopter/bin/python
import rospy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

class FuzzyLogicControl_Position:
    flc_labels = ['NB','NS','Z', 'PS', 'PB']
    rule_table = np.array([
        ['PB', 'PB', 'NB', 'PB', 'PB'],
        ['PB', 'PS', 'NS', 'PS', 'PB'],
        ['PB', 'PS', 'Z',  'PS', 'PB'],
        ['PB', 'PS', 'NS', 'PS', 'PB'],
        ['PB', 'PB', 'NB', 'PB', 'PB']
        ])
    
    alt_table = np.array([
        ['PB', 'PB', 'NB', 'PB', 'PB'],
        ['PB', 'PS', 'NS', 'PS', 'PB'],
        ['PB', 'PS', 'Z',  'PS', 'PB'],
        ['PB', 'PS', 'NS', 'PS', 'PB'],
        ['PB', 'PB', 'NB', 'PB', 'PB']
        ])
    
    q_x_max = 70
    q_x_min = 35

    q_y_max = 70
    q_y_min = 35

    q_alt_max = 120
    q_alt_min = 100

    q_velo_max = 30
    q_velo_min = 20

    q_vz_max = 20
    q_vz_min = 10

    r_x_max = 0.2
    r_x_min = 0.05

    r_y_max = 0.2
    r_y_min = 0.05

    r_alt_max = 0.06
    r_alt_min = 0.04

    def __init__(self):
        # Inisialisasi variabel untuk menghindari altributeError
        self.position_desired = np.zeros(3)
        self.velocity_desired = np.zeros(3)
        self.position_actual = np.zeros(3)
        self.velocity_actual = np.zeros(3)

        # Inisialisasi output weights
        self.q = [60.0, 60.0, 100.0, 20.0, 20.0, 12.0]
        self.r = [0.2, 0.2, 0.06]

        self.prev_x_error_mag = 0.0
        self.prev_y_error_mag = 0.0
        self.prev_alt_error_mag = 0.0
        self.prev_vx_error_mag = 0.0
        self.prev_vy_error_mag = 0.0
        self.prev_vz_error_mag = 0.0

        self.prev_d_x = 0.0
        self.prev_d_y = 0.0
        self.prev_d_alt = 0.0
        self.prev_d_vx = 0.0
        self.prev_d_vy = 0.0
        self.prev_d_vz = 0.0

        self.pose_label = ['pose_error', 'd_pose_error', 'del_d_pose']
        self.alt_label = ['alt_error', 'd_alt_error', 'del_d_alt']
        self.velo_label = ['velo_error', 'd_velo_error', 'del_d_velo']
        self.vz_label = ['vz_error', 'd_vz_error', 'del_d_vz']

        self.pose_sim = self._build_flc_pose()
        self.alt_sim = self._build_flc_alt()
        self.velo_sim = self._build_flc_velo()
        self.vz_sim = self._build_flc_vz()

        self.previous_time = 0.0
        
        # PUBLISH
        self.weights_MPC_pub = rospy.Publisher('/flc/mpc_weights', Float32MultiArray, queue_size=10)
        self.error_avg_pub = rospy.Publisher('/flc/error_avg', Float32MultiArray, queue_size=10)

        # SUBSCRIBE
        self.pose_actual_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_actual_callback, queue_size=10)
        self.velo_actual_sub = rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.velo_actual_callback, queue_size=10)
        self.pose_desired_sub = rospy.Subscriber("/trajectory/ref_pose", PoseStamped, self.pose_desired_callback, queue_size=10)
        self.velo_desired_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_desired_callback, queue_size=10)

    def pose_actual_callback(self, msg:PoseStamped):
        # ENU -> NED
        self.position_actual = np.array([msg.pose.position.y, msg.pose.position.x, -msg.pose.position.z])
        self.orientation_actual_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_actual_euler = euler_from_quaternion(self.orientation_actual_quat)

        self.current_time = rospy.Time.now().to_sec()
        if self.previous_time == 0.0:
            self.previous_time = self.current_time
            return

        x_error = self.position_desired[0]-self.position_actual[0]
        y_error = self.position_desired[1]-self.position_actual[1]
        z_error = self.position_desired[2]-self.position_actual[2]

        vx_error = self.velocity_desired[0] - self.velocity_actual[0]
        vy_error = self.velocity_desired[1] - self.velocity_actual[1]
        vz_error = self.velocity_desired[2] - self.velocity_actual[2]

        x_error_norm = np.clip(x_error/5.0, -1.0, 1.0)
        y_error_norm = np.clip(y_error/5.0, -1.0, 1.0)
        alt_error_norm = np.clip(z_error/0.3, -1.0, 1.0)
        vx_error_norm = np.clip(vx_error/0.75, -1.0, 1.0)
        vy_error_norm = np.clip(vy_error/0.75, -1.0, 1.0)
        vz_error_norm = np.clip(vz_error/0.5, -1.0, 1.0)

        dt = self.current_time - self.previous_time
        if dt <= 0:
            return
        else:
            d_x_error = (x_error - self.prev_x_error_mag) / dt
            d_y_error = (y_error - self.prev_y_error_mag) / dt
            d_alt_error = (z_error - self.prev_alt_error_mag)/dt
            d_vx_error = (vx_error - self.prev_vx_error_mag)/dt
            d_vy_error = (vy_error - self.prev_vy_error_mag)/dt
            d_vz_error = (vz_error - self.prev_vz_error_mag)/dt

        d_x_error_norm = np.clip(d_x_error/1, -1.0, 1.0)
        d_y_error_norm = np.clip(d_y_error/1, -1.0, 1.0)
        d_alt_error_norm = np.clip(d_alt_error/0.5, -1.0, 1.0)
        d_vx_error_norm = np.clip(d_vx_error/0.5, -1.0, 1.0)
        d_vy_error_norm = np.clip(d_vy_error/0.5, -1.0, 1.0)
        d_vz_error_norm = np.clip(d_vz_error/0.2, -1.0, 1.0)

        del_d_x = self.run_flc(x_error_norm, d_x_error_norm, self.pose_sim, self.pose_label)
        del_d_y = self.run_flc(y_error_norm, d_y_error_norm, self.pose_sim, self.pose_label)
        del_d_alt = self.run_flc(alt_error_norm, d_alt_error_norm, self.alt_sim, self.alt_label)
        del_d_vx = self.run_flc(vx_error_norm, d_vx_error_norm, self.velo_sim, self.velo_label)
        del_d_vy = self.run_flc(vy_error_norm, d_vy_error_norm, self.velo_sim, self.velo_label)
        del_d_vz = self.run_flc(vz_error_norm, d_vz_error_norm, self.vz_sim, self.vz_label)

        d_x = np.clip(self.prev_d_x + del_d_x, 0.0, 1.0)
        d_y = np.clip(self.prev_d_y + del_d_y, 0.0, 1.0)
        d_alt  = np.clip(self.prev_d_alt  + del_d_alt,  0.0, 1.0)
        d_vx = np.clip(self.prev_d_vx + del_d_vx, 0.0, 1.0)
        d_vy = np.clip(self.prev_d_vy + del_d_vy, 0.0, 1.0)
        d_vz   = np.clip(self.prev_d_vz   + del_d_vz,   0.0, 1.0)

        self.q_x = self.q_x_min + d_x*(self.q_x_max - self.q_x_min)
        self.q_y = self.q_y_min + d_y*(self.q_y_max - self.q_y_min)
        self.q_alt = self.q_alt_min + d_alt*(self.q_alt_max - self.q_alt_min)
        self.q_vx = self.q_velo_min + d_vx*(self.q_velo_max - self.q_velo_min)
        self.q_vy = self.q_velo_min + d_vy*(self.q_velo_max - self.q_velo_min)
        self.q_vz = self.q_vz_min + d_vz*(self.q_vz_max - self.q_vz_min)
        self.r_x = self.r_x_max - d_x*(self.r_x_max - self.r_x_min)
        self.r_y = self.r_y_max - d_y*(self.r_y_max - self.r_y_min)
        self.r_alt = self.r_alt_max - (d_alt/10)*(self.r_alt_max - self.r_alt_min)

        self.prev_x_error_mag = x_error.copy()
        self.prev_y_error_mag = y_error.copy()
        self.prev_alt_error_mag = z_error.copy()
        self.prev_vx_error_mag = vx_error.copy()
        self.prev_vy_error_mag = vy_error.copy()
        self.prev_vz_error_mag = vz_error.copy()

        self.prev_d_x = d_x.copy()
        self.prev_d_y = d_y.copy()
        self.prev_d_alt = d_alt.copy()
        self.prev_d_vx = d_vx.copy()
        self.prev_d_vy = d_vy.copy()
        self.prev_d_vz = d_vz.copy()

        self.previous_time = self.current_time

        weights_msg = Float32MultiArray()
        weights_msg.data = [self.q_x,self.q_y,self.q_alt,
                            self.q_vx,self.q_vy,self.q_vz,
                            self.r_x,self.r_y,self.r_alt]
        self.weights_MPC_pub.publish(weights_msg)

    def velo_actual_callback(self, msg:TwistStamped):
        # ENU -> NED
        self.velocity_actual = np.array([msg.twist.linear.y, msg.twist.linear.x, -msg.twist.linear.z])
        self.angular_actual = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def pose_desired_callback(self, msg:PoseStamped):
        # ENU -> NED
        self.position_desired = np.array([msg.pose.position.y, msg.pose.position.x, -msg.pose.position.z])
        self.orientation_desired_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_desired_euler = euler_from_quaternion(self.orientation_desired_quat)

    def velo_desired_callback(self, msg:TwistStamped):
        # ENU -> NED
        self.velocity_desired = np.array([msg.twist.linear.y, msg.twist.linear.x, -msg.twist.linear.z])
        self.angular_desired = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def generate_rules(self, error_ante, error_avg_ante, out_mem,table):
        rules = []
        for i, e_label in enumerate(self.flc_labels):
            for j, de_label in enumerate(self.flc_labels):
                out_table = table[i][j]
                
                rule = ctrl.Rule(error_ante[e_label] & error_avg_ante[de_label],out_mem[out_table])
                rules.append(rule)
        return rules

    def _build_flc_pose(self):
        # ANTECEDANT
        error_min = -1.0
        error_max = 1.0
        d_min = -0.25
        d_max = 0.25
        pose_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'pose_error')
        d_pose_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_pose_error')

        # CONSEQUENT
        del_d_pose = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'del_d_pose')

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

        del_d_pose['NB'] = fuzz.trimf(del_d_pose.universe, [d_min, d_min, (1/2)*d_min])
        del_d_pose['NS'] = fuzz.trimf(del_d_pose.universe, [d_min, (1/2)*d_min, 0])
        del_d_pose['Z'] = fuzz.trimf(del_d_pose.universe, [(1/2)*d_min, 0, (1/2)*d_max])
        del_d_pose['PS'] = fuzz.trimf(del_d_pose.universe, [ 0, (1/2)*d_max, d_max])
        del_d_pose['PB'] = fuzz.trimf(del_d_pose.universe, [ (1/2)*d_min, d_max, d_max])

        rules_pose = self.generate_rules(pose_error_ante, d_pose_error_ante, del_d_pose, self.rule_table)
        self.pose_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_pose))
        return self.pose_sim
    
    def _build_flc_alt(self):
        error_min = -1.0
        error_max = 1.0
        d_min = -0.25
        d_max = 0.25
        alt_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'alt_error')
        d_alt_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_alt_error')

        # CONSEQUENT
        del_d_alt = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'del_d_alt')

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

        del_d_alt['NB'] = fuzz.trimf(del_d_alt.universe, [d_min, d_min, (1/2)*d_min])
        del_d_alt['NS'] = fuzz.trimf(del_d_alt.universe, [d_min, (1/2)*d_min, 0])
        del_d_alt['Z'] = fuzz.trimf(del_d_alt.universe, [(1/2)*d_min, 0, (1/2)*d_max])
        del_d_alt['PS'] = fuzz.trimf(del_d_alt.universe, [ 0, (1/2)*d_max, d_max])
        del_d_alt['PB'] = fuzz.trimf(del_d_alt.universe, [ (1/2)*d_min, d_max, d_max])

        rules_alt = self.generate_rules(alt_error_ante, d_alt_error_ante, del_d_alt, self.alt_table)
        self.alt_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_alt))
        return self.alt_sim

    def _build_flc_velo(self):
        error_min = -1.0
        error_max = 1.0
        d_min = -0.25
        d_max = 0.25
        # ANTECEDANT
        velo_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'velo_error')
        d_velo_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_velo_error')

        # CONSEQUENT
        del_d_velo = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'del_d_velo')

        # MEMBERSHIP FUNCTION
        velo_error_ante['NB'] = fuzz.trimf(velo_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        velo_error_ante['NS'] = fuzz.trimf(velo_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        velo_error_ante['Z'] = fuzz.trimf(velo_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        velo_error_ante['PS'] = fuzz.trimf(velo_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        velo_error_ante['PB'] = fuzz.trimf(velo_error_ante.universe, [(1/2)*error_max, error_max, error_max])
                                          
        d_velo_error_ante['NB'] = fuzz.trimf(d_velo_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        d_velo_error_ante['NS'] = fuzz.trimf(d_velo_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        d_velo_error_ante['Z'] = fuzz.trimf(d_velo_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        d_velo_error_ante['PS'] = fuzz.trimf(d_velo_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        d_velo_error_ante['PB'] = fuzz.trimf(d_velo_error_ante.universe, [(1/2)*error_max, error_max, error_max])

        del_d_velo['NB'] = fuzz.trimf(del_d_velo.universe, [d_min, d_min, (1/2)*d_min])
        del_d_velo['NS'] = fuzz.trimf(del_d_velo.universe, [d_min, (1/2)*d_min, 0])
        del_d_velo['Z'] = fuzz.trimf(del_d_velo.universe, [(1/2)*d_min, 0, (1/2)*d_max])
        del_d_velo['PS'] = fuzz.trimf(del_d_velo.universe, [ 0, (1/2)*d_max, d_max])
        del_d_velo['PB'] = fuzz.trimf(del_d_velo.universe, [ (1/2)*d_min, d_max, d_max])

        rules_velo = self.generate_rules(velo_error_ante, d_velo_error_ante, del_d_velo, self.rule_table)
        self.velo_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_velo))
        return self.velo_sim
    
    def _build_flc_vz(self):
        error_min = -1.0
        error_max = 1.0
        d_min = -0.25
        d_max = 0.25
        # ANTECEDANT
        vz_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'vz_error')
        d_vz_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'd_vz_error')

        # CONSEQUENT
        del_d_vz = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'del_d_vz')

        # MEMBERSHIP FUNCTION
        vz_error_ante['NB'] = fuzz.trimf(vz_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        vz_error_ante['NS'] = fuzz.trimf(vz_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        vz_error_ante['Z'] = fuzz.trimf(vz_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        vz_error_ante['PS'] = fuzz.trimf(vz_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        vz_error_ante['PB'] = fuzz.trimf(vz_error_ante.universe, [(1/2)*error_max, error_max, error_max])
                                          
        d_vz_error_ante['NB'] = fuzz.trimf(d_vz_error_ante.universe, [error_min, error_min, (1/2)*error_min])
        d_vz_error_ante['NS'] = fuzz.trimf(d_vz_error_ante.universe, [error_min, (1/2)*error_min, 0.0])
        d_vz_error_ante['Z'] = fuzz.trimf(d_vz_error_ante.universe, [(1/2)*error_min, 0.0, (1/2)*error_max])
        d_vz_error_ante['PS'] = fuzz.trimf(d_vz_error_ante.universe, [0.0, (1/2)*error_max, error_max])
        d_vz_error_ante['PB'] = fuzz.trimf(d_vz_error_ante.universe, [(1/2)*error_max, error_max, error_max])

        del_d_vz['NB'] = fuzz.trimf(del_d_vz.universe, [d_min, d_min, (1/2)*d_min])
        del_d_vz['NS'] = fuzz.trimf(del_d_vz.universe, [d_min, (1/2)*d_min, 0])
        del_d_vz['Z'] = fuzz.trimf(del_d_vz.universe, [(1/2)*d_min, 0, (1/2)*d_max])
        del_d_vz['PS'] = fuzz.trimf(del_d_vz.universe, [ 0, (1/2)*d_max, d_max])
        del_d_vz['PB'] = fuzz.trimf(del_d_vz.universe, [ (1/2)*d_min, d_max, d_max])

        rules_vz = self.generate_rules(vz_error_ante, d_vz_error_ante, del_d_vz, self.rule_table)
        self.vz_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_vz))
        return self.vz_sim
    
    def run_flc(self, err, d_err, sim, label):
        sim.input[label[0]] = np.clip(err, -1.0, 1.0)
        sim.input[label[1]] = np.clip(d_err, -1.0, 1.0)
        sim.compute()

        del_d = sim.output[label[2]]
        return del_d
    
if __name__ == "__main__":
    rospy.init_node("flc_hexacopter")
    FuzzyLogicControl_Position()
    rospy.spin()
