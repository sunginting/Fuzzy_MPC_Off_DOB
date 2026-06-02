#!/home/sundy/venv_hexacopter/bin/python
import rospy
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float32MultiArray
from tf.transformations import euler_from_quaternion

class FuzzyLogicControl_Position:
    flc_labels = ['S','M','L']
    D_table = np.array([
        ['VS', 'S', 'M'],
        ['S', 'M', 'L'],
        ['M', 'L', 'VL']])
    
    q_pose_max = 100
    q_pose_min = 20
    q_alt_max = 100
    q_alt_min = 20
    q_velo_max = 20
    q_velo_min = 3
    q_vz_max = 20
    q_vz_min = 3
    r_max = 1
    r_min = 0.1

    def __init__(self):
        # Inisialisasi variabel untuk menghindari altributeError
        self.position_desired = np.zeros(3)
        self.velocity_desired = np.zeros(3)
        self.position_actual = np.zeros(3)
        self.velocity_actual = np.zeros(3)

        # Inisialisasi output weights
        self.q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.r = [1.0, 1.0, 1.0]

        self.prev_pose_error_mag = 0.0
        self.prev_alt_error_mag = 0.0
        self.prev_velo_error_mag = 0.0
        self.prev_vz_error_mag = 0.0

        self.pose_buffer = []
        self.alt_buffer = []
        self.velo_buffer = []
        self.vz_buffer = []

        self.pose_buffer_norm = []
        self.alt_buffer_norm = []
        self.velo_buffer_norm = []
        self.vz_buffer_norm = []
        self.buffer_size = 50

        self.pose_sim = self._build_flc_pose()
        self.alt_sim = self._build_flc_alt()
        self.velo_sim = self._build_flc_velo()
        self.vz_sim = self._build_flc_vz()
        
        # PUBLISH
        self.weights_MPC_pub = rospy.Publisher('/flc/mpc_weights', Float32MultiArray, queue_size=10)
        self.error_avg_pub = rospy.Publisher('/flc/error_avg', Float32MultiArray, queue_size=10)

        # SUBSCRIBE
        self.pose_actual_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_actual_callback, queue_size=10)
        self.velo_actual_sub = rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.velo_actual_callback, queue_size=10)
        self.pose_desired_sub = rospy.Subscriber("/trajectory/ref_pose", PoseStamped, self.pose_desired_callback, queue_size=10)
        self.velo_desired_sub = rospy.Subscriber("/trajectory/ref_vel", TwistStamped, self.velo_desired_callback, queue_size=10)

    def pose_actual_callback(self, msg:PoseStamped):
        self.position_actual = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.orientation_actual_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_actual_euler = euler_from_quaternion(self.orientation_actual_quat)

        error_x = self.position_desired[0]-self.position_actual[0]
        error_y = self.position_desired[1]-self.position_actual[1]
        error_z = self.position_desired[2]-self.position_actual[2]
        raw_pose_error_mag = np.sqrt((error_x**2 + error_y**2))
        raw_alt_error_mag = np.sqrt((error_z**2))

        error_vx = self.velocity_desired[0] - self.velocity_actual[0]
        error_vy = self.velocity_desired[1] - self.velocity_actual[1]
        error_vz = self.velocity_desired[2] - self.velocity_actual[2]
        raw_velo_error_mag = np.sqrt((error_vx**2 + error_vy**2))
        raw_vz_error_mag = np.sqrt((error_vz**2))

        pose_error_mag = self.lpf(raw_pose_error_mag,self.prev_pose_error_mag)
        alt_error_mag = self.lpf(raw_alt_error_mag, self.prev_alt_error_mag)
        velo_error_mag = self.lpf(raw_velo_error_mag, self.prev_velo_error_mag)
        vz_error_mag = self.lpf(raw_vz_error_mag, self.prev_vz_error_mag)

        self.prev_pose_error_mag = pose_error_mag.copy()
        self.prev_alt_error_mag = alt_error_mag.copy()
        self.prev_velo_error_mag = velo_error_mag.copy()
        self.prev_vz_error_mag = vz_error_mag.copy()

        pose_error_norm = np.clip(pose_error_mag/3, 0.0, 1.0)
        alt_error_norm = np.clip(alt_error_mag/0.15, 0.0, 1.0)
        velo_error_norm = np.clip(velo_error_mag/2, 0.0, 1.0)
        vz_error_norm = np.clip(vz_error_mag/0.5, 0.0, 1.0)

        # DATA LOGGING PURPOSES
        self.pose_buffer.append(pose_error_mag)
        if len(self.pose_buffer) > self.buffer_size:
            self.pose_buffer.pop(0)
        pose_error_avg = np.mean(self.pose_buffer) if self.pose_buffer else 0.0

        self.alt_buffer.append(alt_error_mag)
        if len(self.alt_buffer) > self.buffer_size:
            self.alt_buffer.pop(0)
        alt_error_avg = np.mean(self.alt_buffer) if self.alt_buffer else 0.0

        self.velo_buffer.append(velo_error_mag)
        if len(self.velo_buffer) > self.buffer_size:
            self.velo_buffer.pop(0)
        velo_error_avg = np.mean(self.velo_buffer) if self.velo_buffer else 0.0

        self.vz_buffer.append(vz_error_mag)
        if len(self.vz_buffer) > self.buffer_size:
            self.vz_buffer.pop(0)
        vz_error_avg = np.mean(self.vz_buffer) if self.vz_buffer else 0.0

        error_avg_msg = Float32MultiArray()
        error_avg_msg.data = [pose_error_avg, alt_error_avg, velo_error_avg, vz_error_avg]
        self.error_avg_pub.publish(error_avg_msg)

        # ACTUALLY USED
        self.pose_buffer_norm.append(pose_error_norm)
        if len(self.pose_buffer_norm) > self.buffer_size:
            self.pose_buffer_norm.pop(0)
        pose_error_avg_norm = np.mean(self.pose_buffer_norm) if self.pose_buffer_norm else 0.0

        self.alt_buffer_norm.append(alt_error_norm)
        if len(self.alt_buffer_norm) > self.buffer_size:
            self.alt_buffer_norm.pop(0)
        alt_error_avg_norm = np.mean(self.alt_buffer_norm) if self.alt_buffer_norm else 0.0

        self.velo_buffer_norm.append(velo_error_norm)
        if len(self.velo_buffer_norm) > self.buffer_size:
            self.velo_buffer_norm.pop(0)
        velo_error_avg_norm = np.mean(self.velo_buffer_norm) if self.velo_buffer_norm else 0.0

        self.vz_buffer_norm.append(vz_error_norm)
        if len(self.vz_buffer_norm) > self.buffer_size:
            self.vz_buffer_norm.pop(0)
        vz_error_avg_norm = np.mean(self.vz_buffer_norm) if self.vz_buffer_norm else 0.0

        d_pose, d_alt, d_velo, d_vz = self._run_flc(pose_error_norm, alt_error_norm, velo_error_norm, vz_error_norm, pose_error_avg_norm, alt_error_avg_norm, velo_error_avg_norm, vz_error_avg_norm)

        self.q_pose = self.q_pose_min + d_pose*(self.q_pose_max - self.q_pose_min)
        self.q_alt = self.q_alt_min + d_alt*(self.q_alt_max - self.q_alt_min)
        self.q_velo = self.q_velo_min + d_velo*(self.q_velo_max - self.q_velo_min)
        self.q_vz = self.q_vz_min + d_vz*(self.q_vz_max - self.q_vz_min)
        self.r_pose = self.r_max - d_pose*(self.r_max - self.r_min)
        self.r_alt = self.r_max - d_alt*(self.r_max - self.r_min)

        weights_msg = Float32MultiArray()
        weights_msg.data = [self.q_pose,self.q_pose,self.q_alt,
                            self.q_velo,self.q_velo,self.q_vz,
                            self.r_pose,self.r_pose,self.r_alt]
        self.weights_MPC_pub.publish(weights_msg)

    def velo_actual_callback(self, msg:TwistStamped):
        self.velocity_actual = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.angular_actual = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def pose_desired_callback(self, msg:PoseStamped):
        self.position_desired = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.orientation_desired_quat= np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
        self.orientation_desired_euler = euler_from_quaternion(self.orientation_desired_quat)

    def velo_desired_callback(self, msg:TwistStamped):
        self.velocity_desired = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.angular_desired = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])

    def generate_rules(self, error_ante, error_avg_ante, out_mem,table=None):
        rules = []
        for i, e_label in enumerate(self.flc_labels):
            for j, de_label in enumerate(self.flc_labels):
                out_table = table[i][j]
                
                rule = ctrl.Rule(error_ante[e_label] & error_avg_ante[de_label],out_mem[out_table])
                rules.append(rule)
        return rules

    def _build_flc_pose(self):
        # ANTECEDANT
        error_min = 0.0
        error_max = 1.0
        d_min = 0.0
        d_max = 1.0
        pose_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'pose_error')
        pose_error_avg_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'pose_error_avg')

        # CONSEQUENT
        d_pose = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'd_pose')

        # MEMBERSHIP FUNCTION
        pose_error_ante['S'] = fuzz.trimf(pose_error_ante.universe, [error_min, error_min, error_max/2])
        pose_error_ante['M'] = fuzz.trimf(pose_error_ante.universe, [error_min, error_max/2, error_max])
        pose_error_ante['L'] = fuzz.trimf(pose_error_ante.universe, [error_max/2, error_max, error_max])

        pose_error_avg_ante['S'] = fuzz.trimf(pose_error_avg_ante.universe, [error_min, error_min, error_max/2])
        pose_error_avg_ante['M'] = fuzz.trimf(pose_error_avg_ante.universe, [error_min, error_max/2, error_max])
        pose_error_avg_ante['L'] = fuzz.trimf(pose_error_avg_ante.universe, [error_max/2, error_max, error_max])

        d_pose['VS'] = fuzz.trimf(d_pose.universe, [0.0, 0.0, (1/4)*d_max])
        d_pose['S'] = fuzz.trimf(d_pose.universe, [0.0, (1/4)*d_max, (2/4)*d_max])
        d_pose['M'] = fuzz.trimf(d_pose.universe, [ (1/4)*d_max, (2/4)*d_max, (3/4)*d_max])
        d_pose['L'] = fuzz.trimf(d_pose.universe, [ (2/4)*d_max, (3/4)*d_max, (4/4)*d_max])
        d_pose['VL'] = fuzz.trimf(d_pose.universe, [ (3/4)*d_max, (4/4)*d_max, (4/4)*d_max])

        rules_pose = self.generate_rules(pose_error_ante, pose_error_avg_ante, d_pose, self.D_table)
        self.pose_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_pose))
        return self.pose_sim
    
    def _build_flc_alt(self):
        error_min = 0.0
        error_max = 0.3
        d_min = 0.0
        d_max = 1.0
        alt_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'alt_error')
        alt_error_avg_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'alt_error_avg')

        # CONSEQUENT
        d_alt = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'd_alt')

        # MEMBERSHIP FUNCTION
        alt_error_ante['S'] = fuzz.trimf(alt_error_ante.universe, [error_min, error_min, error_max/10])
        alt_error_ante['M'] = fuzz.trimf(alt_error_ante.universe, [error_min, error_max/10, 0.1])
        alt_error_ante['L'] = fuzz.trapmf(alt_error_ante.universe, [error_max/10, 0.15, error_max, error_max])

        alt_error_avg_ante['S'] = fuzz.trimf(alt_error_avg_ante.universe, [error_min, error_min, error_max/10])
        alt_error_avg_ante['M'] = fuzz.trimf(alt_error_avg_ante.universe, [error_min, error_max/10, 0.15])
        alt_error_avg_ante['L'] = fuzz.trapmf(alt_error_avg_ante.universe, [error_max/10, 0.15, error_max, error_max])

        d_alt['VS'] = fuzz.trimf(d_alt.universe, [0.0, 0.0, (1/4)*d_max])
        d_alt['S'] = fuzz.trimf(d_alt.universe, [0.0, (1/4)*d_max, (2/4)*d_max])
        d_alt['M'] = fuzz.trimf(d_alt.universe, [ (1/4)*d_max, (2/4)*d_max, (3/4)*d_max])
        d_alt['L'] = fuzz.trimf(d_alt.universe, [ (2/4)*d_max, (3/4)*d_max, (4/4)*d_max])
        d_alt['VL'] = fuzz.trimf(d_alt.universe, [ (3/4)*d_max, (4/4)*d_max, (4/4)*d_max])

        rules_alt = self.generate_rules(alt_error_ante, alt_error_avg_ante, d_alt, self.D_table)
        self.alt_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_alt))
        return self.alt_sim

    def _build_flc_velo(self):
        error_min = 0.0
        error_max = 1.0
        d_min = 0.0
        d_max = 1.0
        # ANTECEDANT
        velo_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'velo_error')
        velo_error_avg_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'velo_error_avg')

        # CONSEQUENT
        d_velo = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'd_velo')

        # MEMBERSHIP FUNCTION
        velo_error_ante['S'] = fuzz.trimf(velo_error_ante.universe, [error_min, error_min, error_max/2])
        velo_error_ante['M'] = fuzz.trimf(velo_error_ante.universe, [error_min, error_max/2, error_max])
        velo_error_ante['L'] = fuzz.trimf(velo_error_ante.universe, [error_max/2, error_max, error_max])

        velo_error_avg_ante['S'] = fuzz.trimf(velo_error_avg_ante.universe, [error_min, error_min, error_max/2])
        velo_error_avg_ante['M'] = fuzz.trimf(velo_error_avg_ante.universe, [error_min, error_max/2, error_max])
        velo_error_avg_ante['L'] = fuzz.trimf(velo_error_avg_ante.universe, [error_max/2, error_max, error_max])

        d_velo['VS'] = fuzz.trimf(d_velo.universe, [0.0, 0.0, (1/4)*d_max])
        d_velo['S'] = fuzz.trimf(d_velo.universe, [0.0, (1/4)*d_max, (2/4)*d_max])
        d_velo['M'] = fuzz.trimf(d_velo.universe, [ (1/4)*d_max, (2/4)*d_max, (3/4)*d_max])
        d_velo['L'] = fuzz.trimf(d_velo.universe, [ (2/4)*d_max, (3/4)*d_max, (4/4)*d_max])
        d_velo['VL'] = fuzz.trimf(d_velo.universe, [ (3/4)*d_max, (4/4)*d_max, (4/4)*d_max])

        rules_velo = self.generate_rules(velo_error_ante, velo_error_avg_ante, d_velo, self.D_table)
        self.velo_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_velo))
        return self.velo_sim
    
    def _build_flc_vz(self):
        error_min = 0.0
        error_max = 1.0
        d_min = 0.0
        d_max = 1.0
        # ANTECEDANT
        vz_error_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'vz_error')
        vz_error_avg_ante = ctrl.Antecedent(np.linspace(error_min, error_max, 200), 'vz_error_avg')

        # CONSEQUENT
        d_vz = ctrl.Consequent(np.linspace(d_min, d_max, 200), 'd_vz')

        # MEMBERSHIP FUNCTION
        vz_error_ante['S'] = fuzz.trimf(vz_error_ante.universe, [error_min, error_min, error_max/2])
        vz_error_ante['M'] = fuzz.trimf(vz_error_ante.universe, [error_min, error_max/2, error_max])
        vz_error_ante['L'] = fuzz.trimf(vz_error_ante.universe, [error_max/2, error_max, error_max])

        vz_error_avg_ante['S'] = fuzz.trimf(vz_error_avg_ante.universe, [error_min, error_min, error_max/2])
        vz_error_avg_ante['M'] = fuzz.trimf(vz_error_avg_ante.universe, [error_min, error_max/2, error_max])
        vz_error_avg_ante['L'] = fuzz.trimf(vz_error_avg_ante.universe, [error_max/2, error_max, error_max])

        d_vz['VS'] = fuzz.trimf(d_vz.universe, [0.0, 0.0, (1/4)*d_max])
        d_vz['S'] = fuzz.trimf(d_vz.universe, [0.0, (1/4)*d_max, (2/4)*d_max])
        d_vz['M'] = fuzz.trimf(d_vz.universe, [ (1/4)*d_max, (2/4)*d_max, (3/4)*d_max])
        d_vz['L'] = fuzz.trimf(d_vz.universe, [ (2/4)*d_max, (3/4)*d_max, (4/4)*d_max])
        d_vz['VL'] = fuzz.trimf(d_vz.universe, [ (3/4)*d_max, (4/4)*d_max, (4/4)*d_max])

        rules_vz = self.generate_rules(vz_error_ante, vz_error_avg_ante, d_vz, self.D_table)
        self.vz_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules_vz))
        return self.vz_sim
    
    def _run_flc(self, err_pose, err_alt, err_velo, err_vz, err_avg_pose, err_avg_alt, err_avg_velo, err_avg_vz):
        self.pose_sim.input['pose_error'] = np.clip(err_pose, 0.0, 1.0)
        self.pose_sim.input['pose_error_avg'] = np.clip(err_avg_pose, 0.0, 1.0)
        self.pose_sim.compute()

        self.velo_sim.input['velo_error'] = np.clip(err_velo, 0.0, 1.0)
        self.velo_sim.input['velo_error_avg'] = np.clip(err_avg_velo, 0.0, 1.0)
        self.velo_sim.compute()

        self.alt_sim.input['alt_error'] = np.clip(err_alt, 0, 1)
        self.alt_sim.input['alt_error_avg'] = np.clip(err_avg_alt, 0, 1)
        self.alt_sim.compute()

        self.vz_sim.input['vz_error'] = np.clip(err_vz, 0, 1)
        self.vz_sim.input['vz_error_avg'] = np.clip(err_avg_vz, 0, 1)
        self.vz_sim.compute()

        d_pose = self.pose_sim.output['d_pose']
        d_alt =self.alt_sim.output['d_alt']
        d_velo = self.velo_sim.output['d_velo']
        d_vz = self.vz_sim.output['d_vz']
        return d_pose, d_alt, d_velo, d_vz
    
    def lpf(self, raw, prev_filt, alpha=0.9):
        filtered = alpha*(prev_filt)+(1-alpha)*raw
        return filtered
    
if __name__ == "__main__":
    rospy.init_node("flc_hexacopter")
    FuzzyLogicControl_Position()
    rospy.spin()
