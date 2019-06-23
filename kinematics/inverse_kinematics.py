'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
from numpy import random
import numpy as np 
from autograd import grad    


class InverseKinematicsAgent(ForwardKinematicsAgent):
    
    def error_func(self,theta, target,effector_name):
        ts = self.forward_kinematics(self.chains[effector_name])
        te = ts[-1]
        e = target - te
        return np.sum(e*e)
        
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        N = len(self.chains[effector_name]) - 1 #amount of links
        theta = random.random(N)
        joint_angles = []
        for name in self.chains[effector_name]:
            joint_angles[name] = self.perception.joint[name]
        
        func = lambda t: error_func(t, transform)
        func_grad = grad(func)
        for i in range(1000):
            e = func(theta)
            d = func_grad(theta)
            theta -= d * 1e-2
            if e < 1e-4:
                break
    
        return theta
        # YOUR CODE HERE
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        joint_angles = self.inverse_kinematics(effector_name, transform)

        names = self.chains[effector_name]
        times = [[0, 5]] * len(names)
        keys = []
        for i, name in enumerate(names):
            keys.insert(i, [[self.perception.joint[name], [3, 0, 0]], [joint_angles[name], [3, 0, 0]]])

        self.keyframes = (names, times, keys) 


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = 0.26
    agent.set_transforms('LLeg', T)
    agent.run()
