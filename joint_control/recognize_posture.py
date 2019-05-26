'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''

import pickle
from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
import numpy
from os import listdir, path

ROBOT_POSE_CLF = 'robot_pose.pkl'


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        self.posture_classifier = pickle.load(open(ROBOT_POSE_CLF)) # LOAD YOUR CLASSIFIER

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        # YOUR CODE HERE
	data = []
        data.append(perception.joint['LHipYawPitch'])
        data.append(perception.joint['LHipRoll'])
        data.append(perception.joint['LHipPitch'])
        data.append(perception.joint['LKneePitch'])
        data.append(perception.joint['RHipYawPitch'])
        data.append(perception.joint['RHipRoll'])
        data.append(perception.joint['RHipPitch'])
        data.append(perception.joint['RKneePitch'])
        data.append(perception.imu[0])
        data.append(perception.imu[1])
        data = numpy.array(data).reshape(1, -1)
        # print(data)
        posture = self.posture_classifier.predict(data)
	print (posture)
        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
