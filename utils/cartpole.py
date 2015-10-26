#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: __INIT__.PY
Date: Saturday, February 25 2012
Description: A cartpole implementation based on Rich Sutton's code.
"""

import math
import random as pr

class CartPole(object):

    def __init__(self, x = 0.0, xdot = 0.0, theta = 0.0, thetadot = 0.0):
        self.x = x
        self.xdot = xdot
        self.theta = theta
        self.thetadot = thetadot

        # some constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5		  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02		  # seconds between state updates
        self.fourthirds = 1.3333333333333

    def failure(self):
        twelve_degrees = 0.2094384
        if (not -2.4 < self.x < 2.4) or (not -twelve_degrees < self.theta < twelve_degrees):
            return True
        else:
            return False

    def reset(self):
        self.x,self.xdot,self.theta,self.thetadot = (0.0,0.0,0.0,0.0)

    def random_policy(self, *args):
        return pr.choice([0,1])

    def single_episode(self, policy=None, boxed=False):
        self.reset()
        if policy is None: policy = self.random_policy

        trace = []
        
        # MOD: Support policies which work with boxed states
        if boxed:
            next_action = policy(self.state())
        else:
            next_action = policy(self.x,self.xdot,self.theta,self.thetadot)
            
        while not self.failure():
            pstate, paction, reward, state = self.move(next_action, boxed=boxed)
            
            if boxed:
                next_action = policy(self.state())
            else:
                next_action = policy(self.x,self.xdot,self.theta,self.thetadot)
                
            trace.append([pstate, paction, reward, state, next_action])

        return trace

    def state(self): # get boxed version of the state as per the original code

        one_degree = 0.0174532
        six_degrees = 0.1047192
        twelve_degrees = 0.2094384
        fifty_degrees = 0.87266

        if (not -2.4 < self.x < 2.4) or (not -twelve_degrees < self.theta < twelve_degrees):
            return -1

        box = 0

        if self.x < -0.8:
            box = 0
        elif self.x < 0.8:
            box = 1
        else:
            box = 2

        if self.xdot < -0.5:
            pass
        elif self.xdot < 0.5:
            box += 3
        else:
            box += 6

        if self.theta < -six_degrees:
            pass
        elif self.theta < -one_degree:
            box += 9
        elif self.theta < 0:
            box += 18
        elif self.theta < one_degree:
            box += 27
        elif self.theta < six_degrees:
            box += 36
        else:
            box += 45

        if self.thetadot < -fifty_degrees:
            pass
        elif self.thetadot < fifty_degrees:
            box += 54
        else:
            box += 108

        return box;

    def reward(self):
        if self.failure():
            return -1.0
        else:
            return 0.0

    def move(self, action, boxed=False): # binary L/R action
        force = 0.0
        if action > 0:
            force = self.force_mag
        else:
            force = -self.force_mag

        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)

        tmp = (force + self.polemass_length * (self.thetadot ** 2) * sintheta) / self.total_mass;
        thetaacc = (self.gravity * sintheta - costheta * tmp) / (self.length * (self.fourthirds - self.masspole * costheta ** 2 / self.total_mass))
        xacc = tmp - self.polemass_length * thetaacc * costheta / self.total_mass

        (px,pxdot,ptheta,pthetadot) = (self.x, self.xdot, self.theta, self.thetadot)
        pstate = self.state()

        self.x += self.tau * self.xdot
        self.xdot += self.tau * xacc
        self.theta += self.tau * self.thetadot
        self.thetadot += self.tau * thetaacc

        if boxed:
            return pstate, action, self.reward(), self.state()
        else:
            return [px,pxdot,ptheta,pthetadot],action,self.reward(),[self.x,self.xdot, self.theta, self.thetadot]

if __name__ == '__main__':

    cp = CartPole()

    global ccount
    ccount = 1

    def alternating_policy(x,xdot,theta,thetadot):
        global ccount
        ccount = ccount + 1
        if ccount % 2 == 0:
            return 1
        else:
            return 0


    if False:
        for i in range(100):
            print cp.update(i % 2), cp.state()
        for i in range(100):
            print cp.update(i % 2),  cp.state()

    if True:
        t = cp.single_episode()
        for i in t:
            print i[2],i[1]

        t = cp.single_episode(alternating_policy)
        for i in t:
            print i[2],i[1]
