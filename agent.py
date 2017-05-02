import random
import pandas as pd
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.q = {}  # q value table
        self.alpha = 0.6    # learning rate
        self.gamma = 0.8   # discount
        self.epsilon = 1.0   
        self.pre_action = None
        self.pre_reward = 0.0
        self.success = []  # record success trip per 100 trials
        self.invalidmove = []  # record average of rewards received in each 100 trials. 
        self.time = []     # record time used for each trial. The time is presented by actually used time/ the trial's initial deadline which means its range is between 0 - 1.
                           # For trips that exceed deadline, use 2 (can be regarded as heavy penalty for time) to present its used time in the current trial.
        self.orgdeadline = 0 # The initial deadline calculated by system when new trial starts.
        self.count = 0 # Count the number of trials.

        # Initialize all the q table values
        for l in ['red', 'green']:
            for o in self.env.valid_actions:
                for le in self.env.valid_actions:
                    for nw in ['forward', 'left','right']:
                        sta = 'light: {}; oncoming: {}; left: {}; next_waypoint: {}'.format(l, o, le, nw)
                        self.q[sta] = {}
                        for act in self.env.valid_actions:
                            self.q[sta][act] = 0.0

        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.pre_action = None
        self.state = None
        self.pre_reward = None

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
                

        # TODO: Update state

        pre_state = self.state

        self.state = 'light: {}; oncoming: {}; left: {}; next_waypoint: {}'.format(inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)


        # TODO: Select action according to your policy

        if pre_state == None or random.random() < self.epsilon:
            action = random.choice(self.env.valid_actions) 
        else:
            action = max(self.q[self.state], key = lambda act: self.q[self.state][act])
            maxQ = max(self.q[self.state].values())
            actlist = filter(lambda act : self.q[self.state][act] == maxQ, self.q[self.state])
            if len(actlist) > 1:
                action = random.choice(actlist)
        

        # Execute action and get reward
        
        reward = self.env.act(self, action)  

        # Record success trials and num of invalid moves earned and the number of trials

        if self.count % 100 == 0 and pre_state == None:   # Use 100 trials as a unit to record success trials, earned rewards and used time.
            self.success.append(0)
            self.invalidmove.append(0)
            self.time.append([])

        if reward == -1:
            self.invalidmove[-1] -= reward

        # Make sure the self.orgdeadline has the initial deadline of each new trials.
        if pre_state == None:
            self.count += 1
            self.orgdeadline = deadline
        
        # Update epsilon every 10 trials by multiplying 0.97
        if self.count % 10 == 0 and pre_state == None:
            self.epsilon = self.epsilon * 0.97

        # Update used time for each trial by detecting whether current location = destination
        location = self.env.agent_states[self]["location"] 
        destination = self.env.agent_states[self]["destination"]

        if location == destination:
            self.success[-1] += 1
            self.time[-1].append((t + 0.0)/self.orgdeadline)
        elif deadline == 0:
            self.time[-1].append(2)    # 2 is the heavy penalty for trial has exceeded deadline

        # Output statistics of results
        if self.count == 1000:
            if location == destination or deadline == 0:
                results={}                
                for i, item in enumerate(self.time):
                    results[i * 100 + 100] = item

                df = pd.DataFrame(data = results)
                print df.describe()
                #df.to_csv('timespendvsdeadline.csv')

                print 'Number of Invalid Move per 100 Trials: {}'.format(self.invalidmove)
                print 'Success Trials per 100: {}'.format(self.success)
                print 'alpha: {}, gamma: {}'.format(self.alpha, self.gamma)

        
        #print 'current trial number: {}'.format(self.count)

        # TODO: Learn policy based on state, action, reward
        if pre_state != None:
            self.q[pre_state][self.pre_action] = (1 - self.alpha) * self.q[pre_state][self.pre_action] + self.alpha * (self.pre_reward + self.gamma * self.q[self.state][action])            

        self.pre_action = action
        self.pre_reward = reward
        # LearningAgent.update(): 

        #print "deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=1000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()

