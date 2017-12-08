# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util, math
from game import os, Directions, Actions
import game
import cPickle as pickle
import distanceCalculator

#################
# Team creation #
#################
class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:

    def registerInitialState(self, state): # inspects the starting state
    """
    def __init__(self, index=0):
        #self.index = index
        pass

    def getAction1(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        util.raiseNotDefined()





def createTeam(firstIndex, secondIndex, isRed,
                             first = 'QCTFAgent', second = 'QCTFAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''


    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)














        

class ValueEstimationAgent(Agent):
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining = 10):
        """
        Sets options, which can be passed in via the Pacman command line using -a alpha=0.5,...
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(numTraining)

    ####################################
    #    Override These Functions      #
    ####################################
    def getQValue(self, state, action):
        """
        Should return Q(state,action)
        """
        util.raiseNotDefined()

    def getValue(self, state):
        """
        What is the value of this state under the best action?
        Concretely, this is given by

        V(s) = max_{a in actions} Q(s,a)
        """
        util.raiseNotDefined()

    def getPolicy(self, state):
        """
        What is the best action to take in the state. Note that because
        we might want to explore, this might not coincide with getAction
        Concretely, this is given by

        policy(s) = arg_max_{a in actions} Q(s,a)

        If many actions achieve the maximal Q-value,
        it doesn't matter which is selected.
        """
        util.raiseNotDefined()

    def getAction1(self, state):
        """
        state: can call state.getLegalActions()
        Choose an action and return it.
        """
        util.raiseNotDefined()

class ReinforcementAgent(ValueEstimationAgent):
    """
      Abstract Reinforcemnt Agent: A ValueEstimationAgent
            which estimates Q-Values (as well as policies) from experience
            rather than a model

        What you need to know:
                    - The environment will call
                      observeTransition(state,action,nextState,deltaReward),
                      which will call update1(state, action, nextState, deltaReward)
                      which you should override.
        - Use state.getLegalActions(state) to know which actions
                      are available in a state
    """
    ####################################
    #    Override These Functions      #
    ####################################

    def update1(self, state, action, nextState, reward):
        """
                This class will call this function, which you write, after
                observing a transition and reward
        """
        util.raiseNotDefined()

    ####################################
    #    Read These Functions          #
    ####################################

    def getLegalActions(self,state):
        """
          Get the actions available for a given
          state. This is what you should use to
          obtain legal actions for a state
        """
        return self.actionFn(state)

    def observeTransition(self, state,action,nextState,deltaReward):
        """
            Called by environment to inform agent that a transition has
            been observed. This will result in a call to self.update
            on the same arguments

            NOTE: Do *not* override or call this function
        """
        self.episodeRewards += deltaReward
        self.update1(state,action,nextState,self.getReward(state))

    def startEpisode(self):
        """
          Called by environment when new episode is starting
        """
        self.lastState = None
        self.lastAction = None
        self.episodeRewards = 0.0

    def stopEpisode(self):
        """
          Called by environment when episode is done
        """
        if self.episodesSoFar < self.numTraining:
            self.accumTrainRewards += self.episodeRewards
        else:
            self.accumTestRewards += self.episodeRewards
        self.episodesSoFar += 1
        if self.episodesSoFar >= self.numTraining:
            # Take off the training wheels
            self.epsilon = 0.0    # no exploration
            self.alpha = 0.0      # no learning

    def isInTraining(self):
        return self.episodesSoFar < self.numTraining

    def isInTesting(self):
        return not self.isInTraining()

    def __init__(self, actionFn = None, numTraining=100, epsilon=0.5, alpha=0.5, gamma=1):
        """
        actionFn: Function which takes a state and returns the list of legal actions

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        if actionFn == None:
            actionFn = lambda state: state.getLegalActions()
        self.actionFn = actionFn
        self.episodesSoFar = 0
        self.accumTrainRewards = 0.0
        self.accumTestRewards = 0.0
        self.numTraining = int(numTraining)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)

    ################################
    # Controls needed for Crawler  #
    ################################
    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setLearningRate(self, alpha):
        self.alpha = alpha

    def setDiscount(self, discount):
        self.discount = discount

################################################################################################################################ Eric Changed Stuff Below
    def doAction(self,state,action):
        """
            Called by inherited class when
            an action is taken in a state
        """
        self.lastState = state
        self.lastAction = action

    ###################
    # Pacman Specific #
    ###################
    def observationFunction(self, state):
        """
            This is where we ended up after our last action.
            The simulation should somehow ensure this is called
        """
        ########################################################################################Eric Is Changing Rewards

        if not self.lastState is None:
            reward = (state.getScore() - self.lastState.getScore())*50

            if state.isOnRedTeam():
                if len(util.matrixAsList(state.getBlueFood())) < len(util.matrixAsList(self.lastState.getBlueFood())):
                    reward += 10

                killScore = 0
                enemyIndicies = state.getBlueTeamIndices()
                prevDistances = util.Counter()
                for enemy in enemyIndicies:
                    if self.lastState.getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(self.lastState.getAgentPosition(enemy), self.lastState.getAgentPosition(self.index))
                newDistances = util.Counter()
                for enemy in enemyIndicies:
                    if state.getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(state.getAgentPosition(enemy), state.getAgentPosition(self.index))
                
                x1,y1 = state.getAgentPosition(self.index)

                halfway = state.data.layout.width/2
                if x1 < halfway:
                
                    for enemyIndex in enemyIndicies:
                        if prevDistances[enemyIndex] == 1:
                            if x1 != 30 and (newDistances[enemyIndex] > 2 or newDistances[enemyIndex] == 0):
                                #TODO: Give more score for killing enemy with food
                                killScore += 100
                    reward += killScore


            else:
                if len(util.matrixAsList(state.getRedFood())) < len(util.matrixAsList(self.lastState.getRedFood())):
                    reward += 10
                killScore = 0
                enemyIndicies = state.getRedTeamIndices()
                prevDistances = util.Counter()
                for enemy in enemyIndicies:
                    if self.lastState.getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(self.lastState.getAgentPosition(enemy), self.lastState.getAgentPosition(self.index))
                newDistances = util.Counter()
                for enemy in enemyIndicies:
                    if state.getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(state.getAgentPosition(enemy), state.getAgentPosition(self.index))
                
                x1,y1 = state.getAgentPosition(self.index)

                halfway = state.data.layout.width/2
                if x1 > halfway:
                
                    for enemyIndex in enemyIndicies:
                        if prevDistances[enemyIndex] == 1:
                            if x1 != 30 and (newDistances[enemyIndex] > 2 or newDistances[enemyIndex] == 0):
                                #TODO: Give more score for killing enemy with food
                                killScore += 100
                    reward += killScore

            #reward -= 50

            print "REWARD: ", reward












            self.observeTransition(self.getLastState(), self.lastAction, state, reward)
        return state

    def registerInitialState(self, state):
        self.startEpisode()
        if self.episodesSoFar == 0:
            print 'Beginning %d episodes of Training' % (self.numTraining)

    def final(self, state):
        """
          Called by Pacman game at the terminal state
        """
        deltaReward = state.getScore() - self.lastState.getScore()
        self.observeTransition(self.lastState, self.lastAction, state, deltaReward)
        self.stopEpisode()

        # Make sure we have this var
        if not 'episodeStartTime' in self.__dict__:
            self.episodeStartTime = time.time()
        if not 'lastWindowAccumRewards' in self.__dict__:
            self.lastWindowAccumRewards = 0.0
        self.lastWindowAccumRewards += state.getScore()

        NUM_EPS_UPDATE = 100
        if self.episodesSoFar % NUM_EPS_UPDATE == 0:
            print 'Reinforcement Learning Status:'
            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            if self.episodesSoFar <= self.numTraining:
                trainAvg = self.accumTrainRewards / float(self.episodesSoFar)
                print '\tCompleted %d out of %d training episodes' % (
                       self.episodesSoFar,self.numTraining)
                print '\tAverage Rewards over all training: %.2f' % (
                        trainAvg)
            else:
                testAvg = float(self.accumTestRewards) / (self.episodesSoFar - self.numTraining)
                print '\tCompleted %d test episodes' % (self.episodesSoFar - self.numTraining)
                print '\tAverage Rewards over testing: %.2f' % testAvg
            print '\tAverage Rewards for last %d episodes: %.2f'  % (
                    NUM_EPS_UPDATE,windowAvg)
            print '\tEpisode took %.2f seconds' % (time.time() - self.episodeStartTime)
            self.lastWindowAccumRewards = 0.0
            self.episodeStartTime = time.time()

        if self.episodesSoFar == self.numTraining:
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - state.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q = util.Counter()




    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
       # print "GETQVALUE in QLEARNINGAGENT, QVALUE: ",self.Q[(state, action)]

        return self.Q[(state, action)]


        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        valueList = []
        ############################################################################################################ Eric Changed state to self.index
        for a in state.getLegalActions(self.index):
            valueList.append(self.getQValue(state, a))
        if len(valueList) == 0:
            return 0.0
        return max(valueList)


        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
                ############################################################################################################ Eric Changed state to self.index
        
        legalActions = state.getLegalActions(self.index)
        #print "COMPUTEACTIONFROMQVALUES in QLEARNINGAGENT, LEGALACTIONS: ", legalActions
        if len(legalActions) == 0:
            return None
        maxValue = self.getQValue(state, legalActions[0])
        maxAction = legalActions[0]

        for a in legalActions:
            myQValue = self.getQValue(state, a)
            #print "COMPUTEACTIONFROMQVALUES in QLEARNINGAGENT, MYQVALUE: ", myQValue, " MAXVALUE: ", maxValue
            if myQValue > maxValue:
                maxValue = self.getQValue(state, a)
                maxAction = a
            if myQValue == maxValue:
                if util.flipCoin(0.5):
                    maxValue = self.getQValue(state, a)
                    maxAction = a
        #print "COMPUTEACTIONFROMQVALUES in QLEARNINGAGENT, MAXACTION: ", maxAction
        return maxAction
        util.raiseNotDefined()

    def getAction1(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
                ############################################################################################################ Eric Changed state to self.index

        legalActions = state.getLegalActions(self.index)
        #print "LEGAL ACTIONS IN GETACTION1 IN QLEARNINGAGENT: ", legalActions
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return None
        coinTruth = util.flipCoin(self.epsilon)
        if coinTruth:
            acToReturn = random.choice(legalActions)
            #print "GETACTION1 IN QLEARNINGAGENT COINTRUTH IS TRUE, ACTION IS : ", acToReturn
            return acToReturn



        #util.raiseNotDefined()
        acToReturn = self.computeActionFromQValues(state)
        #print "GETACTION1 IN QLEARNINGAGENT COINTRUTH IS FALSE< ACTION IS : ", acToReturn
        #self.doAction(state, acToReturn)
        return acToReturn

    def update1(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
                ############################################################################################################ Eric Changed nextState to other stuff
        
        actionList = nextState.getLegalActions(self.index)

        if (not (nextState == None)) and len(actionList) > 0 :
            expectedRewardList = []
            #print "state ",nextState," has legal actions ", state.getLegalActions(nextState)
            for a in actionList:
                #print "next state: ",nextState," action: ",a, "Value: ", self.Q[(nextState, a)]
                expectedRewardList.append(self.Q[(nextState, a)])
            #print "expected reward list: ", expectedRewardList
            self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * (reward + self.discount * max(expectedRewardList) - self.Q[(state, action)])
            #print self.Q
            return
        else:
            self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * (reward - self.Q[(state, action)])
            return

        #print "I should never be here"
        #util.raiseNotDefined()


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.5,gamma=0.8,alpha=0.5, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        #self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction1(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction1(self,state)
################################################################################################################################ Eric Changed Stuff Below
        #print "ACTION IN PACMANQAGENT: ", action
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        extractor = 'CTFExtractor'
        #########################
        #print "what is the extractor name: ", extractor
        ############################
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        #print "getQValue in ApproximateQAgent"
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        "*** YOUR CODE HERE ***"
        weights = self.getWeights()
        features = self.featExtractor.getFeatures(state, action, self)

        value = 0

        #print "FEATURES: ", features
        #print "WEIGHTS: ", weights

        for feature in features:
            value += features[feature]*weights[feature]
        return value
        #util.raiseNotDefined()

    def update1(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        #print "update1 in ApproximateQAgent"
        "*** YOUR CODE HERE ***"
        ##################################################################################################################################Eric Did Stuff
        actionList = nextState.getLegalActions(self.index)

        print "UPDATE IS CALLED, REWARD IS :", reward
        #print "Action List", actionList




        weights = self.getWeights()

        features = self.featExtractor.getFeatures(state, action, self)
        #self.myFeats = features
        if self.index == 0:
            pass
            #print "FEATURES: ",features
        value = self.computeValueFromQValues(nextState)
        qValue = self.getQValue(state,action)
        #print "value", value, "qValue", qValue
        for feature in features:
            if len(actionList) != 0:
                weights[feature] = weights[feature] + self.alpha * (reward + self.discount * value - qValue) * features[feature]
            else:
                weights[feature] = weights[feature] + self.alpha * (reward - qValue) * features[feature]
            #print "feature", feature, "weights", weights[feature]
            #print "weights", weights

        #util.raiseNotDefined()
        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #print "weights",weights
            pass

class FeatureExtractor:
    def getFeatures(self, state, action, thisAgent):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action, thisAgent):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action, thisAgent):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action, thisAgent):

        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        return features

#RUU section

class EnemyAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomEnemy( EnemyAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = [] # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getAgentPosition(self.index) # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index: # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass

        
class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent);
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles


    def initializeUniformly(self, gameState):
        """
        Initializes a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        "*** YOUR CODE HERE ***"
        #this works I think, but I don't have a way to test. Try it out on paper if you want.
        numPositions = len(self.legalPositions)
        #print "legal positions: ",self.legalPositions
        #print "numPositions: ", numPositions
        step = (numPositions % self.numParticles) - 1
        particleList = [None]*self.numParticles #poor attempt at optimization
        last = step * -1 #ensures start at 0
        #print "step: ",step
        
        for p in range (0, self.numParticles):
            #print "last: ",last
            
            new = last + step
            #print "new: ", new
            last = new
            
            particleList[p] = (self.legalPositions[new % numPositions])
            
        
        self.particles = particleList
        #print "Initial distribution: ", particleList
        return particleList

    def observe(self, observation, gameState, callerIndex):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        noisyDistance = observation
        #emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getAgentPosition(callerIndex)
        "*** YOUR CODE HERE ***"
        s = [None]*self.numParticles
        #print "got observation: ", noisyDistance
        #Special Case #1: Pacman eats the ghost
        if noisyDistance == None:
            for i in range(0, self.numParticles):
                s[i] = self.getJailPosition()
            self.particles = s
            return s
        
        
        #update the beliefs. This step is taking 5ever
        #print "Updating belief distribution"
        beliefs = util.Counter()
        beliefDistribution = self.getBeliefDistribution()
        for p in self.particles:
            
            #weight = emissionModel[util.manhattanDistance(p, pacmanPosition)] * beliefDistribution[p]
            weight = gameState.getDistanceProb(util.manhattanDistance(p, pacmanPosition), noisyDistance)
            beliefs[p] = weight
            #print "given ",p," weight ", weight
            #beliefs.normalize()
        
        #Special Case #2: All particles have 0 weight
        if beliefs.totalCount() == 0:
            #print "SPECIAL CASE 2 TRIGGERED"
             #get the prior distribution. 
            priorDistribution = util.Counter()
            for p in self.initializeUniformly(gameState):
                priorDistribution[p] = 1
            beliefs = priorDistribution.normalize()
        
        #sample from new distribution
        #print "sampling from new distribution"
        if beliefs == None:
            #print "BELIEFS IS NULL"
            beliefs = util.Counter()
            for p in self.initializeUniformly(gameState):
                beliefs[p] = 1
            beliefs.normalize()
        
        #beliefs.normalize()
        for i in range(0, self.numParticles):
            s[i] = util.sample(beliefs)
        
        #update the particle list
        self.particles = s
        return s
        
        
        util.raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        "*** YOUR CODE HERE ***"
        newList = []

        for oldPos in self.particles:
            newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))
            newList.append(util.sample(newPosDist))

        self.particles = newList
        
        #util.raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """
        "*** YOUR CODE HERE ***"
        beliefs = util.Counter()
        for p in self.particles:
            beliefs[p] += 1
        beliefs.normalize()
        return beliefs
      
def d2lPosReciever(agentIndex, state, position):
    halfway = state.data.layout.width/2
    print "AGENT INDEX: ", agentIndex
    dist = position[0] - halfway
    
    if not state.isOnRedTeam(agentIndex):
        dist *= -1
    return dist
        
def distanceToLine(agentIndex, state):
    #returns the manhattan distance to the line. Positive if on enemy side, negative if on home side
    halfway = state.data.layout.width/2
    #print "AGENT INDEX: ", agentIndex
    position = state.getAgentPosition(agentIndex)
    dist = position[0] - halfway
    
    if not state.isOnRedTeam(agentIndex):
        dist *= -1
    return dist
       
        
class CTFExtractor(FeatureExtractor):
    """
    returns features:
    
    """
    def getFeatures(self, state, action, thisAgent):
    
        features = util.Counter()
    
        #info gathering
        #food = state.getFood()
        nextState = state.generateSuccessor(thisAgent.index, action)
        
        walls = state.getWalls()
        halfway = state.data.layout.width/2
        index = thisAgent.index
        myPosition = state.getAgentPosition(index)
        d2l = myPosition[0] - halfway
        nextScore = thisAgent.getScore(nextState)
        timeLeft = state.data.timeleft
        
        # compute the location of pacman after he takes the action
        x, y = myPosition   
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        
        #team specific- move to init?
        if thisAgent.isRed:
            myFood = state.getRedFood()
            enemyFood = state.getBlueFood()
            myCapsules = state.getRedCapsules()
            enemyCapsules = state.getBlueCapsules()
            enemies = state.getBlueTeamIndices()
            myTeam = state.getRedTeamIndices()
        else: #on blue
            myFood = state.getBlueFood()
            enemyFood = state.getRedFood()
            myCapsules = state.getBlueCapsules()
            enemyCapsules = state.getRedCapsules()
            enemies = state.getRedTeamIndices()
            myTeam = state.getBlueTeamIndices()
            d2l *= -1
            
        if myTeam[0] == index:
            teammateIndex = myTeam[1]
        else:
            teammateIndex = myTeam[0]
            
        teammatePosition = state.getAgentPosition(teammateIndex)
        teammateD2L = distanceToLine(teammateIndex, state)


        features["teammate-distance-to-line"] = teammateD2L / 10

        d2t = thisAgent.distances.getDistance(myPosition, teammatePosition)
        features["distance-to-teammate"] = d2t / 10
            
            
        enemy1Position = state.getAgentPosition(enemies[0])
        if enemy1Position != None:
            enemy1DistanceToLine = distanceToLine(enemies[0], state) #gives manhat distance to line and ghost/pacman in the form of +/-
            features["enemy1-distance-to-line"] = enemy1DistanceToLine
            d2e1 = thisAgent.distances.getDistance(myPosition, enemy1Position)
            features["distance-to-enemy1"] = d2e1
            td2e1 = thisAgent.distances.getDistance(teammatePosition, enemy1Position)
            features["td2e1"] = td2e1
        else:
            enemy1Position = thisAgent.pfilters[0].getBeliefDistribution().argMax()
            enemy1DistanceToLine = d2lPosReciever(enemies[0], state, enemy1Position) #gives manhat distance to line and ghost/pacman in the form of +/-
            features["enemy1-distance-to-line"] = enemy1DistanceToLine
            d2e1 = thisAgent.distances.getDistance(myPosition, enemy1Position)
            features["distance-to-enemy1"] = d2e1
            td2e1 = thisAgent.distances.getDistance(teammatePosition, enemy1Position)
            features["td2e1"] = td2e1
        
        enemy2Position = state.getAgentPosition(enemies[1])
        if enemy2Position != None:
            enemy2DistanceToLine = distanceToLine(enemies[1], state)
            features["enemy2-distance-to-line"] = enemy2DistanceToLine
            d2e2 = thisAgent.distances.getDistance(myPosition, enemy2Position)
            features["distance-to-enemy2"] = d2e2
            td2e2 = thisAgent.distances.getDistance(teammatePosition, enemy2Position)
            features["td2e2"] = td2e2
        else:
            enemy2Position = thisAgent.pfilters[0].getBeliefDistribution().argMax()
            enemy2DistanceToLine = d2lPosReciever(enemies[1], state, enemy2Position) #gives manhat distance to line and ghost/pacman in the form of +/-
            features["enemy2-distance-to-line"] = enemy1DistanceToLine
            d2e2 = thisAgent.distances.getDistance(myPosition, enemy1Position)
            features["distance-to-enemy1"] = d2e2
            td2e2 = thisAgent.distances.getDistance(teammatePosition, enemy1Position)
            features["td2e2"] = td2e2    
       
        
        #food logic- distance to nearest food
        dist = closestFood((next_x, next_y), enemyFood, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height) * 10
        #food logic- amount of food carrying
        foodHolding = thisAgent.foodHolding
        features["food-holding"] = foodHolding
        
        enemy1FoodHolding = thisAgent.foodHoldings[enemies[0]]
        features["enemy1-food-held"] = enemy1FoodHolding
        
        enemy2FoodHolding = thisAgent.foodHoldings[enemies[1]]
        features["enemy2-food-held"] = enemy2FoodHolding
        
        
        
        #assign to features
        features["distance-to-line"] = d2l /10
        
        features["next-score"] = nextScore
        
        features["time-left"] = timeLeft / 1200

        features["enemy-food-left-after"] = len(thisAgent.getFood(nextState).asList())
        features["my-food-left-after"] = len(thisAgent.getFoodYouAreDefending(nextState).asList())

        #features.divideAll(10.0)
        return features
        
        
    def distanceToLine(self, agentIndex, state):
        #returns the manhattan distance to the line. Positive if on enemy side, negative if on home side --> not actually true anymore -Eric
        halfway = state.data.layout.width/2
        
        dist = abs(state.getAgentPosition(agentIndex)[0] - halfway)
        #if not state.isOnRedTeam(agentIndex):
        #    dist *= -1
        return dist




    







class QCTFAgent(CaptureAgent, ApproximateQAgent):
    epsilon = .3
    alpha = 0.5
    discount = 1
    featExtractor = CTFExtractor()
    #statics
    enemy1FoodHolding = 0
    enemy2FoodHolding = 0
    lastState = None

    foodHoldings = [0,0,0,0]


    pfilters = [ParticleFilter(RandomEnemy(a)) for a in range (0,2)]


    def registerInitialState(self, gameState):
        #print "PRINT 1"
        CaptureAgent.registerInitialState(self, gameState)
        self.isRed = gameState.isOnRedTeam(self.index)
        if self.isRed:
            self.enemyIndices = gameState.getBlueTeamIndices()
            self.teamIndices = gameState.getRedTeamIndices()
        else:
            self.enemyIndices = gameState.getRedTeamIndices()
            self.teamIndices = gameState.getBlueTeamIndices()
        
        if self.index == self.teamIndices[0]:
            self.myPf = self.pfilters[0]
        else:
            self.myPf = self.pfilters[1]

        self.distances = distanceCalculator.Distancer(gameState.data.layout)
        self.distances.getMazeDistances()
        
        self.foodHolding = 0
        self.enemy1FoodHolding = 0
        self.enemy2FoodHolding = 0
        self.lastState = gameState
        
        #RUU
        self.myPf.initialize(gameState)
        
        #print "PRINT 2"


        ApproximateQAgent.__init__(self)
        self.startEpisode()
        #print "PRINT 2"
        self.fileName = "testFile" + str(self.index)
        if not os.path.isfile(self.fileName):
            for i in range(0,20):
                print "404 FILE NOT FOUND ------------ ERROR ERROR ERROR ERROR ERROR ERROR ERROR "
            self.weights = util.Counter()
        else:
            myFile = open(self.fileName, "rb")
            unpickled = pickle.load(myFile)
            self.weights = unpickled
        #print "PRINT 3"

        self.stateActionPairs = []


        self.initial_food = self.getFood(gameState).count()
        self.initial_defending_food = self.getFoodYouAreDefending(gameState).count()





    def chooseAction(self, gameState):
        #print "PRINT 4"
        #print "MY INDEX: ", self.index
        #print "POSSIBLE ACTIONS: ", gameState.getLegalActions(self.index)
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        x1,y1 = myPos
        lastAgentIndex = (self.index - 1)%4
        #calculate enemy food holdings
        print "debugging"
        print "DEBUG: ",self.getFoodYouAreDefending(gameState)
        print "Last State: ", self.getLastState()

        if self.getLastState() is not None:
            dOurFood = len(self.getFoodYouAreDefending(gameState).asList()) - len(self.getFoodYouAreDefending(self.getLastState()).asList())
            self.foodHoldings[lastAgentIndex] += dOurFood
            lastEnemyPosition = gameState.getAgentPosition(lastAgentIndex)
            if lastEnemyPosition == None:
                lastEnemyPosition = self.myPf.getBeliefDistribution().argMax()
            if d2lPosReciever(lastAgentIndex, gameState, lastEnemyPosition) < 0:
                self.foodHoldings[lastAgentIndex] = 0
                
        #RUU
        
        self.myPf.elapseTime(gameState)
        self.myPf.observe(gameState.getAgentDistances()[(self.index - 1) % 4], gameState, self.index)
        
        #print "MYPOS: ", myPos
        #Agent is in left factory
        #if self.index in gameState.getBlueTeamIndices():
        if(x1 == 30):
            if(y1 != 1):
                #print "returning south"
                return Directions.SOUTH
            else:
                #print "returning west"
                return Directions.WEST
        elif(y1 == 1 and (x1 == 30 or x1 == 29)):
            #print "returning west"
            return Directions.WEST
        elif(x1 == 28 and (y1 == 1 or y1 == 2 or y1 == 3 or y1 == 4)):
            #print "returning North"
            return Directions.NORTH



        #elif self.index in gameState.getRedTeamIndicies():
        elif(x1 == 1):
            if(y1 != 14):
                #print "returning south"
                return Directions.NORTH
            else:
                #print "returning west"
                return Directions.EAST
        elif(y1 == 14) and (x1 == 1 or x1 == 2):
            return Directions.EAST
        elif(x1 == 3) and (y1 == 14 or y1 == 13 or y1 == 12 or y1 == 11):
            return Directions.SOUTH

        #print "my pos: ", myPos
        

        else:
            print "returning calced action"
            action = ApproximateQAgent.getAction1(self, gameState)
            
            

            originalState = gameState
            nextState = gameState.generateSuccessor(self.index, action)
            

            #nextState = gameState.generateSuccessor(self.index, action)
            if len(self.stateActionPairs) is not 0:
                self.update1(self.getLastState(), action, gameState, self.getReward(gameState))
            self.stateActionPairs.append((gameState, action))
            #self.doAction(gameState, action)
            #if self.index == 1:
                #print "WEIGHTS: ", self.weights
                #print "MY FEATURES: ", self.myFeats
            #print "Action: ", action
            dFood = len(self.getFood(originalState).asList()) - len(self.getFood(nextState).asList())
            self.foodHolding += dFood
            if distanceToLine(self.index, nextState) < 0:
                self.foodHolding = 0
            self.foodHoldings[self.index] = self.foodHolding
            print action
            return action
    def pfDistance(self, gameState, index1, index2):
        pos1 = gameState.getAgentPosition(index1)
        if pos1 == None:
            if index1 < 2:
                pfIndex = 0
            else:
                pfIndex = 1
            pos1 = self.pfilters[pfIndex].getBeliefDistribution().argMax()

        pos2 = gameState.getAgentPosition(index2)
        if pos2 == None:
            if index2 < 2:
                pfIndex = 0
            else:
                pfIndex = 1
            pos2 = self.pfilters[pfIndex].getBeliefDistribution().argMax()

        return self.distances.getDistance(pos1, pos2)
    def getLastState(self):
        if len(self.stateActionPairs) is not 0:
            return self.stateActionPairs[len(self.stateActionPairs)-1][0]

    def getLastAction(self):
        if len(self.stateActionPairs) is not 0:
            return self.stateActionPairs[len(self.stateActionPairs)-1][1]
    def getReward(self, state):
        #print "self.lastState: ",self.lastState

        if len(self.stateActionPairs) is not 0:
            reward = (state.getScore() - self.getLastState().getScore())*50

            if state.isOnRedTeam(self.index):
                print "ENEMY CURRENT FOOD: ",len(state.getBlueFood().asList()) 
                print "ENEMY PREV FOOD: ",len(self.getLastState().getBlueFood().asList())
                print len(state.getBlueFood().asList()) < len(self.getLastState().getBlueFood().asList())
                
                #if len(state.getBlueFood().asList()) < len(self.lastState.getBlueFood().asList()):
                #    reward += 10
                myAgentState = state.getAgentState(self.index)
                #reward += (20 - len(state.getBlueFood().asList())) * 10
                reward -= (20 - len(state.getRedFood().asList())) * 10
                

                reward += myAgentState.numCarrying
                

                





                killScore = 0





                enemyIndicies = state.getBlueTeamIndices()
                prevDistances = util.Counter()
                for enemy in enemyIndicies:
                    if self.getLastState().getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(self.getLastState().getAgentPosition(enemy), self.getLastState().getAgentPosition(self.index))
                newDistances = util.Counter()
                for enemy in enemyIndicies:
                    if state.getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(state.getAgentPosition(enemy), state.getAgentPosition(self.index))
                
                x1,y1 = state.getAgentPosition(self.index)

                halfway = state.data.layout.width/2
                if x1 < halfway:
                
                    for enemyIndex in enemyIndicies:
                        if prevDistances[enemyIndex] == 1:
                            if x1 != 30 and (newDistances[enemyIndex] > 2 or newDistances[enemyIndex] == 0):
                                #TODO: Give more score for killing enemy with food
                                killScore += 100
                    reward += killScore


            else:
                if len(util.matrixAsList(state.getRedFood())) < len(util.matrixAsList(self.getLastState().getRedFood())):
                    reward += 10
                killScore = 0
                enemyIndicies = state.getRedTeamIndices()
                prevDistances = util.Counter()
                for enemy in enemyIndicies:
                    if self.getLastState().getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(self.getLastState().getAgentPosition(enemy), self.getLastState().getAgentPosition(self.index))
                newDistances = util.Counter()
                for enemy in enemyIndicies:
                    if state.getAgentPosition(enemy) is not None:
                        prevDistances[enemy] = util.manhattanDistance(state.getAgentPosition(enemy), state.getAgentPosition(self.index))
                
                x1,y1 = state.getAgentPosition(self.index)

                halfway = state.data.layout.width/2
                if x1 > halfway:
                
                    for enemyIndex in enemyIndicies:
                        if oldDistances[enemyIndex] == 1:
                            if x1 != 30 and (newDistances[enemyIndex] > 2 or newDistances[enemyIndex] == 0):
                                #TODO: Give more score for killing enemy with food
                                killScore += 100
                    reward += killScore

            #reward -= 50

            print "REWARD: ", reward
            return reward


    
    def final(self, gameState):
        ApproximateQAgent.final(self, gameState)
        #ReinforcementAgent.final(self, gameState)
        self.stopEpisode()

        toBePickledFile = open(self.fileName, "wb")
        pickle.dump(self.weights, toBePickledFile)
        toBePickledFile.close()