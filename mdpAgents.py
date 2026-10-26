from pacman import Directions
from game import Agent
import api
import random
import game
import util

class MDPAgent(Agent):

    def __init__(self):
        self.gamma = 0.85
        self.theta = 0.001
        self.maxEvalItns = 50
        self.maxPolicyItns = 50

        self.pelletReward = 50.0
        self.powerPelletReward = 100.0
        self.actionCost = -0.1
        self.ghostPenalty = -1000.0
        self.scaredGhostReward = 100.0

        self.walls = set()
        self.states = set()
        self.legalActionsMap = {}
        self.width = 0
        self.height = 0

        self.V = {}
        self.policy = {}

    def registerInitialState(self, state):
        self.walls = set(api.walls(state))
        
        topRight = max(api.corners(state))
        self.width = topRight[0] + 1
        self.height = topRight[1] + 1    
    
        self.states = {
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.walls
        }

        all_actions = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        self.legalActionsMap = {}
        for s in self.states:
            valid = [a for a in all_actions if self.getSuccessors(s, a) not in self.walls]
            self.legalActionsMap[s] = valid if valid else all_actions

        self.V = {s: 0.0 for s in self.states}
        self.policy = {s: self.legalActionsMap[s][0] for s in self.states}

    def getSuccessors(self, pos, action):
        x, y = pos
        if action == Directions.NORTH: return (x, y + 1)
        if action == Directions.SOUTH: return (x, y - 1)
        if action == Directions.EAST:  return (x + 1, y)
        if action == Directions.WEST:  return (x - 1, y)
        return pos

    def getTransitions(self, s, action):
        nonDetermOptions = {
            Directions.NORTH: [Directions.WEST, Directions.EAST],
            Directions.SOUTH: [Directions.EAST, Directions.WEST],
            Directions.EAST:  [Directions.NORTH, Directions.SOUTH],
            Directions.WEST:  [Directions.SOUTH, Directions.NORTH],
        }

        distribution = {}
        outcomes = [(action, 0.8)]
        for option in nonDetermOptions[action]:
            outcomes.append((option, 0.1))

        for direction, prob in outcomes:
            nextPosition = self.getSuccessors(s, direction)
            target = s if nextPosition in self.walls else nextPosition
            distribution[target] = distribution.get(target, 0.0) + prob

        return distribution
    
    def final(self, state):
        self.V = {}
        self.policy = {}

    def getMazeDistancesFrom(self, start_pos, max_dist=100):
        distances = {start_pos: 0}
        queue = [start_pos]
        while queue:
            curr = queue.pop(0)
            d = distances[curr]
            if d >= max_dist:
                continue
            x, y = curr
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nxt = (x + dx, y + dy)
                if nxt in self.states and nxt not in distances:
                    distances[nxt] = d + 1
                    queue.append(nxt)
        return distances

    def runPolicyIteration(self, R):
        for _ in range(self.maxPolicyItns):
            for _ in range(self.maxEvalItns):
                delta = 0.0
                newV = {}
                for s in self.states:
                    a = self.policy[s]
                    trans = self.getTransitions(s, a)
                    vVal = sum(
                        prob * (R[next_s] + self.gamma * self.V[next_s])
                        for next_s, prob in trans.items()
                    )
                    delta = max(delta, abs(vVal - self.V[s]))
                    newV[s] = vVal

                self.V = newV
                if delta < self.theta:
                    break

            policyStable = True
            for s in self.states:
                oldAction = self.policy[s]
                bestAction = oldAction
                bestQ = -float('inf')

                for a in self.legalActionsMap[s]:
                    trans = self.getTransitions(s, a)
                    q_val = sum(
                        prob * (R[next_s] + self.gamma * self.V[next_s])
                        for next_s, prob in trans.items()
                    )
                    if q_val > bestQ:
                        bestQ = q_val
                        bestAction = a

                self.policy[s] = bestAction
                if bestAction != oldAction:
                    policyStable = False

            if policyStable:
                break

    def buildRewardGrid(self, state):
        R = {s: self.actionCost for s in self.states}
        pacmanPos = api.whereAmI(state)

        food_list = api.food(state)
        capsules = set(api.capsules(state))
        pac_dists = self.getMazeDistancesFrom(pacmanPos, max_dist=100)

        for f in food_list:
            if f in R:
                d = pac_dists.get(f, 20)
                R[f] = self.pelletReward + (100.0 / (d + 1.0))

        for c in capsules:
            if c in R:
                d = pac_dists.get(c, 20)
                R[c] = self.powerPelletReward + (150.0 / (d + 1.0))

        for (gx, gy), scaredTimer in api.ghostStatesWithTimes(state):
            gPos = (int(round(gx)), int(round(gy)))
            if scaredTimer > 2:
                if gPos in R:
                    R[gPos] = self.scaredGhostReward
            else:
                if gPos in R:
                    if R[gPos] > 0:
                        R[gPos] = 0.0
                    R[gPos] += self.ghostPenalty
                x, y = gPos
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    adj = (x + dx, y + dy)
                    if adj in R and adj not in capsules:
                        if R[adj] > 0:
                            R[adj] = 0.0
                        R[adj] += self.ghostPenalty * 0.6

        return R

    def getAction(self, state):
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        R = self.buildRewardGrid(state)
        self.runPolicyIteration(R)

        pacmanPos = api.whereAmI(state)
        action = self.policy.get(pacmanPos, None)

        if action not in legal:
            action = random.choice(legal)

        return api.makeMove(action, legal)