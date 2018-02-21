import numpy as np

class State:
    def __init__(self,numDemons,y,x):
        self.NeighbourStates = ['up','down','left','right']
        self.Values = [0 for i in range(numDemons)]
        self.Occupant = [0]
        self.Visited = [False for i in range(numDemons)]
        self.index = [y,x]
        self.oneDIndex = 0
class Environment:
    djikstraQueue = []
    def __init__(self,n,m,numDemons):
        self.n = n
        self.m = m
        self.states = MakeMatrix(n,m,numDemons)
        self.oneDStates = self.flattenStates(self.states)
        self.demonStates = [self.states[0] for i in range(numDemons)]

    def flattenStates(self,states):
        output = []
        for indY, list in enumerate(states):
            for indX, state in enumerate(list):
                if(type(state) == State):
                    state.oneDIndex = len(output)
                    output.append(state)
        return output
    def resetStateVisited(self,states,demonId):
        for ind,val in enumerate(states):
            val.Visited[demonId] = False
    def demonFindState(self,demonId):
        return self.demonStates[demonId]
    def getActionSpace(self,demonId):
        curState = self.demonFindState(demonId)
        return curState.NeighbourStates
    def step(self,demonId,action):
        s = self.demonStates[demonId]
        s1 = s.NeighbourStates[action]
        self.demonStates[demonId] = s1
        a = s1.NeighbourStates
        d = s1.Values[demonId] > 0
        return
    def grassFire(self,givenState,demonId,discount):
        initState = givenState
        initState.Values[demonId] = 1
        initState.Visited[demonId] = True
        for ind,state in enumerate(initState.NeighbourStates):
            self.grassFireSpread(state,demonId,discount,False)
        self.resetStateVisited(self.oneDStates,demonId)
        initState.Visited[demonId] = True
        for ind,state in reversed(list(enumerate(initState.NeighbourStates))):
            self.grassFireSpread(state,demonId,discount,True)
    def grassFireSpread(self, state, demonId,discount,reverse):
        for ind, neigState in enumerate(state.NeighbourStates):
            state.Values[demonId] = max(state.Values[demonId],neigState.Values[demonId])
        state.Values[demonId] *= discount
        state.Visited[demonId] = True
        if reverse:
            for ind, neigState in reversed(list(enumerate(state.NeighbourStates))):
                if not neigState.Visited[demonId]:
                    self.grassFireSpread(neigState,demonId,discount,True)
        else:
            for ind, neigState in enumerate(state.NeighbourStates):
                if not neigState.Visited[demonId]:
                    self.grassFireSpread(neigState,demonId,discount,False)
    def djikstra(self,givenState,demonId,discount):
        self.djikstraQueue = [givenState]
        givenState.Values[demonId] = 1
        givenState.Visited[demonId] = True
        i = 0
        while(len(self.djikstraQueue) > 0 and i < 10000):
            i+=1
            self.djikstaCheck(self.djikstraQueue[0],demonId,discount)
    def djikstaCheck(self,givenState,demonId,discount):
        for ind, neigState in enumerate(givenState.NeighbourStates):
            if not neigState.Visited[demonId]:
                self.djikstraQueue.append(neigState)
                neigState.Values[demonId] = givenState.Values[demonId] * discount
                neigState.Visited[demonId] = True
            else:
                if givenState.Values[demonId] * discount > neigState.Values[demonId]:
                    neigState.Values[demonId] = givenState.Values[demonId] * discount
        self.djikstraQueue.remove(givenState)
    def findLowerRightSquareStateIndexes(self):
        output = []
        for ind, state in enumerate(self.oneDStates):
            if(state.index[0] > self.n/2 and state.index[1] > self.m/2):
                output.append(ind)
        return output


def MakeMatrix(n,m,numDemons):
    if(n%2 == 0):
        vertMid = (int)(n/2)
    else:
        vertMid = round(n/2)
    upDoor = (int)(vertMid - vertMid / 2)-1
    downDoor = (int)(vertMid + vertMid / 2)+1

    if(m%2 == 0):
        horMid = (int)(m/2)
    else:
        horMid = round(m/2)

    leftDoor = (int)(horMid - horMid/2)-1
    rightDoor = (int)(horMid + horMid/2)+1

    matrix = [[State(numDemons,j,i) if i != horMid else 0 for i in range(m)] if j != vertMid else [0 for i in range(m)] for j in range(n)]
    matrix[horMid][leftDoor] = State(numDemons,horMid,leftDoor); matrix[horMid][rightDoor] = State(numDemons,horMid,rightDoor); matrix[upDoor][vertMid] = State(numDemons,upDoor,vertMid); matrix[downDoor][vertMid] = State(numDemons,downDoor,vertMid)
    for indY, list in enumerate(matrix):
        for indX, state in enumerate(list):
            if (type(state) == State):
                if (indY - 1 >= 0 and type(matrix[indY - 1][indX]) == State):
                    state.NeighbourStates[0] = matrix[indY - 1][indX]
                if (indY + 1 != len(matrix) and type(matrix[indY + 1][indX]) == State):
                    state.NeighbourStates[1] = matrix[indY + 1][indX]
                if (indX - 1 >= 0 and type(matrix[indY][indX - 1]) == State):
                    state.NeighbourStates[2] = matrix[indY][indX - 1]
                if (indX + 1 != len(matrix[0]) and type(matrix[indY][indX + 1]) == State):
                    state.NeighbourStates[3] = matrix[indY][indX + 1]
                temp = state.NeighbourStates.copy()
                for ind,val in enumerate(temp):
                    if(type(val) != State):
                        state.NeighbourStates.remove(val)
    return matrix
