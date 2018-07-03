import sys , time
import numpy as np
import json
class World(object):
    def __init__(self,mapfile):
        self.map = self.generateMap(mapfile)
        self.rows = len(self.map)
        self.cols = len(self.map[0])
        self.traps = [0,1,3,10]
        self.walls = [9]
        self.targets = [2]
        # print(self.legalstate)

        self.actionmap = {
            0:'up',
            1:'down',
            2:'left',
            3:'right'
        }
        items = self.actionmap.items()
        self.actionFromCommand = dict( zip( [ i[1] for i in items ] , [ i[0] for i in items ] ) )

        self.targetReward = 10.0
        self.trapReward = -1.0
        self.wallReward = -1.5
        self.norReward = -1.0
        self.map_to_state , self.state_to_map = self.init()
        self.stateNumber = len(self.map_to_state)
        self.legalstate = filter(lambda x:not ( x in self.traps or x in self.targets or x in self.walls )  , [ i for i in range(self.rows * self.cols)])
        
        self.legalstate = map(lambda x:self.map_to_state[x],self.legalstate)
        self.part = []
        self.showFlag = True
        self.oP = []
        # print(self.legalstate)

    def init(self):
        map_to_state = {}
        state_to_map = {}
        
        mapindex = 0
        stateindex = 0

        self.traps = []
        self.walls = []
        self.targets = []

        for i in range(self.rows) :
            for j in range(self.cols) :
                if self.map[i][j] != 'b' :
                    map_to_state[mapindex] = stateindex
                    state_to_map[stateindex] = mapindex
                    stateindex += 1
                
                if i == 0 or j == 0 or i == self.rows - 1 or j == self.cols - 1 :
                    self.walls.append(mapindex)
                if  self.map[i][j] == "-" :
                    self.traps.append(mapindex)
                elif self.map[i][j] == "g" :
                    self.targets.append(mapindex)
                elif self.map[i][j] == "b" :
                    self.walls.append(mapindex)
                mapindex += 1
        # print(self.traps,self.walls,self.targets)
        # print(len(self.walls))
        # print(self.walls,self.traps,self.targets)
        # exit(0)
        return map_to_state , state_to_map
        print(len(state_to_map))
        print(map_to_state)
        print(state_to_map)

    def generateMap(self,mapfile):
        # Map = [
        #     ['-','-','g','-'],
        #     ['.','.','.','.'],
        #     ['.','b','-','.'],
        #     ['.','.','.','.'],
        # ]
        # row = 10
        # size = 10
        # Map = [ ['.']*row for _ in range(size) ]
        # a = [ [np.random.randint(0,size),np.random.randint(0,row)] for _ in range(row) ]
        # for item in a :
        #     Map[item[0]][item[1]] = "-"
        # Map = [['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '-'], ['.', '.', '.', '.', '.', '.', '.', '.', '-', '.'], ['.', '.', '.', '.', '-', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '-', '.', '.', '.', '.', '-'], ['.', '.', '-', '.', '.', '.', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '-', '.', '.', '.'], ['-', '.', '.', '.', '.', '-', '.', '.', '.', '.'], ['.', '.', '.', '.', '.', '.', '.', '.', '-', '.']]
        # Map[0][5] = 'g'
        # Map[7][7] = 'b'
        with open(mapfile,'r') as f :
            Map = json.load(f,encoding="utf-8")
            Map = Map['map']
        # print(Map)
        # exit(0)
        # print a
        # print Map
        # exit(0)
        return Map
    
    def setPart(self,spx,length,spy,width):
        for i in range(spx,length):
            for j in range(spy,width):
                self.part.append(self.cols*i+j)

    def simulation(self,state,action,stayON=True):
        if state not in self.state_to_map :
            print("exception: illegal state")
            exit(0)
        if stayON and action == 0 :
            return [state,action,state,self.norReward,0]
        if stayON :
            action -= 1
        m_state = self.state_to_map[state]
        nextstate = self.__execute(m_state,action)
        c = 0
        r = 0.0
        if self.isLegalState(nextstate):
            c , r = self.getReward(nextstate)
        else:
            r = self.norReward
            c = 0
            nextstate = m_state
        if nextstate in self.walls :
            nextstate = m_state
        self.Print(nextstate,isshow=True)
        
        if stayON :
            action += 1

        return [state,action,self.map_to_state[nextstate],r,c]
    def __execute(self,state,action):
        prob = 1.0
        prob_a = 0.0
        prob_b = 0.0
        random_w = np.random.random()
        if action == 0 :
            nextstate = state - self.rows
            a = np.random.choice(3,1,p=[prob,prob_a,prob_b])[0]
            if a == 0 :
                nextstate = nextstate
            elif a == 1 :
                nextstate -= 1
            else:
                nextstate += 1
        elif action == 1 :
            nextstate = state + self.rows
            a = np.random.choice(3,1,p=[prob,prob_a,prob_b])[0]
            if a == 0 :
                nextstate = nextstate
            elif a == 1 :
                nextstate += 1
            else:
                nextstate -= 1
        elif action == 2 :
            nextstate = state - 1
            a = np.random.choice(3,1,p=[prob,prob_a,prob_b])[0]
            if a == 0 :
                nextstate = nextstate
            elif a == 1 :
                nextstate += self.rows
            else:
                nextstate -= self.rows
        elif action == 3 :
            nextstate = state + 1            
            a = np.random.choice(3,1,p=[prob,prob_a,prob_b])[0]
            if a == 0 :
                nextstate = nextstate
            elif a == 1 :
                nextstate -= self.rows
            else:
                nextstate += self.rows
        else:
            nextstate = state
        return nextstate
    
    def isTarget(self,state):
        if state < 0 or state >= self.rows * self.cols:
            return False
        state = self.state_to_map[state]
        row = state / self.cols
        col = state - row * self.cols
        if self.map[row][col] == 'g':
            return True
        return False
            

    def getReward(self,state):
        if state in self.targets :
            return (-1,self.targetReward)
        elif state in self.traps :
            return (-1,self.trapReward)
        elif state in self.walls :
            return (0,self.wallReward)
        else:
            return (0,self.norReward)

    def isLegalState(self,state):
        if state >= 0 and state <= self.rows * self.cols - 1 :
            return True
        else:
            return False
        
    def state_to_coordinate(self,state):
        state = self.state_to_map[state]
        if self.isLegalState(state) :
            row = state / self.cols
            col = state - self.cols * row
            return (row+1,col+1)
        else:
            return (None,None)

    def SuggestionForNextActions(self,state,stayON=True):
        # just support one target
        actions = []
        target = self.targets[0]
        targetCoordinate = self.state_to_coordinate(target)
        
        if targetCoordinate[0] > state[0] :
            actions.append(self.actionFromCommand['down'])
        elif targetCoordinate[0] < state[0] :
            actions.append(self.actionFromCommand['up'])

        if targetCoordinate[1] > state[1] :
            actions.append(self.actionFromCommand['right'])
        elif targetCoordinate[1] < state[1] :
            actions.append(self.actionFromCommand['left'])
        if stayON :
            for i in range(len(actions)) :
                actions[i] += 1
                
        return actions
        


    def Print(self,s,P=False,isshow=False):
        if not isshow or not self.showFlag :
            return
        if P == True :
            self.oP = s
        Map = []
        print("\r")
        for r in range(self.rows) :
            temp = []
            for c in range(self.cols) :
                sn = c + r * self.cols
                if sn == s :
                    # temp.append("+")
                    print("+"),
                elif sn in self.targets :
                    # temp.append("g")
                    print("g"),
                elif sn == self.oP :
                    print("S"),                
                elif sn in self.walls :
                    print("b"),
                elif sn in self.part :
                    print("u"),
                else:
                    # temp.append(".")
                    print("."),
            # Map.append(temp)
            print("")
        sys.stdout.flush()
            
        # print(Map)
        time.sleep(0.5)
        # exit(0)


if __name__ == '__main__' :
    world = World()
    print(world.cols)   
    state = 2
    print(world.state_to_coordinate(state))
    print(world.simulation(state,1))
    print(world.state_to_coordinate(state))
    
    print(world.state_to_coordinate(3))