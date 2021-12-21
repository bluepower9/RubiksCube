from pickle import FRAME
import torch
from models.LGF import LGF2d, LGF3d
from rubik.cube import FACE, Cube
from utils.Maps import moveMap
from utils.helper import convert_input, convert_one_input, convert_one_input2d, convert_one_input3d
from copy import deepcopy
from generatedata import generate_combination
import random, math

SOLVED_STATE = 'UUUUUUUUULLLFFFRRRBBBLLLFFFRRRBBBLLLFFFRRRBBBDDDDDDDDD'
WEIGHTS_PATH = './weights/weights4'

class ESCube:
    def __init__(self, state:str, model: LGF3d):
        self.cube = Cube(state)
        self.model = model
        self.move_list = []
        cubeinput = convert_one_input3d(state).unsqueeze(0)
        self.distance = model(cubeinput)[0].item()
        self.distance_list = [round(self.distance)]


    def turn(self, move) -> bool:
        moveMap[move](self.cube)
        self.move_list.append(move)
        cubestr = convert_one_input3d(self.cube.flat_str()).unsqueeze(0)
        d = self.model(cubestr)[0].item()

        result = d < self.distance
        self.distance = d
        self.distance_list.append(round(d))
        return result


    def random_turn(self) -> int:
        '''
        performs random turn on the cube.
        returns True if the new cube is predicted to be closer to solved and False otherwise.
        '''
        move = random.choice(list(moveMap.keys()))
        while (len(self.move_list) >= 2 and move == self.move_list[-1] and self.move_list[-1] == self.move_list[-2]) or (len(self.move_list) >= 1 and len(move) != len(self.move_list[-1]) and move[0] == self.move_list[-1][0]):
            move = random.choice(list(moveMap.keys()))

        moveMap[move](self.cube)
        self.move_list.append(move)
        cubestr = convert_one_input3d(self.cube.flat_str()).unsqueeze(0)
        d = self.model(cubestr)[0].item()

        result = d < self.distance
        self.distance = d
        self.distance_list.append(round(d))
        return d

    def solved(self) -> bool:
        '''
        returns True if the cube is solved, false otherwise.
        '''
        return self.cube.flat_str() == SOLVED_STATE


    def __str__(self) -> str:
        return self.cube.flat_str()



def es_solve(state, population_size = 50, mutation_rate = .5):
    '''
    Performs evolutionary strategy to solve cube.
    '''
    model = LGF3d()
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    population = [ESCube(state, model) for i in range(population_size)]
    solved = state == SOLVED_STATE


    variance = 0    #allows for this amount of worse states to be added to population
    turns = 0
    distance = round(model(convert_one_input3d(state).unsqueeze(0))[0].item())

    while not solved:# and turns < 1000:
        cubes = []
        moves = list(moveMap.keys())
        worse_cubes = []
        for cube in population:
            #performs random turn for cube and checks if it is better than the current expected distance
            d = cube.random_turn()
            if math.floor(d) <= distance:
                cubes.append(cube)
                
            else:
                worse_cubes.append(cube)

            #if solved return its move list
            if cube.solved():
                return cube.move_list

        best = min([x.distance for x in population])

        print('turns: ', turns, 'num better: ', len(cubes), 'distance: ', distance, 'best: ', best)

        if len(cubes) == 0:
            population = [ESCube(state, model) for i in range(population_size)]
            distance = round(model(convert_one_input3d(state).unsqueeze(0))[0].item())
        else:
            population = [deepcopy(random.choice(cubes)) for i in range(population_size)]
            #population = [deepcopy(random.choices(cubes, k = 1, weights=[1/(i+2) for i in range(len(cubes))])[0]) for i in range(population_size)]
            if best < distance:
                distance -= 1

        #randomly mutates (performs another move)
        for c in population:
            if random.random() < mutation_rate:
                c.random_turn()

        turns += 1
    
    for c in population:
        print(c.distance_list)


    return []   #returns empty list if no solution found

         
         


def solve(state):
    cube = Cube(state)
    move_list = []

    model = LGF3d()
    model.load_state_dict(torch.load(WEIGHTS_PATH))

    while cube.flat_str() != SOLVED_STATE and len(move_list) < 100:
        moves = {}
        for move in moveMap.keys():
            c = deepcopy(cube)
            moveMap[move](c)
            cubestr = convert_one_input3d(c.flat_str()).unsqueeze(0)
            out = model(cubestr)[0]
            moves[move] = out.item()


        best_move = min(moves.keys(), key = lambda x: moves[x]) #calculates best move found
        moveMap[best_move](cube)    #performs best move on real cube
        move_list.append(best_move)

    return move_list


            
if __name__ == '__main__':
    for i in range(100):
        state, num = generate_combination(10)
        moves = es_solve(state, mutation_rate=.5, population_size=100)
        print('state ' + str(i ) +': ', state, 'num moves: ', num, ' solved move count: ', len(moves))