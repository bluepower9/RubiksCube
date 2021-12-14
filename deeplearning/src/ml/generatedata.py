from rubik.cube import Cube
import random
import argparse, time
from alive_progress import alive_bar

FILENAME = "./data/data1.txt"

COUNT = 1000000

#maps turn to cube function
movemap = {
    'U': Cube.U,
    'Ui': Cube.Ui,
    'D': Cube.D,
    'Di': Cube.Di,
    'L': Cube.L,
    'Li': Cube.Li,
    'R': Cube.R,
    'Ri': Cube.Ri,
    'F': Cube.F,
    'Fi': Cube.Fi,
    'B': Cube.B,
    'Bi': Cube.Bi
}

def generate_combination() -> tuple:
    '''
    Generates a random cube combination and records how far away from solved state it is.
    Utilizes piece face position (U, D, L, R, F, B) instead of colors to keep data color independent.
    '''

    c = Cube('UUUUUUUUULLLFFFRRRBBBLLLFFFRRRBBBLLLFFFRRRBBBDDDDDDDDD')
    
    #based on proof from 2014 that you only need 26 quarter turns to solve a rubkik's cube
    numMoves = random.randrange(1, 27)
    #keeps track of turns to make sure you do not surpass 2 turns of the same move.
    prevmove = ''
    prevcount = 0

    possibleMoves = list(movemap.keys())


    for i in range(numMoves):
        move = random.choice(possibleMoves)
        #if move is same as previous move and it has already done it twice, finds another move.
        while move == prevmove and prevcount == 2:
            move = random.choice(possibleMoves)
        
        #resets prevcount if it is a different move else increments
        if(move != prevmove):
            prevcount = 1
        else:
            prevcount += 1
        
        prevmove = move
        movemap[move](c)

    return c.flat_str(), numMoves



def save_data(data: list, name:str = FILENAME) -> bool:
    '''
    takes in a list of data with the cube string and number of turns from solved position separated by a tab.
    returns false if it fails and true if it successfully saves file.
    '''

    try:
        with open(name, 'w') as file:
            for cube, num in data:
                file.write(cube + '\t' +str(num) + '\n')
    except Exception as e:
        print(e)
        return False

    return True

def generate_data(count:int = COUNT, file:str = FILENAME) -> bool:
    '''
    generates data and saves it in a file.
    '''

    data = []

    print('generating data size: ' + str(count))
    with alive_bar(count) as bar:
        for i in range(count):
            data.append((generate_combination()))
            bar()

    
    print('saving data...')
    result = save_data(data, file)
    if result:
        print('successfully saved data.')
    else:
        print('Failed to save data.')
    
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-count', type=int, required=False)
    parser.add_argument('-file', type=str, required=False)

    args = parser.parse_args()

    count = COUNT
    file = FILENAME
    if args.count:
        count = args.count
    if args.file:
        file = args.file 

    generate_data(count, file)




