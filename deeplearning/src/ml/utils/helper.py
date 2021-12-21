from .Maps import faceMap
import torch

def convert_input(input, intype = '1d') -> list:
        '''
        coverts input string list into array of ints to represent face values.
        '''
        result = []
        for value in input:
            if intype == '1d':
                result.append(convert_one_input(value))
            elif intype == '2d': 
                result.append(convert_one_input2d(value))
            elif intype == '3d':
                result.append(convert_one_input3d(value))

        return result

def convert_one_input(input) -> torch.FloatTensor:
    '''
    converts single input into a Tensor.
    '''
    arr = []
    for i in input:
        arr.append(faceMap[i])
    return torch.FloatTensor(arr)


def convert_one_input2d(input) -> torch.Tensor:
    arr = [[0 for i in range(6*9)] for i in range(6)]
    for i in range(len(input)):
        arr[faceMap[input[i]]][i] = 1
    return torch.tensor(arr, dtype=torch.float)



def convert_one_input3d(input) -> torch.Tensor:
    arr = [[[[0,0,0] for i in range(3)] for i in range(6)] for i in range(6)]

    for i in range(9):
        uPiece = input[i]
        dPiece = input[len(input)-i-1]
        row = i//3
        col = i%3
        arr[faceMap[uPiece]][0][row][col] = 1   #top of cube is 0 index in cube array
        arr[faceMap[dPiece]][-1][row][col] = 1  #bottom om cube is last index in cube array

    #handles intermediate faces
    for i in range(9, len(input) - 9):
        piece = input[i]
        face = (i-9)//3%4 + 1   #calculates which face it is on
        row = (i-9)//12
        col = (i-9)%3
        arr[faceMap[piece]][face][row][col] = 1


    return torch.tensor(arr, dtype=torch.float)


if __name__ == '__main__':
    arr = convert_one_input3d('UUUUUUUUULLLFFFRRRBBBLLLFFFRRRBBBLLLFFFRRRBBBDDDDDDDDD')
    for i in arr:
        for r in i:
            print(r)
        print()
    

