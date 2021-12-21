from torch.utils.data import Dataset
from utils.Maps import faceMap
from utils.helper import convert_input
import torch


class CubeDataSet(Dataset):
    def __init__(self, filename, inputtype = '1d', size = -1) -> None:
        self.maxsize = size
        data = self.read_data(filename)
        self.x = [i[0] for i in data]
        self.x = convert_input(self.x, intype = inputtype)
        self.y = [i[1] for i in data]   # /26 to normalize between 0 - 1



    def read_data(self, filename) -> list:
        '''
        reads the data from the given text file and parses it into List of form: [(str, int)]
        file should be in the form: str \\t int
        '''
        result = []
        count = 0
        with open(filename, 'r') as file:
            line = file.readline()
            while line and (self.maxsize == -1 or count < self.maxsize):
                parsedData = line.split('\t')
                result.append((parsedData[0], int(parsedData[1])))
                line = file.readline()
                count += 1

        return result

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple:
        return [self.x[idx], self.y[idx]]




class CubeDataSet2(Dataset):
    def __init__(self, filename, inputtype = '') -> None:
        data = self.read_data(filename)
        self.x = [i[0] for i in data]
        self.x = convert_input(self.x, intype='2d')
        self.y = [i[1] for i in data]   # /26 to normalize between 0 - 1



    def read_data(self, filename) -> list:
        '''
        reads the data from the given text file and parses it into List of form: [(str, int)]
        file should be in the form: str \\t int
        '''
        result = []
        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                parsedData = line.split('\t')
                result.append((parsedData[0], int(parsedData[1])))
                line = file.readline()

        return result

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> tuple:
        return [self.x[idx], self.y[idx]]