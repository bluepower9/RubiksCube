from torch.utils.data import Dataset
from utils.FaceMap import faceMap
import torch


class CubeDataSet(Dataset):
    def __init__(self, filename, device = 'cpu') -> None:
        data = self.read_data(filename)
        self.x = [i[0] for i in data]
        self.x = self.convert_input(self.x)
        self.y = [i[1]/26 for i in data]


    def convert_input(self, input) -> list:
        '''
        coverts input string list into array of ints to represent face values.
        '''
        result = []
        for value in input:
            arr = []
            for i in value:
                arr.append(faceMap[i])
            result.append(torch.FloatTensor(arr))

        return result


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


