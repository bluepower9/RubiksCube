from random import seed
from torch import full
from torch.serialization import save
from models.LGF import LGF, LGF2d, LGF3d
from data.CubeDataSet import CubeDataSet, CubeDataSet2
from torch.utils.data import DataLoader, random_split
import torch, sys
from argparse import ArgumentParser
from matplotlib import pyplot as plt


DATAPATH = './data/data3.txt'
SAVEPATH = './weights/weights3'
MODELTYPE = LGF3d
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 624

#sets seed
torch.manual_seed(SEED)



def load_data() -> tuple:
    '''
    loads data and returns tuple of size 3 with training, test, and full data
    '''

    print('Loading dataset...')

    dataset = CubeDataSet(DATAPATH, inputtype='3d', size=3000000)
    train, test = random_split(dataset, [len(dataset) - 50000, 50000])
    train_dl = DataLoader(train, batch_size = 32, shuffle = True,pin_memory=True)
    test_dl = DataLoader(test, batch_size = 1, shuffle = False)
    full_dl = DataLoader(dataset, batch_size = len(dataset), shuffle = True)
    
    print('Successfully loaded dataset.')

    return train_dl, test_dl, full_dl



def save_test(testresults) -> bool:
    '''
    saves test results to a file
    '''
    with open('./data/testresults.txt', 'w') as file:
        totalerror = sum([abs(testresults['predicted'][i] - testresults['expected'][i]) for i in range(len(testresults['predicted']))])
        avgitemloss =  totalerror / len(testresults['predicted'])
        file.write('RESULTS\n')
        file.write('-'*50)
        file.write('\nTOTAL ERROR: ' + str(round(totalerror,5 )) + '\t AVERAGE ERROR: ' + str(round(avgitemloss, 5)) + '\n')
        file.write('-' * 50 + '\n')
        for i in range(len(testresults['predicted'])):
            file.write('expected:  ' + str(testresults['expected'][i]) + '  \tpredicted:  ' + str(round(testresults['predicted'][i], 5)) + '\n')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-new', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-epochs', type=int, default=10000)
    parser = parser.parse_args()


    train_dl, test_dl, full_dl = load_data()
    model = MODELTYPE(training=not parser.test)
    if parser.test:
        print('performing test...')
        use_dl = test_dl
    else:
        use_dl = train_dl

    #default loads weights otherwise does not try to load weights.
    if not parser.new or parser.test:
        print('loading weights: ', SAVEPATH + '...')
        model.load_state_dict(torch.load(SAVEPATH))
        print('successfully loaded weights')

    model.to(DEVICE)


    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.1)

    best_loss = -1

    #sets epoch count
    epoch_count = parser.epochs
    if parser.test:
        epoch_count = 1


    for epoch in range(epoch_count):
        total_loss = []
        batchcount = []
        target_loss = 0 #how many it got incorrect
        avg_loss = 0    #used to find avg loss for each individual item
        
        testresults = {'expected': [], 'predicted': []}

        #iterates over data set
        for  i,(inputs, targets) in enumerate(use_dl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            yhat = model(inputs).squeeze(-1)
            loss = criterion(yhat, targets.float())

            #adds to testresults if testing
            if parser.test:
                testresults['expected'].append(targets[0].item())
                testresults['predicted'].append(yhat[0].item())
                

            #calculate how many model got correct.
            for i in range(len(yhat)):
                if round(yhat[i].item()) != round(targets[i].item()):
                    target_loss += 1
                avg_loss += abs(targets[i] - yhat[i])

            #gets data to graph and computate losses
            total_loss.append(loss.item())
            batchcount.append(i)            

            #backpropogation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #saves model if it improves
        if not parser.test and (best_loss == -1 or best_loss > sum(total_loss)):
            print('saving model...')
            torch.save(model.state_dict(), SAVEPATH)
            best_loss = sum(total_loss)

        if not parser.test:
            print('epoch: ', epoch, 'total loss: ', round(sum(total_loss), ndigits=5), 
            '\tavg batch loss: ', round(sum(total_loss)/len(train_dl), ndigits=5), 
            '\tavg item loss: ', round(avg_loss.item()/len(train_dl.dataset), ndigits=5),
            '\ttarget loss: ', round(target_loss*100/len(train_dl.dataset), ndigits=5))
        
        else:
            save_test(testresults)
            print('finished test. find results in ./data/testresults.txt')


