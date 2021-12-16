from torch import full
from models.LGF import LGF
from data.CubeDataSet import CubeDataSet
from torch.utils.data import DataLoader, random_split
import torch

DATAPATH = './data/data1.txt'
SAVEPATH = './weights/weights1'
MODELTYPE = LGF
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data() -> tuple:
    '''
    loads data and returns tuple of size 3 with training, test, and full data
    '''

    print('Loading dataset...')

    dataset = CubeDataSet(DATAPATH)
    train, test = random_split(dataset, [round(len(dataset)*3/4), round(len(dataset)/4)])
    train_dl = DataLoader(train, batch_size = len(train)//100, shuffle = True,pin_memory=True)
    test_dl = DataLoader(test, batch_size = 1, shuffle = False)
    full_dl = DataLoader(dataset, batch_size = len(dataset), shuffle = True)
    
    print('Successfully loaded dataset.')

    return train_dl, test_dl, full_dl




if __name__ == '__main__':
    train_dl, test_dl, full_dl = load_data()
    model = MODELTYPE()

    model.to(DEVICE)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1000, gamma = 0.1)


    for epoch in range(1000000):
        total_loss = 0
        true_loss= 0
        avg_loss = 0
        for  i,(inputs, targets) in enumerate(train_dl):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            yhat = model(inputs).squeeze(-1)
            loss = criterion(yhat, targets.float())

            #calculate how many model got correct.
            for i in range(len(yhat)):
                if yhat[i] != targets[i]:
                    true_loss += 1
                avg_loss += abs(targets[i] - yhat[i])

            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch%10 == 0 or epoch == 0:
            print('saving model...')
            torch.save(model.state_dict(), SAVEPATH)

        #test = run_test(test_dl = test_dl, savepath = savepath, print_values=False, modeltype=MODELTYPE, input_size=input_size)
        #print('epoch: ', epoch, 'batch loss: ', total_loss, ' training data loss: ',true_loss*100/len(train_dl.dataset), 'avg loss: ', avg_loss.item()/len(train_dl.dataset))
        print('epoch: ', epoch, 'batch loss: ', total_loss, '\tloss: ', avg_loss.item()*100/len(train_dl.dataset))
        #scheduler.step()


