import torch, time, pickle
from torch import nn
from nnLayer import *
from metrics import *
import networkx as nx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseModel:
    def __init__(self):
        pass

    def train(self, dataClass, trainSize, batchSize, epoch,
              lr=0.001, weightDecay=0.0, stopRounds=10, threshold=0.2, earlyStop=10,
              savePath='model/MKHCNet', saveRounds=1, isHigherBetter=True, metrics="MiF", report=["ACC", "MiF"]):
        assert batchSize % trainSize == 0
        metrictor = Metrictor(dataClass.classNum)
        self.batchSize = batchSize
        self.stepCounter = 0
        self.stepUpdate = batchSize // trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum + trainSize - 1) // trainSize
        mtc, bestMtc, stopSteps = 0.0, 0.0, 0
        if dataClass.validSampleNum > 0:
            validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device)
        st = time.time()

        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X, Y = next(trainStream)
                loss = self._train_step(X, Y, optimizer)

                if stopRounds > 0 and (e * itersPerEpoch + i + 1) % stopRounds == 0:
                    self.to_eval_mode()
                    print("After iters %d: [train] loss= %.3f;" % (e * itersPerEpoch + i + 1, loss), end='')
                    if dataClass.validSampleNum > 0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X, Y)
                        print(' [valid] loss= %.3f;' % loss, end='')

                    restNum = ((itersPerEpoch - i - 1) + (epoch - e - 1) * itersPerEpoch) * trainSize
                    speed = (e * itersPerEpoch + i + 1) * trainSize / (time.time() - st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;" % (speed, restNum / speed))

            if dataClass.validSampleNum > 0 and (e + 1) % saveRounds == 0:
                self.to_eval_mode()
                print('========== Epoch:%5d ==========' % (e + 1))
                print('[Total Train]', end='')
                Y_pre, Y = self.calculate_y_prob_by_iterator(
                    dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                metrictor(report)

                print('[Total Valid]', end='')
                Y_pre, Y = self.calculate_y_prob_by_iterator(
                    dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y, threshold)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')

                if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
                    print('⬆️ Bingo!!! Better model found with %s: %.3f' % (metrics, mtc))
                    bestMtc = mtc
                    self.save("%s.pkl" % savePath, e + 1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps >= earlyStop:
                        print('No improvement on %s for %d steps, stopping training.' % (metrics, earlyStop))
                        break

        self.load("%s.pkl" % savePath, dataClass=dataClass)
        os.rename("%s.pkl" % savePath, "%s_%s.pkl" % (savePath, ("%.3lf" % bestMtc)[2:]))

        Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)

        Y_pre, Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y, threshold)

        print('================================')
        return res

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict.update({
                'trainIdList': dataClass.trainIdList,
                'validIdList': dataClass.validIdList,
                'testIdList': dataClass.testIdList,
                'nword2id': dataClass.nword2id,
                'tword2id': dataClass.tword2id,
                'id2nword': dataClass.id2nword,
                'id2tword': dataClass.id2tword,
                'icd2id': dataClass.id2icd,
                'id2icd': dataClass.icd2id
            })
        torch.save(stateDict, path)
        print('Model saved to "%s".' % path)

    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name], strict=False)
        if dataClass is not None:
            dataClass.trainIdList = parameters['trainIdList']
            dataClass.validIdList = parameters['validIdList']
            dataClass.testIdList = parameters['testIdList']
            dataClass.nword2id = parameters['nword2id']
            dataClass.tword2id = parameters['tword2id']
            dataClass.id2nword = parameters['id2nword']
            dataClass.id2tword = parameters['id2tword']
            dataClass.id2icd = parameters['id2icd']
            dataClass.icd2id = parameters['icd2id']
        print("%d epochs and %.3lf val Score model loaded." % (parameters['epochs'], parameters['bestMtc']))

    def calculate_y_prob(self, X):
        logits = self.calculate_y_logit(X)['y_logit']
        return torch.sigmoid(logits)

    def calculate_y(self, X, threshold=0.2):
        Y_prob = self.calculate_y_prob(X)
        return (Y_prob > threshold).float()

    def calculate_loss(self, X, Y):
        out = self.calculate_y_logit(X)
        loss = self.crition(out['y_logit'], Y)
        if 'loss' in out:
            loss += out['loss']
        return loss

    def calculate_indicator_by_iterator(self, dataStream, classNum, report, threshold):
        metrictor = Metrictor(classNum)
        Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y, threshold)
        return metrictor(report)

    def calculate_y_prob_by_iterator(self, dataStream):
        YArr, Y_preArr = [], []
        while True:
            try:
                X, Y = next(dataStream)
            except:
                break
            Y_pre = self.calculate_y_prob(X).cpu().numpy()
            Y = Y.cpu().numpy()
            Y_preArr.append(Y_pre)
            YArr.append(Y)
        return np.vstack(Y_preArr).astype('float32'), np.vstack(YArr).astype('int32')

    def calculate_y_by_iterator(self, dataStream, threshold=0.2):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return (Y_preArr > threshold).astype(int), YArr

    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        p = self.stepCounter >= self.stepUpdate
        if p:
            self.stepCounter = 0

        loss = self.calculate_loss(X, Y) / self.stepUpdate
        loss.backward()
        if p:
            nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20)
            optimizer.step()
            optimizer.zero_grad()
        return loss * self.stepUpdate


def reload_graph(file):
    with open(file, "rb") as f:
        G = pickle.load(f)
        assert isinstance(G, nx.DiGraph), "Reloaded graph is not a DiGraph"
    return G


def load_matrix(adjacency_matrix):
    """
    Convert numpy adjacency matrix to PyTorch tensors.
    Returns: (in_matrix, out_matrix)
    """
    in_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)
    out_matrix = torch.tensor(adjacency_matrix.T, dtype=torch.float32)
    return in_matrix, out_matrix


class MKHCNet(BaseModel):
    def __init__(self, classNum, embedding, labDescVec,
                 rnnHiddenSize=64, embDropout=0.4, attnList=[384],
                 hdnDropout=0.4, fcDropout=0.5, device=torch.device('cuda:0'),
                 useCircleLoss=False):

        self.embedding = TextEmbedding(embedding=embedding, dropout=embDropout, freeze=False).to(device)
        self.ehr = ehrrepresentation(feaSize=embedding.shape[1], hiddenSize=rnnHiddenSize, num_layers=1,dropout=0.5, bidirectional=True, name='TextLSTM').to(device)
        self.mamba = Mamba(input_dim=rnnHiddenSize * 2, hidden_dim=rnnHiddenSize * 2).to(device)
        self.Lnorm = LayerNorm(feaSize=rnnHiddenSize * 2, dropout=0.3).to(device)
        self.hpla = HPLA(inSize=rnnHiddenSize * 2, classNum=classNum,labSize=labDescVec.shape[1], hdnDropout=hdnDropout, attnList=attnList, labDescVec=labDescVec).to(device)
        self.graph_net = HGJK(input_dim=labDescVec.shape[1], time_step=2, in_matrix=in_matrix, out_matrix=out_matrix).to(device)
        self.contraNorm = ContraNorm(dim=labDescVec.shape[1], scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False,
                                     positive=False, identity=False).to(device)
        self.fastkan = FastKANLayer(input_dim=labDescVec.shape[1]).to(device)

        self.moduleList = nn.ModuleList([
            self.embedding, self.biLSTM, self.mamba,
            self.LNandDP, self.icdAttn, self.graph_net,
            self.contraNorm, self.fastkan
        ])

        self.crition = nn.MultiLabelSoftMarginLoss
        self.labDescVec = torch.tensor(labDescVec, dtype=torch.float32).to(device)
        self.classNum = classNum
        self.device = device
        self.hdnDropout = hdnDropout
        self.fcDropout = fcDropout

    def calculate_y_logit(self, input):
        x = input['noteArr']
        x = self.embedding(x)
        x = self.ehr(x)
        x = self.mamba(x)
        x = self.Lnorm(x)
        x = self.graph_net(x)
        x = self.contraNorm(x)
        x = self.hpla(x)
        x = self.fastkan(x)
        return {'y_logit': x}
