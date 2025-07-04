import numpy as np
from sklearn import metrics as skmetrics


def lgb_MaF(preds, dtrain):
    # LightGBM custom macro F1 (currently unused)
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1, len(Y))
    Y_pre = np.argmax(preds, axis=0)
    return 'macro_f1', float(MaF(preds.shape[0], Y_pre, Y)), True


def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1, len(Y))
    Y_pre = np.argmax(preds, axis=0)
    return 'precision', float(Counter(Y == Y_pre)[True] / len(Y)), True


class Metrictor:
    def __init__(self, classNum):
        self.classNum = classNum
        self._reporter_ = {
            "MaF": self.MaF, "MiF": self.MiF, "skMiF": self.skMiF, "P@8": self.P8, "P@5": self.P5,
            "ACC": self.ACC, "LOSS": self.LOSS, "MaAUC": self.MaAUC, "MiAUC": self.MiAUC,
            "MaMCC": self.MaMCC, "MiMCC": self.MiMCC
        }

    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(" %s=%6.3f" % (mtc, v), end=';')
            res[mtc] = v
        print(end=end)
        return res

    def set_data(self, Y_prob_pre, Y, threshold=0.5, multi_label=True):
        self.Y_prob_pre = Y_prob_pre.copy()
        isONE = Y_prob_pre > threshold
        self.Y_pre = Y_prob_pre.copy()
        self.Y_pre[isONE], self.Y_pre[~isONE] = 1, 0
        self.Y = Y.copy()
        self.N = len(Y)

    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report) * 8 + 6
        print("=" * (lineLen // 2 - 6) + "FINAL RESULT" + "=" * (lineLen // 2 - 6))
        print("%6s" % ('-',) + "".join(["%8d" % i for i in report]))
        for i, res in enumerate(resList):
            print("%6s" % (rowName + '_' + str(i + 1)) + "".join(["%8.3f" % res[j] for j in report]))
        print("%6s" % ('MEAN') + "".join(["%8.3f" % (np.mean([res[i] for res in resList])) for i in report]))
        print("======" + "========" * len(report))

    def each_class_indictor_show(self, id2lab):
        id2lab = np.array(id2lab)
        TPi, FPi, TNi, FNi = _TPiFPiTNiFNi(self.classNum, self.Y_pre, self.Y)
        MCCi = fill_inf((TPi * TNi - FPi * FNi) / np.sqrt((TPi + FPi) * (TPi + FNi) * (TNi + FPi) * (TNi + FNi)))
        Pi = fill_inf(TPi / (TPi + FPi))
        Ri = fill_inf(TPi / (TPi + FNi))
        Fi = fill_inf(2 * Pi * Ri / (Pi + Ri))
        sortedIndex = np.argsort(id2lab)
        classRate = self.Y.sum(axis=0)[sortedIndex] / self.N
        id2lab, MCCi, Pi, Ri, Fi = id2lab[sortedIndex], MCCi[sortedIndex], Pi[sortedIndex], Ri[sortedIndex], Fi[sortedIndex]
        print("-" * 28 + "MACRO INDICATOR" + "-" * 28)
        print("%30s%8s%8s%8s%8s%8s" % (' ', 'rate', 'MCCi', 'Pi', 'Ri', 'Fi'))
        for i, c in enumerate(id2lab):
            print("%30s%8.2f%8.3f%8.3f%8.3f%8.3f" % (c, classRate[i], MCCi[i], Pi[i], Ri[i], Fi[i]))
        print("-" * 70)

    def MaF(self): return F1(self.classNum, self.Y_pre, self.Y, average='macro')
    def MiF(self): return F1(self.classNum, self.Y_pre, self.Y, average='micro')
    def skMiF(self): return skmetrics.f1_score(self.Y, self.Y_pre, average='micro')
    def ACC(self): return ACC(self.classNum, self.Y_pre, self.Y)
    def MaMCC(self): return MCC(self.classNum, self.Y_pre, self.Y, average='macro')
    def MiMCC(self): return MCC(self.classNum, self.Y_pre, self.Y, average='micro')
    def MaAUC(self): return AUC(self.classNum, self.Y_prob_pre, self.Y, average='macro')
    def MiAUC(self): return AUC(self.classNum, self.Y_prob_pre, self.Y, average='micro')
    def LOSS(self): return LOSS(self.Y_prob_pre, self.Y)
    def P5(self): return PrecisionInTop(self.Y_prob_pre, self.Y, n=5)
    def P8(self): return PrecisionInTop(self.Y_prob_pre, self.Y, n=8)


def _TPiFPiTNiFNi(classNum, Y_pre, Y):
    # True Positives, False Positives, True Negatives, False Negatives per class
    isValid = (Y.sum(axis=0) + Y_pre.sum(axis=0)) > 0
    Y, Y_pre = Y[:, isValid], Y_pre[:, isValid]
    TPi = np.array([Y_pre[:, i][Y[:, i] == 1].sum() for i in range(Y.shape[1])], dtype='float32')
    FPi = Y_pre.sum(axis=0) - TPi
    TNi = (1 ^ Y).sum(axis=0) - FPi
    FNi = Y.sum(axis=0) - TPi
    return TPi, FPi, TNi, FNi


def PrecisionInTop(Y_prob_pre, Y, n):
    # Precision@N for multi-label classification
    Y_pre = (1 - Y_prob_pre).argsort(axis=1)[:, :n]
    return sum([sum(y[yp]) for yp, y in zip(Y_pre, Y)]) / (len(Y) * n)


def RecallInTop(Y_prob_pre, Y, n):
    Y_pre = (1 - Y_prob_pre).argsort(axis=1)[:, :n]
    return np.mean([sum(y[yp]) / sum(y) for yp, y in zip(Y_pre, Y)])


def ACC(classNum, Y_pre, Y):
    TPi, FPi, TNi, FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    return (TPi.sum() + TNi.sum()) / (len(Y) * classNum)


from sklearn.metrics import roc_auc_score


def AUC(classNum, Y_prob_pre, Y, average='macro'):
    assert average in ['micro', 'macro'], "Average must be 'micro' or 'macro'"

    if average == 'micro':
        try:
            return roc_auc_score(Y.ravel(), Y_prob_pre.ravel())
        except ValueError:
            return 0.0
    else:
        auc_list = []
        for i in range(classNum):
            y_true = Y[:, i]
            y_prob = Y_prob_pre[:, i]
            if len(np.unique(y_true)) < 2:
                continue
            try:
                auc = roc_auc_score(y_true, y_prob)
                auc_list.append(auc)
            except ValueError:
                continue
        return np.mean(auc_list) if auc_list else 0.0


def MCC(classNum, Y_pre, Y, average='micro'):
    TPi, FPi, TNi, FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average == 'micro':
        TP, FP, TN, FN = TPi.sum(), FPi.sum(), TNi.sum(), FNi.sum()
        return fill_inf((TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    else:
        MCCi = fill_inf((TPi * TNi - FPi * FNi) / np.sqrt((TPi + FPi) * (TPi + FNi) * (TNi + FPi) * (TNi + FNi)))
        return MCCi.mean()


def Precision(classNum, Y_pre, Y, average='micro'):
    TPi, FPi, _, _ = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average == 'micro':
        return fill_inf(TPi.sum() / (TPi.sum() + FPi.sum()))
    else:
        Pi = fill_inf(TPi / (TPi + FPi))
        return Pi.mean()


def Recall(classNum, Y_pre, Y, average='micro'):
    TPi, _, _, FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average == 'micro':
        return fill_inf(TPi.sum() / (TPi.sum() + FNi.sum()))
    else:
        Ri = fill_inf(TPi / (TPi + FNi))
        return Ri.mean()


def F1(classNum, Y_pre, Y, average='micro'):
    if average == 'micro':
        MiP = Precision(classNum, Y_pre, Y, average='micro')
        MiR = Recall(classNum, Y_pre, Y, average='micro')
        return fill_inf(2 * MiP * MiR / (MiP + MiR))
    else:
        TPi, FPi, _, FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
        Pi = fill_inf(TPi / (TPi + FPi))
        Ri = fill_inf(TPi / (TPi + FNi))
        Fi = fill_inf(2 * Pi * Ri / (Pi + Ri))
        return Fi.mean()


def LOSS(Y_prob_pre, Y):
    # Binary cross-entropy loss for multi-label classification
    Y = Y.reshape(-1, 1)
    Y_prob_pre = Y_prob_pre.reshape(len(Y), -1)
    Y_prob_pre[Y_prob_pre > 0.99] -= 1e-3
    Y_prob_pre[Y_prob_pre < 0.01] += 1e-3
    lossArr = Y * np.log(Y_prob_pre) + (1 - Y) * np.log(1 - Y_prob_pre)
    return -lossArr.mean(axis=1).mean()


from collections.abc import Iterable

def fill_inf(x, v=0.0):
    # Replace NaNs and infs in x with value v
    if isinstance(x, Iterable):
        x = np.where(np.isinf(x) | np.isnan(x), v, x)
    else:
        x = v if np.isinf(x) or np.isnan(x) else x
    return x
