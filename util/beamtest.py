from nets.ctcnet import *
from util.beam import BeamSearch
from util.file import FileHolder
import matplotlib.pyplot as p

CHECKPOINTS = [
    "checkpoint-0049-0.49-CTCModel",
    "checkpoint-0046-0.43-CTCModel"
]

def show(inp):
    #print(outp[0])
    p.imshow((inp * 0.2 + 0.9).squeeze(0).permute(1, 2, 0))
    p.show()

class Test:
    def __init__(self, file, quiet=False):
        self.model = CTCModel().cuda()
        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint['model'])
        # set to evaluation mode!!!
        self.model.eval()
        self.fh = FileHolder()
        self.quiet = quiet

    def getone(self):
        inputs, outputs = self.fh.get_batch_tensor(1, validation=True)
        logits = self.model(inputs)
        logits = logits.detach().cpu().numpy()
        return inputs, outputs, logits

    def test(self, m):
        def fn(j):
            inputs, outputs, logits = self.getone()
            answer = beam(logits, 8)
            if answer[0].str() != outputs[0]:
                if not self.quiet:
                    print("%s image was wrong!" % nth(j))
                    print("answer", answer[0].str())
                    print("correct:", outputs)
                    print("top 3 guesses:", [z.str() for z in answer[:3]])
                    print("image:")
                    show(inputs)
                return 0
            else:
                return 1
        #count = sum(fn(j) for j in range(m)) / m
        count = sum(fn(j) for j in range(m))
        return count

SUFFIX = ["th", "st", "nd", "rd"] + ["th"] * 6
def nth(j):
    "Return j with an appropriate ordinal suffix. e.g. nth(3) == '3rd'"
    suffix = SUFFIX[j % 10]
    if 11 <= j < 19:
        suffix = "th"
    return "%d%s" % (j, suffix)

def beam(logits, nbeams):
    """Do a beam search on the logits.

    Logits should be shape (1, length, nchars)"""
    beas = BeamSearch(nbeams)
    for j in range(logits.shape[1]):
        beas.add_logit(logits[0, j, :])
    return beas.topbeams()

def test(m, quiet=False):
    count = Test(CHECKPOINTS[0], quiet).test(m)
    return "Total correct: %.2f%% (%d/%d)" % (count / m * 100, count, m)

def test2(m, quiet=False):
    count = Test(CHECKPOINTS[1], quiet).test(m)
    return "Total correct: %.2f%% (%d/%d)" % (count / m * 100, count, m)
