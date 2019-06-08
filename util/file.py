import os, random, pickle, re

TRAINING_PATH = "/content/AI4Good---Meza-OCR-Challenge"
#TRAINING_PATH = "../AI4Good---Meza-OCR-Challenge"

def my(*f):
  return os.path.join(PATH, *f)
def data_file(*f):
  return os.path.join(TRAINING_PATH, "cell_images", *f)
def train_file(*f):
  return data_file("training_set", *f)
def valid_file(*f):
  return data_file("validation_set", *f)

def parse(*filename):
  f = open(data_file(*filename))
  first = True
  line = next(f)
  assert line == "filename;value\n"
  out = {}
  for line in f:
    filename, value = line.strip().split(";")
    out[filename] = value
    if not re.match(r"-?[0-9]*[.,]?[0-9]*", value):
      print("bad value:", value)
      raise Exception()
  return out

def to_buffer(filename):
  i = PIL.Image.open(filename)
  a = np.array(i.getdata())
  return a.reshape((i.size[1], i.size[0], 3)).astype(np.float64) / 255

class FileHolder:
  def __init__(self):
    pass

  def _save(self, file="index"):
    f = open(file, "wb")
    pickle.dump(self.info, f)

  def _load(self, file="index"):
    "Load the index if it's not already loaded."
    if getattr(self, "info", None) is not None:
      return
    f = open(file, "rb")
    self.info = pickle.load(f)

  def make(self):
    "Rebuild the index."
    self.labeled = parse("training_set_values.txt")
    self.labeled_items = list(self.labeled.items())
    random.shuffle(self.labeled_items)
    split = int(len(self.labeled_items) * 0.9)
    self.info = {
      'training': self.labeled_items[:split],
      'validation': self.labeled_items[split:]
    }
    self.save()

  def random_training(self):
    """Picks a random training example and returns the filename and value.
    Example:

    fh = FileHolder()
    filename, value = fh.random_training()"""
    self._load()
    return random.choice(self.info['training'])

  def random_validation(self):
    """Picks a random validation example and returns the filename and value.
    Example:

    fh = FileHolder()
    filename, value = fh.random_validation()"""
    self._load()
    return random.choice(self.info['validation'])

  @staticmethod
  def rebuild():
    fh = FileHolder()
    fh.make()
