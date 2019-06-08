import os, random, pickle, re
import numpy as np
import PIL

TRAINING_PATH_POSSIBILITIES = [
  "/content/AI4Good---Meza-OCR-Challenge",
  "../AI4Good---Meza-OCR-Challenge"
]

TRAINING_PATH = None
for tp in TRAINING_PATH_POSSIBILITIES:
  if os.path.exists(tp):
    print("Using path", tp)
    TRAINING_PATH = tp
    break

if TRAINING_PATH is None:
  print("Couldn't find the data set anywhere.")
  print("")
  print("Possible paths:")
  for tp in TRAINING_PATH_POSSIBILITIES:
    print("  " + tp)
  print("Or add a new path to util/file.py.")

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
  def __init__(self, file="index"):
    self.file = file

  def _save(self):
    f = open(self.file, "wb")
    pickle.dump(self.info, f)

  def _load(self):
    "Load the index if it's not already loaded."
    if getattr(self, "info", None) is not None:
      return
    f = open(self.file, "rb")
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

def to_buffer(filename):
  i = PIL.Image.open(filename)
  a = np.array(i.getdata())
  return a.reshape((i.size[1], i.size[0], 3)).astype(np.float64) / 255
