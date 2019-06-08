import os, random, pickle, re
import numpy as np
import PIL, skimage

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
def labeled_file(*f):
  return data_file("training_set", *f)
def unlabeled_file(*f):
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

  def get_batch(self, m, validation=False):
    inputs = []
    outputs = []
    if validation:
      fn = self.random_validation
    else:
      fn = self.random_training
    for i in range(m):
      file, val = fn()
      image = to_buffer(labeled_file(file))
      inputs.append(cleanup(image))
      outputs.append(val)
    return inputs, outputs

  @staticmethod
  def rebuild():
    fh = FileHolder()
    fh.make()

def to_buffer(filename, height=None):
  i = PIL.Image.open(filename)
  if height is not None:
    scale = 128 / i.size[1]
    i = i.resize((int(i.size[0] * scale), 128), resample=PIL.Image.BICUBIC)
  return pil_to_buffer(i)

def pil_to_buffer(i):
  a = np.array(i.getdata())
  return a.reshape((i.size[1], i.size[0], 3)).astype(np.float64)

def buffer_to_pil(i):
  return PIL.Image.fromarray(i, 'RGB')

def __clip(buf):
  grayscale = np.mean(buf, axis=2)
  flat = np.sort(grayscale.reshape(-1))
  top = flat[int(flat.shape[0] * 0.3)]
  bottom = flat[int(flat.shape[0] * 0.01)]
  buf = (buf - bottom) / (top - bottom)
  buf = np.clip(buf, 0, 1)
  return buf

def __crop(buf, crop_axis):
  assert crop_axis == 0 or crop_axis == 1
  CROP_DISTANCE = 10
  CROP_WHEN = 10/128
  "buf is close to 0 where there is ink"
  ink = 1 - buf
  if crop_axis == 0:
    a = np.mean(ink[:, :CROP_DISTANCE])
    b = np.mean(ink[:, -CROP_DISTANCE:])
    if a > CROP_WHEN:
      buf = buf[CROP_DISTANCE:, :]
    if b > CROP_WHEN:
      buf = buf[:-CROP_DISTANCE, :]
  else:
    a = np.mean(ink[:CROP_DISTANCE, :])
    b = np.mean(ink[-CROP_DISTANCE:, :])
    if a > CROP_WHEN:
      buf = buf[:, CROP_DISTANCE:]
    if b > CROP_WHEN:
      buf = buf[:, :-CROP_DISTANCE]
  return buf

def __median(buf, med_axis):
  assert med_axis == 0 or med_axis == 1
  buf = np.mean(buf, axis=2)
  buf = np.sum(1 - buf, axis = 1 - med_axis)
  total = np.cumsum(buf)
  total /= total[-1]
  return np.argmax(total > 0.5)

def __getpads(med, dim):
  from math import floor
  if med < dim / 2:
    return floor(dim - 2 * med), 0
  else:
    return 0, floor(2 * med - dim)

def __crop_whitespace(buf):
  grayscale = np.mean(buf, axis=(0, 2))
  left = np.argmax(grayscale < 0.99)
  right = grayscale.shape[0] - np.argmax(grayscale[::-1] < 0.99) - 1
  left = max(0, left - 32)
  right = min(grayscale.shape[0], right + 32)
  return buf[:, left:right, :]

SCALE_IMAGE_TO = 128
PAD_HORIZONTALLY_TO = 384

def cleanup(image, dontclip=False):
  if not dontclip:
    image = __clip(image)
  oldimage = image
  image = __crop(image, 0)
  image = __crop(image, 1)
  mediany = __median(image, 0)
  medianx = __median(image, 1)
  "If the median is less than image.shape[1] / 2, then pad on the left"
  pady = __getpads(mediany, image.shape[0])
  padx = __getpads(medianx, image.shape[1])
  image = skimage.util.pad(image, (pady, padx, (0, 0)),
      'constant',
      constant_values = 1
  )
  image = __crop_whitespace(image)
  scale = SCALE_IMAGE_TO / image.shape[0]
  image = skimage.transform.resize(image, (SCALE_IMAGE_TO, int(image.shape[1] * scale)))
  if image.shape[1] > PAD_HORIZONTALLY_TO:
    raise Exception("image too wide: %d" % image.shape[1])
  pad = (PAD_HORIZONTALLY_TO - image.shape[1]) // 2
  image = skimage.util.pad(image, ((0, 0), (PAD_HORIZONTALLY_TO - image.shape[1] - pad, pad), (0, 0)),
    'constant', constant_values = 1)
  return image

