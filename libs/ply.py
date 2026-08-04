"""Minimal PLY vertex I/O on numpy, without pulling in torch or pycolmap."""

import numpy as np

_PLY_TYPES = {
  "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
  "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
  "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
  "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}

def read_header(path):
  """Read a PLY header: (fields, vertex count, encoding), file left after the header."""
  with open(path, "rb") as f:
    return _read_header(f)

def _read_header(f):
  fields, count, encoding = [], 0, None
  while True:
    line = f.readline().decode("ascii").strip()
    if line.startswith("format"):
      encoding = line.split()[1]
    elif line.startswith("element vertex"):
      count = int(line.split()[2])
    elif line.startswith("property"):
      _, kind, name = line.split()
      fields.append((name, _PLY_TYPES[kind]))
    elif line == "end_header":
      return fields, count, encoding

def read_ply(path):
  """Read a binary little endian or ascii PLY vertex list as a numpy structured array."""
  with open(path, "rb") as f:
    fields, count, encoding = _read_header(f)
    if encoding == "ascii":
      return np.loadtxt(f, dtype=np.dtype(fields), max_rows=count)
    if encoding != "binary_little_endian":
      raise ValueError(f"Unsupported PLY encoding {encoding} in {path}")
    return np.frombuffer(f.read(count * np.dtype(fields).itemsize), dtype=np.dtype(fields), count=count)

def write_ply(path, columns):
  """Write float32 columns ({name: (N,) array}) as a binary little endian PLY."""
  names = list(columns)
  count = len(columns[names[0]])
  header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
  header += [f"property float {name}" for name in names]
  header.append("end_header")
  data = np.empty(count, dtype=np.dtype([(name, "f4") for name in names]))
  for name in names:
    data[name] = columns[name]
  with open(path, "wb") as f:
    f.write(("\n".join(header) + "\n").encode("ascii"))
    f.write(data.tobytes())
