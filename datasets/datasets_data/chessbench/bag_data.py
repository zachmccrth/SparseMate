
from numcodecs import zstd
from collections.abc import Sequence
import mmap
import os
import struct
from typing import Any, SupportsIndex

import torch.utils.data
import zstandard as zstd

from datasets import DatasetsAPI
from leela_interp.core.leela_board import LeelaBoard

from apache_beam import coders


CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
CODERS['state_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
))
CODERS['action_value'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))
CODERS['behavioral_cloning'] = coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
))



class BagFileReader(Sequence[bytes]):
  """Reader for single Bagz files."""

  def __init__(
      self,
      filename: str,
      *,
      separate_limits: bool = False,
      decompress: bool | None = None,
  ) -> None:
    """Creates a BagFileReader.

    Args:
      filename: The name of the single Bagz file to read.
      separate_limits: Whether the limits are stored in a separate file.
      decompress: Whether to decompress the records. If None, uses the file
        extension to determine whether to decompress.
    """
    if decompress or (decompress is None and filename.endswith('.bagz')):
      self._process = lambda x: zstd.decompress(x) if x else x
    else:
      self._process = lambda x: x
    self._filename = filename
    fd = os.open(filename, os.O_RDONLY)
    try:
      self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
      file_size = self._records.size()
    except ValueError:
      self._records = b''
      file_size = 0
    finally:
      os.close(fd)
    if separate_limits:
      directory, name = os.path.split(filename)
      fd = os.open(os.path.join(directory, 'limits.' + name), os.O_RDONLY)
      try:
        self._limits = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
        index_size = self._limits.size()
      except ValueError:
        self._limits = b''
        index_size = 0
      finally:
        os.close(fd)
      index_start = 0
    else:
      if 0 < file_size < 8:
        raise ValueError('Bagz file too small')
      self._limits = self._records
      if file_size:
        (index_start,) = struct.unpack('<Q', self._records[-8:])
      else:
        index_start = 0
      assert file_size >= index_start
      index_size = file_size - index_start
    assert index_size % 8 == 0
    self._num_records = index_size // 8
    self._limits_start = index_start

  def __len__(self) -> int:
    """Returns the number of records in the Bagz file."""
    return self._num_records

  def __getitem__(self, index: SupportsIndex) -> bytes:
    """Returns a record from the Bagz file."""
    i = index.__index__()
    if not 0 <= i < self._num_records:
      raise IndexError('bagz.BragReader index out of range')
    end = i * 8 + self._limits_start
    if i:
      rec_range = struct.unpack('<2q', self._limits[end - 8 : end + 8])
    else:
      rec_range = (0, *struct.unpack('<q', self._limits[end : end + 8]))
    return self._process(self._records[slice(*rec_range)])


@DatasetsAPI.register("ChessBenchDataset")
class ChessBenchDataset(torch.utils.data.Dataset):
    def __init__(self, file_path = "/home/zachary/PycharmProjects/SparseMate/datasets_data/chessbench/action_value-00000-of-02148_data.bag",):
        self.device = None
        self.file_path = file_path
        self.bag_reader = BagFileReader(file_path)

    def preprocess(self, fen):
        return LeelaBoard.from_fen(fen)

    def __len__(self):
        return len(self.bag_reader)

    def __getitem__(self, index):
        fen, _, _ = CODERS['action_value'].decode(self.bag_reader[index])

        return self.preprocess(fen)

    def __iter__(self):
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < len(self):
            item = self[self._current_index]
            self._current_index += 1
            return item
        else:
            raise StopIteration

