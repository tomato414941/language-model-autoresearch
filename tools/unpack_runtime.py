"""Stream XZ through the liblzma already supplied by the pinned Ubuntu image."""

import argparse
import ctypes as c
import hashlib
import io
import json
from pathlib import PurePosixPath
import sys
import tarfile


class Stream(c.Structure):
    # Stable liblzma ABI, including its reserved fields: lzma/base.h.
    _fields_ = [('next_in',c.c_void_p),('avail_in',c.c_size_t),('total_in',c.c_uint64),
        ('next_out',c.c_void_p),('avail_out',c.c_size_t),('total_out',c.c_uint64),
        ('allocator',c.c_void_p),('internal',c.c_void_p),('reserved_ptrs',c.c_void_p*4),
        ('seek_pos',c.c_uint64),('reserved_int2',c.c_uint64),
        ('reserved_sizes',c.c_size_t*2),('reserved_enums',c.c_int*2)]


class Options(c.Structure):
    # Stable liblzma ABI: lzma/container.h. All reserved fields stay zero.
    _fields_ = [('flags',c.c_uint32),('threads',c.c_uint32),('block_size',c.c_uint64),
        ('timeout',c.c_uint32),('preset',c.c_uint32),('filters',c.c_void_p),
        ('check',c.c_int),('reserved_enums',c.c_int*3),('reserved_ints',c.c_uint32*4),
        ('memlimit_threading',c.c_uint64),('memlimit_stop',c.c_uint64),
        ('reserved_tail',c.c_uint64*2),('reserved_ptrs',c.c_void_p*4)]


class Reader(io.RawIOBase):
    def __init__(self, source, threads=4):
        super().__init__()
        self.active = False
        if not isinstance(threads,int) or threads <= 0:
            raise ValueError('Use a positive decoder thread count.')
        self.source = source
        self.input_eof = self.ended = False
        self.input_buffer = None
        self.stream = Stream()
        self.library = c.CDLL('liblzma.so.5')
        self.library.lzma_stream_decoder_mt.argtypes = [c.POINTER(Stream),c.POINTER(Options)]
        self.library.lzma_stream_decoder_mt.restype = c.c_int
        self.library.lzma_code.argtypes = [c.POINTER(Stream),c.c_int]
        self.library.lzma_code.restype = c.c_int
        self.library.lzma_end.argtypes = [c.POINTER(Stream)]
        self.library.lzma_end.restype = None
        self.library.lzma_version_string.restype = c.c_char_p
        self.version = self.library.lzma_version_string().decode()
        options = Options(flags=8,threads=threads,timeout=300,
                          memlimit_threading=2**30,memlimit_stop=2**30)
        result = self.library.lzma_stream_decoder_mt(c.byref(self.stream),c.byref(options))
        if result != 0:
            raise RuntimeError(f'XZ decoder initialization failed: {result}')
        self.active = True

    def readable(self):
        return True

    def readinto(self, target):
        if self.closed:
            raise ValueError('The XZ reader is closed.')
        if self.ended or not len(target):
            return 0
        output = (c.c_ubyte*len(target)).from_buffer(target)
        self.stream.next_out = c.addressof(output)
        self.stream.avail_out = len(target)
        while True:
            if self.stream.avail_in == 0 and not self.input_eof:
                block = self.source.read(1024*1024)
                if block:
                    self.input_buffer = c.create_string_buffer(block)
                    self.stream.next_in = c.addressof(self.input_buffer)
                    self.stream.avail_in = len(block)
                else:
                    self.input_eof = True
                    self.stream.next_in = None
            result = self.library.lzma_code(c.byref(self.stream),3 if self.input_eof else 0)
            written = len(target)-self.stream.avail_out
            if result == 1:
                self.ended = True
            elif result != 0:
                raise ValueError(f'Invalid, truncated or unsupported XZ stream: {result}')
            if written or self.ended:
                return written

    def close(self):
        if self.active:
            self.library.lzma_end(c.byref(self.stream))
            self.active = False
        super().close()


def tree_identity(source):
    rows = {}
    with tarfile.open(fileobj=source,mode='r|',bufsize=1024*1024) as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            if name == '.':
                continue
            row = dict(path=name,mode=member.mode)
            if member.isdir():
                row.update(type='directory')
            elif member.issym():
                row.update(type='symlink',target=member.linkname)
            elif member.islnk():
                target = rows[str(PurePosixPath(member.linkname))]
                row.update(type='file',bytes=target['bytes'],sha256=target['sha256'])
            elif member.isfile():
                digest = hashlib.sha256()
                with archive.extractfile(member) as contents:
                    for block in iter(lambda:contents.read(1024*1024),b''):
                        digest.update(block)
                row.update(type='file',bytes=member.size,sha256=digest.hexdigest())
            else:
                raise ValueError('Unsupported runtime file type.')
            rows[name] = row
    ordered = sorted(rows.values(),key=lambda row:PurePosixPath(row['path']))
    encoded = json.dumps(ordered,sort_keys=True,separators=(',',':')).encode()
    return dict(sha256=hashlib.sha256(encoded).hexdigest(),entries=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('copy','identity','extract'))
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--directory',default='.')
    args = parser.parse_args()
    result = None
    with Reader(sys.stdin.buffer,args.threads) as raw, io.BufferedReader(raw,buffer_size=1024*1024) as decoded:
        if args.action == 'identity':
            result = tree_identity(decoded)
        elif args.action == 'extract':
            with tarfile.open(fileobj=decoded,mode='r|',bufsize=1024*1024) as archive:
                archive.extractall(args.directory,filter='fully_trusted')
        while block := decoded.read(1024*1024):
            if args.action == 'copy':
                sys.stdout.buffer.write(block)
        if not raw.ended:
            raise ValueError('The XZ stream did not reach its checked end.')
    if result is not None:
        print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
