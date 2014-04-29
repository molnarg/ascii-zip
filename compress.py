#!/usr/bin/env python

import binascii
import zlib
import struct
import sys
import argparse


def binary(n, length, reverse=False):
    bits = ''.join(str(1 & (n >> i)) for i in range(length)[::-1])
    return bits if reverse else bits[::-1]


class WritableBitStream(object):
    def __init__(self):
        self.bits = ''

    def write(self, value, length=None, reverse=False):
        bits = value if (length is None) else binary(value, length, reverse)
        self.bits += bits

    def data(self):
        data = ''
        for cursor in range(0, len(self.bits), 8):
            bits = self.bits[cursor:cursor+8][::-1]
            #bits += '0' * (8 - len(bits))
            data += chr(int(bits, 2))
        return data


class ASCIICompressor(object):
    def __init__(self, allowed):
        self.allowed = map(lambda x: binary(ord(x), 8), allowed)
        self.stream = WritableBitStream()
        # self._test()
        self.block_count = 0
        # self._padding_block()

    def _test(self):
        decompressor = zlib.decompressobj()
        decompressor.decompress('\x08' + chr(31 - (0x08*256) % 31))
        print 'self test:',\
        repr(decompressor.decompress(self.stream.data()))
        # print 'self test flush:', \
        repr(decompressor.flush())

    def compress(self, uncompressed_data):
        data = uncompressed_data
        previous_block_type = 2
        while len(data) > 0:
            block_type = 2
            cursor = 1

            # Choose the longest possible chunk for the type 2 encoder
            distinct_bytes = {ord(data[0])}
            while (cursor < len(data) and
                   len(distinct_bytes) <= 50 and
                   max(distinct_bytes) < 216):
                distinct_bytes.add(ord(data[cursor]))
                cursor += 1
            if cursor != len(data):
                cursor -= 1

            # Reduce the chunk until the type 2 encoder can actually encode it
            while cursor > 0:
                huffman = self._generate_huffman_2(data[:cursor])
                if huffman is None:
                    cursor -= 1
                else:
                    break

            # If the type 1 encoder does better, then use that
            if cursor == 0:
                cursor += 1
            while cursor <= len(data):
                new_huffman = self._generate_huffman(data[:cursor])
                if new_huffman is None:
                    break
                else:
                    huffman = new_huffman
                    block_type = 1
                    cursor += 1
            if block_type == 1:
                cursor -= 1

            # Do the actual encoding with the calculated huffman
            chunk = data[:cursor]
            data = data[cursor:]
            self.block_count += 1
            print 'compress', self.block_count, repr(chunk), huffman
            if previous_block_type == 2:
                self._padding_block()
            (self._compress_chunk if block_type == 1 else self._compress_chunk_2)(
                chunk,
                huffman[0],
                huffman[1],
                (len(data) == 0)
            )
            previous_block_type = block_type

        print 'size:', len(self.stream.data())

        return self.stream.data(), uncompressed_data

    def _generate_huffman(self, data):
        # print '_generate_huffman', repr(data)
        first_valid_8bit_code = 0b00011100
        valid_codes = sorted(map(lambda c: int(c, 2), self.allowed))
        valid_codes = filter(lambda c: c >= first_valid_8bit_code, valid_codes)

        distinct_bytes = sorted(map(ord, list(set(data))))

        def assign_codes(symbols, codes, valid):
            #print symbols, codes
            if len(symbols) == len(codes):
                return codes
            prev_code = codes[-1]
            prev_symbol = symbols[len(codes)-1]
            symbol = symbols[len(codes)]

            max_code = prev_code + (symbol - prev_symbol)
            reachable_codes = filter(lambda c: c <= max_code, valid)
            for chosen_code in reachable_codes[::-1]:
                assigned_codes = assign_codes(
                    symbols,
                    codes + [chosen_code],
                    filter(lambda c: c > chosen_code, valid)
                )
                if assigned_codes:
                    return assigned_codes

        assigned_codes = assign_codes(
            [-1] + distinct_bytes,
            [first_valid_8bit_code - 1],
            valid_codes
        )
        if not assigned_codes:
            return None
        assigned_codes = assigned_codes[1:]
        symbols = dict(zip(distinct_bytes, assigned_codes))
        symbols[256] = 0b000011

        needed_6 = 3  # plus the end of block symbol and 3 after that
        needed_8 = assigned_codes[0] - first_valid_8bit_code
        code_lengths = []
        while len(code_lengths) < 257 or needed_6 or needed_8:
            if len(distinct_bytes) > 0 and len(code_lengths) == distinct_bytes[0]:
                assert needed_8 == 0
                code_lengths.append(8)
                this_code = assigned_codes.pop(0)
                distinct_bytes.pop(0)
                if len(assigned_codes) > 0:
                    needed_8 = assigned_codes[0] - this_code - 1
                else:
                    needed_8 = 228 - code_lengths.count(8)
            elif len(code_lengths) == 256:
                if needed_6 > 0:
                    return None
                else:
                    code_lengths.append(6)
                    needed_6 = 3
            elif needed_8 > 0:
                code_lengths.append(8)
                needed_8 -= 1
            elif needed_6 > 0:
                code_lengths.append(6)
                needed_6 -= 1
            else:
                code_lengths.append(0)

        assert ((pow(2, 6) - code_lengths.count(6))*4 - code_lengths.count(8)) == 0
        return code_lengths, symbols

    def _generate_huffman_2(self, data):
        print '_generate_huffman_2', repr(data)
        valid_codes = filter(
            lambda c: (binary(c & 0b00111111, 6, True) + '10') in self.allowed,
            range(0b10000000, 0b11000000)
        )
        first_valid_8bit_code = 0b10000100
        valid_codes = filter(lambda c: c >= first_valid_8bit_code, valid_codes)
        # print self.allowed, len(self.allowed), valid_codes, len(valid_codes)
        # for c in valid_codes:
        #     self.stream.write(c, 8, reverse=True)
        # print self.stream.data()

        # print valid_codes

        distinct_bytes = sorted(map(ord, list(set(data))))
        # print 'distinct bytes:', len(distinct_bytes), distinct_bytes

        def assign_codes(symbols, codes, valid):
            # print symbols, codes
            if len(symbols) == len(codes):
                return codes
            prev_code = codes[-1]
            prev_symbol = symbols[len(codes)-1]
            symbol = symbols[len(codes)]

            max_code = min(
                prev_code + (symbol - prev_symbol),  # max possible code
                valid[-(len(symbols) - len(codes))]  # leave space for others
            )
            reachable_codes = filter(lambda c: c <= max_code, valid)
            if symbol == ord(data[-1]):
                # The last char's code must be OK with 00 in the most
                # significant bits, since 00 is the end of block marker's code
                reachable_codes = filter(
                    lambda c: (binary(c, 8, True)[2:] + '00') in self.allowed,
                    reachable_codes
                )
            for chosen_code in reachable_codes[::-1]:
                assigned_codes = assign_codes(
                    symbols,
                    codes + [chosen_code],
                    filter(lambda c: c > chosen_code, valid)
                )
                if assigned_codes:
                    return assigned_codes

        assigned_codes = assign_codes(
            [-1] + distinct_bytes,
            [first_valid_8bit_code - 1],
            valid_codes
        )
        if not assigned_codes:
            return None
        assigned_codes = assigned_codes[1:]
        symbols = dict(zip(distinct_bytes, assigned_codes))
        symbols[256] = 0

        # A 257 legyen 2-es (00 kod)
        # Kell 1db 2-es a 257 utan (01 kod)
        # Kell 1db 6-s vhol (100001 kod)
        # Az utolso szimbolum vegzodjon 011-re (hogy utana lehessen 00 kod)
        needed_2 = 0
        needed_6 = 1
        needed_8 = assigned_codes[0] - first_valid_8bit_code
        code_lengths = []
        while len(code_lengths) < 257 or needed_2 or needed_6 or needed_8:
            if len(distinct_bytes) > 0 and len(code_lengths) == distinct_bytes[0]:
                assert needed_8 == 0
                code_lengths.append(8)
                this_code = assigned_codes.pop(0)
                distinct_bytes.pop(0)
                if len(assigned_codes) > 0:
                    needed_8 = assigned_codes[0] - this_code - 1
                else:
                    # 256 - (covered by 2s) - (covered by 6s) - (covered by 8s)
                    needed_8 = 256 - 64*2 - 4 - code_lengths.count(8)
            elif len(code_lengths) == 256:
                code_lengths.append(2)
                needed_2 = 1
            elif needed_8 > 0:
                code_lengths.append(8)
                needed_8 -= 1
            elif needed_6 > 0:
                code_lengths.append(6)
                needed_6 -= 1
            elif needed_2 > 0:
                code_lengths.append(2)
                needed_2 -= 1
            else:
                code_lengths.append(0)

        extra_codelengths = 257 - len(code_lengths)
        if 13 <= extra_codelengths <= 15 or extra_codelengths > 28:
            # HLIT would be invalid
            return None

        assert sum(map(lambda l: l and pow(2, 8-l), code_lengths)) == 256
        # sys.exit()
        return code_lengths, symbols

    def _padding_block(self):
        """Makes the next block start at (byte boundary - 2 bits)"""

        # Header
        self.stream.write(0, 1)   # Not last block
        self.stream.write(2, 2)   # Dynamic Huffman
        self.stream.write(8, 5)   # HLIT = 8
        self.stream.write(16, 5)  # HDIST = 16
        self.stream.write(9, 4)   # HCLEN = 9

        # Lengths Huffman table definition
        self.stream.write(2, 3)  # 16 length = 2
        self.stream.write(5, 3)  # 17 length = 5
        self.stream.write(0, 3)  # 18 length = 0
        self.stream.write(4, 3)  # 0  length = 4
        self.stream.write(3, 3)  # 8  length = 3
        self.stream.write(0, 3)  # 7  length = 0
        self.stream.write(6, 3)  # 9  length = 6
        self.stream.write(4, 3)  # 6  length = 4
        self.stream.write(4, 3)  # 10 length = 4
        self.stream.write(4, 3)  # 5  length = 4
        self.stream.write(4, 3)  # 11 length = 4
        self.stream.write(6, 3)  # 4  length = 6
        self.stream.write(2, 3)  # 12 length = 2

        # Liternal+length Huffman table definition
        def repeat(code, n):
            first = True
            while n > 0:
                # print n, len(self.stream.bits) % 8
                if n > 6 and not first and len(self.stream.bits) % 8 == 0:
                    x = min(n, 10)
                    self.stream.write('01', reverse=True)  # Huffman 16
                    self.stream.write(x-7, 2)  # Repeat 3-6x
                    self.stream.write('01', reverse=True)  # Huffman 16
                    self.stream.write(1, 2)  # Repeat 4x
                    n -= x
                else:
                    self.stream.write(code, reverse=True)
                    n -= 1
                first = False
        repeat('1010', 197)
        repeat('1100', 261 - 197)
        repeat('1010', 265 - 261) # TODO: Kell ez?

        # Distance Huffman table definition
        repeat('1010', 17)

        # Data
        self.stream.write('111011', reverse=True)  # End of Block

    overhead = 0

    def _compress_chunk(self, chunk, code_lengths, symbols, last):
        l = len(self.stream.bits)

        # Header
        self.stream.write(last, 1)                   # Is it the last block?
        self.stream.write(2, 2)                      # Dynamic Huffman
        self.stream.write(len(code_lengths)-257, 5)  # HLIT
        self.stream.write(25, 5)                     # HDIST = 25
        self.stream.write(9, 4)                      # HCLEN = 9

        # Lengths Huffman table definition
        self.stream.write(2, 3)  # 16 length = 2
        self.stream.write(4, 3)  # 17 length = 4
        self.stream.write(3, 3)  # 18 length = 3
        self.stream.write(4, 3)  # 0  length = 4
        self.stream.write(4, 3)  # 8  length = 4
        self.stream.write(5, 3)  # 7  length = 5
        self.stream.write(4, 3)  # 9  length = 4
        self.stream.write(4, 3)  # 6  length = 4
        self.stream.write(4, 3)  # 10 length = 4
        self.stream.write(0, 3)  # 5  length = 0
        self.stream.write(3, 3)  # 11 length = 3
        self.stream.write(5, 3)  # 4  length = 5
        self.stream.write(4, 3)  # 12 length = 4

        # Liternal+length Huffman table definition
        def repeat(code, n):
            first = True
            while n > 0:
                if n > 6 and not first and len(self.stream.bits) % 8 == 2:
                    x = n/6
                    for i in range(x):
                        self.stream.write('00', reverse=True)  # Huffman 16
                        self.stream.write(3, 2)  # Repeat previous 6x
                    n -= x*6
                else:
                    self.stream.write(code, reverse=True)
                    n -= 1
                first = False
        runs = []
        for code_length in code_lengths:
            if len(runs) > 0 and runs[-1][0] == code_length:
                runs[-1][1] += 1
            else:
                runs.append([code_length, 1])
        code_values = {
            0: '1000',
            6: '1001',
            8: '1010'
        }
        for run in runs:
            repeat(code_values[run[0]], run[1])
        # print runs

        # Distance Huffman table definition
        if len(self.stream.bits) % 8 == 2:
            self.stream.write('011', reverse=True)   # Huffman 18
            self.stream.write(11, 7)                 # Repeat zero (11+11)x
            self.stream.write('00', reverse=True)    # Huffman 16
            self.stream.write(1, 2)                  # Repeat previous 4x
        else:
            self.stream.write('1000', reverse=True)  # Huffman 0
            self.stream.write('011', reverse=True)   # Huffman 18
            self.stream.write(10, 7)                 # Repeat zero (11+10)x
            self.stream.write('00', reverse=True)    # Huffman 16
            self.stream.write(1, 2)                  # Repeat previous 4x

        # Data
        for byte in chunk:
            symbol = symbols[ord(byte)]
            # print byte, symbol
            self.stream.write(symbol, 8, reverse=True)
        self.stream.write(symbols[256], 6, reverse=True)

        overhead = (len(self.stream.bits) - l) / 8
        self.overhead += overhead
        # print overhead, float(overhead) / len(chunk)

    def _compress_chunk_2(self, chunk, code_lengths, symbols, last):
        # Header
        self.stream.write(last, 1)                   # Is it the last block?
        self.stream.write(2, 2)                      # Dynamic Huffman
        self.stream.write(len(code_lengths)-257, 5)  # HLIT
        self.stream.write(5, 5)                      # HDIST = 5
        self.stream.write(13, 4)                     # HCLEN = 13

        # Lengths Huffman table definition
        self.stream.write(2, 3)  # 16 length = 2
        self.stream.write(5, 3)  # 17 length = 5
        self.stream.write(3, 3)  # 18 length = 3
        self.stream.write(4, 3)  # 0  length = 4
        self.stream.write(4, 3)  # 8  length = 4
        self.stream.write(5, 3)  # 7  length = 5
        self.stream.write(4, 3)  # 9  length = 4
        self.stream.write(4, 3)  # 6  length = 4
        self.stream.write(4, 3)  # 10 length = 4
        self.stream.write(0, 3)  # 5  length = 0
        self.stream.write(3, 3)  # 11 length = 3
        self.stream.write(5, 3)  # 4  length = 5
        self.stream.write(0, 3)  # 12 length = 0
        self.stream.write(5, 3)  # 3  length = 5
        self.stream.write(0, 3)  # 13 length = 0
        self.stream.write(4, 3)  # 2  length = 4
        self.stream.write(0, 3)  # 14 length = 0

        # Liternal+length Huffman table definition
        def repeat(code, n):
            first = True
            while n > 0:
                if n > 6 and not first and len(self.stream.bits) % 8 == 2:
                    x = n/6
                    for i in range(x):
                        self.stream.write('00', reverse=True)  # Huffman 16
                        self.stream.write(3, 2)  # Repeat previous 6x
                    n -= x*6
                else:
                    self.stream.write(code, reverse=True)
                    n -= 1
                first = False
        runs = []
        for code_length in code_lengths:
            if len(runs) > 0 and runs[-1][0] == code_length:
                runs[-1][1] += 1
            else:
                runs.append([code_length, 1])
        code_values = {
            0: '1000',
            2: '1001',
            6: '1010',
            8: '1011',
        }
        for run in runs:
            repeat(code_values[run[0]], run[1])

        # Distance Huffman table definition
        if len(self.stream.bits) % 8 == 2:
            self.stream.write('1000', reverse=True)  # Huffman 0
            self.stream.write('1000', reverse=True)  # Huffman 0
            self.stream.write('00', reverse=True)    # Huffman 16
            self.stream.write(1, 2)                  # Repeat previous 4x
        else:
            self.stream.write('1001', reverse=True)  # Huffman 2
            self.stream.write('00', reverse=True)    # Huffman 16
            self.stream.write(0, 2)                  # Repeat previous 3x
            self.stream.write('1000', reverse=True)  # Huffman 0
            self.stream.write('1000', reverse=True)  # Huffman 0

        # Data
        for byte in chunk:
            symbol = symbols[ord(byte)]
            self.stream.write(symbol, 8, reverse=True)
        self.stream.write(symbols[256], 2, reverse=True)


parser = argparse.ArgumentParser()
parser.add_argument('filename',
                    type=str,
                    help='the file to compress')
parser.add_argument('--mode',
                    type=str,
                    choices=['raw', 'gzip', 'zlib', 'swf'],
                    help='format of the output data')
parser.add_argument('--output',
                    type=str,
                    default=None,
                    help='output file (the default is the standard output)')
args = parser.parse_args()

data = open(args.filename).read()
compressor = ASCIICompressor(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    # 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    # map(chr, range(1, 128))
    # map(chr, range(32,127))
 )

output = open(args.output, 'wb') if args.output else sys.stdout


def wrap_gzip(compressed):
    return (
        binascii.unhexlify(
            '1f8b' +      # Magic
            '08' +        # Compression Method
            '00' +        # Flags
            '00000000' +  # MTime
            '00' +        # Extra flags
            '99'          # OS
        ) +
        compressed +
        struct.pack('<L', zlib.crc32(data) % pow(2, 32)) +
        struct.pack('<L', len(data) % pow(2, 32))
    )


def wrap_zlib(compressed, data):
    return (
        'x\xda' +
        compressed +
        struct.pack('!L', zlib.adler32(data) % pow(2, 32))
    )

if args.mode == 'raw':
    compressed = compressor.compress(data)[0]
    print repr(compressed)
    output.write(compressed)

elif args.mode == 'gzip':
    output.write(wrap_gzip(compressor.compress(data)[0]))

elif args.mode == 'zlib':
    compressed, data = compressor.compress(data)
    print repr(compressed)
    output.write(wrap_zlib(compressed, data))

elif args.mode == 'swf':
    body = data[8:]
    compressed, body = compressor.compress(body)
    compressed = wrap_zlib(compressed, body)
    output.write(
        'CWS' +                                   # Signature
        data[4] +                                 # File version
        struct.pack('<L', len(compressed) + 8) +  # Entire file length
        compressed                                # Zlib compressed data
    )
