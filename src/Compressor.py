from bitarray import bitarray

from Posting import Posting
from Record import Record


def compress_number_gamma(previous, number):
    if previous is None:
        value = number
    else:
        value = number - previous
    value_bin_str = "{0:b}".format(value)
    length = len(value_bin_str)
    length_bin_str = length * '1' + '0'
    bin_str = length_bin_str + value_bin_str
    return bitarray(bin_str)


def compress_number_vl(previous, number):
    if previous is None:
        value = number
    else:
        value = number - previous
    value_bin_str = "{0:b}".format(value)
    length = len(value_bin_str)
    zero_offset_length = 0 if length % 7 == 0 else 7 - length % 7
    value_bin_str = (zero_offset_length * '0') + value_bin_str
    byte_count = int(len(value_bin_str) // 7)
    code_str = ''
    for i in range(byte_count):
        first_bit = '0' if i != byte_count - 1 else '1'
        rest = value_bin_str[i * 7: i * 7 + 7]
        code_str += first_bit + rest
    return bitarray(code_str)


def compress_number(previous, number, is_gamma):
    if is_gamma:
        return compress_number_gamma(previous, number)
    return compress_number_vl(previous, number)


def compress_positions(positions, is_gamma):
    cpositions = []
    for i in range(len(positions)):
        if i == 0:
            previous = None
        else:
            previous = positions[i - 1]
        current = positions[i]
        cpositions.append(compress_number(previous, current, is_gamma))
    return cpositions


def compress_record(record, is_gamma):
    cposting = []
    for i in range(len(record.postings)):
        if i == 0:
            previous_id = None
        else:
            previous_id = record.postings[i - 1].doc_id
        current_id = record.postings[i].doc_id
        compressed_id = compress_number(previous_id, current_id, is_gamma)
        compressed_positions = compress_positions(record.postings[i].positions, is_gamma)
        cposting.append(Posting(compressed_id, compressed_positions))
    return Record(record.term, cposting)


def compress_index(index, is_gamma):
    cindex = {}
    for record in index.values():
        cindex[record.term] = compress_record(record, is_gamma)
    return cindex




def decompress_index(index):
    pass


if __name__ == "__main__":
    print("compress_number_vl(None, 214577)", compress_number_vl(None, 214577))
    print("compress_number_gamma(None, 5)", compress_number_gamma(None, 5))

    print("compress_positions([5, 5, 6, 7, 9], False)", compress_positions([5, 5, 6, 7, 9], False))
    print("compress_positions([5, 5, 6, 7, 9], True)", compress_positions([5, 5, 6, 7, 9], True))

    record = Record('first', [Posting(10, [1]), Posting(11, [1])])
    compressed_record = compress_record(record, True)
    compressed_record = compress_record(record, False)
