import sys

import gflags
import numpy as np
import tables


channels = {
    "MW": ["FP1"],
    "EP": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
    "MU": ["TP9", "FP1", "FP2", "TP10"],
    "IN": ["AF3", "AF4", "T7", "T8", "PZ"]
}

channel_ids = {k: {channel: id for id, channel in enumerate(channels_i)}
               for k, channels_i in channels.items()}


FLAGS = gflags.FLAGS

gflags.DEFINE_enum("device", "MW", channels.keys(), "")
gflags.DEFINE_string("out", "mbd.h5", "")
gflags.DEFINE_string("dtype", "int32", "")
gflags.DEFINE_integer("n_events", 0, "")
gflags.DEFINE_integer("seq_length", 0, "")


def main(argv):
    argv = FLAGS(argv)
    argv = argv[1:] # remove Python script name

    fname = argv[0]
    h5file = tables.open_file(FLAGS.out, "w")
    n_channels = len(channels[FLAGS.device])
    channel_map = channel_ids[FLAGS.device]

    out_x = h5file.create_carray(h5file.root, "X",
                                 atom=tables.Atom.from_type(FLAGS.dtype),
                                 shape=(FLAGS.n_events, n_channels, FLAGS.seq_length))
    out_len = h5file.create_carray(h5file.root, "lengths",
                                   atom=tables.Atom.from_type("int32"),
                                   shape=(FLAGS.n_events,))
    out_y = h5file.create_carray(h5file.root, "y",
                                 atom=tables.Atom.from_type("int32"),
                                 shape=(FLAGS.n_events,))

    with open(fname, "r") as data_f:
        for i, line in enumerate(data_f):
            if i % 1000 == 0:
                print i

            _, event, _, channel, y, seq_length, data = line.split()
            event, y, seq_length = int(event), int(y), int(seq_length)
            channel_id = channel_map[channel]

            data = np.fromstring(data, dtype=FLAGS.dtype, count=seq_length,
                                 sep=",")

            try:
              out_x[event, channel_id, :seq_length] = data
            except IndexError:
              print "\t", event
              continue
            out_len[event] = seq_length
            out_y[event] = y

    h5file.close()

if __name__ == "__main__":
    main(sys.argv)
