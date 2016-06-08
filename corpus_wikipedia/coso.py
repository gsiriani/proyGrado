from __future__ import print_function
from codecs import open, BOM_UTF8
import tensorflow as tf

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

filename = "prueba.csv"

# setup text reader
file_length = file_len(filename)
filename_queue = tf.train.string_input_producer(["prueba.csv"])
reader = tf.TextLineReader()
_, csv_row = reader.read(filename_queue)

# setup CSV decoding
record_defaults = [["0"],["0"],["0"],["0"],["0"],[0],[1]]
col1,col2,col3,col4,col5,col6,col7 = tf.decode_csv(csv_row, record_defaults=record_defaults)

# turn features back into a tensor
features = tf.pack([col1,col2,col3,col4,col5])
res = tf.pack([col6,col7])

print("loading, " + str(file_length) + " line(s)\n")
with tf.Session() as sess:
  tf.initialize_all_variables().run()

  # start populating filename queue
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(file_length):
    # retrieve a single instance
    example, label = sess.run([features, res])
    print(example, label)

  coord.request_stop()
  coord.join(threads)
  print("\ndone loading")