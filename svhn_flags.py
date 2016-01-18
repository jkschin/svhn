import tensorflow as tf

tf.app.flags.DEFINE_string('data_dir', '/home/samuelchin/svhn/data',
							"""Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 29,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('test_dir', '/home/samuelchin/svhn/data/test',
                           """Directory where test data is held.""")
tf.app.flags.DEFINE_string('test_file', 'combined_1_wrong.bin',
                           """Name of test file.""")
tf.app.flags.DEFINE_string('eval_dir', '/home/samuelchin/svhn/tmp/svhn_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/samuelchin/svhn/tmp/svhn_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 29,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

tf.app.flags.DEFINE_string('train_dir', '/home/samuelchin/svhn/tmp/svhn_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")