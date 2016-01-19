import tensorflow as tf
import os

"""
Directory Flags
"""
base_dir = "/home/samuelchin"
tf.app.flags.DEFINE_string('eval_dir', os.path.join(base_dir,'svhn/tmp/svhn_eval'),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('test_dir', os.path.join(base_dir,'svhn/data/test'),
                           """Directory where test data is held.""")
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(base_dir,'svhn/tmp/svhn_train'),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('data_dir', os.path.join(base_dir,'svhn/data/mstar'),
                            """Path to the SVHN data directory.""")
tf.app.flags.DEFINE_string('train_dir', os.path.join(base_dir,'svhn/tmp/svhn_train'),
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('predictions_dir', os.path.join(base_dir,'svhn/tmp/svhn_results'),
                            """Directory where results will be written""")
"""
Training Parameters
"""
tf.app.flags.DEFINE_integer('image_size', 50,
                            """Size of image.""")
tf.app.flags.DEFINE_integer('num_classes', 2,
                            """Number of classes.""")
tf.app.flags.DEFINE_integer('num_examples_per_epoch_for_train', 50000,
                            """Number of examples per epoch for training.""")
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999,
                            """The moving average decay.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 350.0,
                            """How many epochs to run after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                            """The learning rate decay factor.""")
tf.app.flags.DEFINE_float('learning_rate',0.1,
                            """Initial learning rate.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_channels', 1,
                            """Number of channels""")

"""
Test Parameters
"""
#Remember to change the batch.
tf.app.flags.DEFINE_string('test_file', 'mstar_test_batch.bin',
                           """Name of test file.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 3300,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")