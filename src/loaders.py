import os
import tensorflow as tf


def load_session(sess, checkpoint_path, variables):
    """
    Loads session from checkpoint.

    :param sess: Session in which to restore.
    :param checkpoint_path: Path to checkpoints.
    :param variables: Variables to restore from saved checkpoint.
    :return:
    """
    import re
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver = tf.train.Saver({v.op.name: v for v in variables})
        saver.restore(sess, os.path.join(checkpoint_path, ckpt_name))
        counter = int(
            next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print("Success to read {}".format(ckpt_name))
        return True, counter
    else :
        print("Failed to find a checkpoint")
        return False, 0


def load_attributes_vectors(path_to_file):
    # TODO
    pass