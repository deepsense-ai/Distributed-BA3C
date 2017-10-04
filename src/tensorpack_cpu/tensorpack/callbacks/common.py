# -*- coding: UTF-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import os, shutil
import re
from time import time

from .base import Callback
from ..utils import *
from ..tfutils.varmanip import get_savename_from_varname

__all__ = ['ModelSaver', 'MinSaver', 'MaxSaver']

class ModelSaver(Callback):
    """
    Save the model to logger directory.
    """
    def __init__(self, keep_recent=10, keep_freq=0.5,
            var_collections=tf.GraphKeys.GLOBAL_VARIABLES,
            models_dir=None):
        """
        :param keep_recent: see `tf.train.Saver` documentation.
        :param keep_freq: see `tf.train.Saver` documentation.
        """
        self.keep_recent = keep_recent
        self.keep_freq = keep_freq
        self.var_collections = var_collections
        self.models_dir = models_dir
        self.i = 0

    def _setup_graph(self):
        self.saver = tf.train.Saver(
            write_version = saver_pb2.SaverDef.V1,
            var_list=self.var_collections,
            max_to_keep=0)
        self.meta_graph_written = False
        self.start_time = time()
        # self.times_fd = open(os.path.join(self.models_dir, 'time.txt'))
        # self.i = 1

    @staticmethod
    def _get_var_dict(vars):
        var_dict = {}
        for v in vars:
            name = get_savename_from_varname(v.name)
            if name not in var_dict:
                if name != v.name:
                    logger.info(
                        "{} renamed to {} when saving model.".format(v.name, name))
                var_dict[name] = v
            else:
                logger.warn("Variable {} won't be saved \
because {} will be saved".format(v.name, var_dict[name].name))
        return var_dict

    def trigger_step(self):
        dir_path = os.path.join(self.models_dir, 'iter_{}'.format(self.i))
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except Exception:
                pass
        if os.listdir(dir_path) != []:
            return
        time_diff_h = (time() - self.start_time) / 3600.
        self.path = os.path.join(dir_path, 'model-{}'.format(time_diff_h))
        self.i += 1
        try:
            # if not self.meta_graph_written:
                # self.saver.export_meta_graph(
                        # os.path.join(logger.LOG_DIR,
                            # 'graph-{}.meta'.format(logger.get_time_str())),
                        # collection_list=self.graph.get_all_collection_keys())
                # self.meta_graph_written = True

            print("====== SAVING MODEL ========")
            print("path : {}".format(self.path))
            print("============================")

            self.saver.save(
                tf.get_default_session().get()._sess._sess._sess._sess,
                self.path,
                global_step=self.global_step,
                write_meta_graph=False)


            try:
                self.saver.save(
                    tf.get_default_session().get()._sess._sess._sess._sess,
                    self.path,
                    global_step=self.global_step,
                    write_meta_graph=False)
            except Exception:
                logger.warn("exception while saving parameters")
        except (OSError, IOError, TypeError):   # disk error sometimes.. just ignore it
            logger.exception("Exception in ModelSaver.trigger_epoch!")


class MinSaver(Callback):
    def __init__(self, monitor_stat, reverse=True, filename=None):
        self.monitor_stat = monitor_stat
        self.reverse = reverse
        self.filename = filename
        self.min = None

    def _get_stat(self):
        try:
            v = self.trainer.stat_holder.get_stat_now(self.monitor_stat)
        except KeyError:
            v = None
        return v

    def _need_save(self):
        v = self._get_stat()
        if not v:
            return False
        return v > self.min if self.reverse else v < self.min

    def _trigger_epoch(self):
        if self.min is None or self._need_save():
            self.min = self._get_stat()
            if self.min:
                self._save()

    def _save(self):
        ckpt = tf.train.get_checkpoint_state(logger.LOG_DIR)
        if ckpt is None:
            raise RuntimeError(
                "Cannot find a checkpoint state. Do you forget to use ModelSaver?")
        path = ckpt.model_checkpoint_path
        newname = os.path.join(logger.LOG_DIR,
                self.filename or
                ('max-' if self.reverse else 'min-' + self.monitor_stat + '.tfmodel'))
        shutil.copy(path, newname)
        logger.info("Model with {} '{}' saved.".format(
            'maximum' if self.reverse else 'minimum', self.monitor_stat))

class MaxSaver(MinSaver):
    def __init__(self, monitor_stat):
        super(MaxSaver, self).__init__(monitor_stat, True)



