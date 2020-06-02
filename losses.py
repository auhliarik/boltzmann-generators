import tensorflow as tf

class Loss_ML_normal(tf.keras.losses.Loss):
    """ The ML loss """

    w_last = 0
    # TODO: the loss should be an average over batch, multiplying by
    # 1/N should be implemented + check signs!

    def __init__(self, bg, w, std=1):
        self.std = std
        self.bg = bg

        w_add = w - self.__class__.w_last
        print("w_last:", self.__class__.w_last)
        print("w_add:", w_add)
        self.__class__.w_last = w

        if w_add:
            for l in bg.layers:
                if hasattr(l, 'log_det_Jxz'):
                    bg.Txz.add_loss(l.log_det_Jxz * w_add)
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, z_true, z_pred):
        return 0 * (0.5 / (self.std**2)) * tf.reduce_sum(z_pred**2, axis=1)
