import tensorflow as tf

class Loss_ML_normal(tf.keras.losses.Loss):
    """ The maximum-likelihood (ML) loss """
    # Last weight of ML loss
    w_last = 0

    def __init__(self, bg, w, std=1):
        """ Sets log(det J) part of the ML loss to be used with weight 'w' in
        the BG 'bg' with normal prior of st. deviation 'std'.

        IMPORTANT NOTE: Loss_ML_normal(bg, 0) has to be called if ML loss should
        no longer be used and it has been used before. Otherwise the log(det J)
        part of the ML loss would be still applied. """
        self.std = std
        self.bg = bg

        # Currently the ML loss is implemented in that way, that the log(det J)
        # part of the loss is added using the tf.keras.Model.add_loss method and
        # the (0.5/std**2)*(F**2) part is added using call method.
        # The former part of the loss can not be removed later (as far as we
        # know), it can only be made zero by adding exactly the same loss but
        # with the opposite sign. Therefore if user wants to stop using ML loss,
        # Loss_ML_normal(bg, 0) must be called on the boltzmann generator.
        # As we consider this error prone, the following info printing is used:
        w_add = w - self.__class__.w_last
        print("ML loss wight info:", "last:" , self.__class__.w_last,
              "added:", w_add, "new:", w)
        self.__class__.w_last = w

        if w_add:
            for l in bg.layers:
                if hasattr(l, 'log_det_Jxz'):
                    bg.Txz.add_loss((-1) * l.log_det_Jxz * w_add)
        # Do not sum over batch and do not use any other loss reduction
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, z_true, z_pred):
        """ Returns the (0.5/std**2)*(F**2) part of the ML loss """
        return (0.5 / (self.std**2)) * tf.reduce_sum(z_pred**2, axis=1)
