import tensorflow as tf


class LossMLNormal(tf.keras.losses.Loss):
    """ The maximum-likelihood (ML) loss """
    # Last weight of ML loss
    w_last = 0

    def __init__(self, bg, tf_model, w, std=1):
        """ Sets log(det J) part of the ML loss to be used with weight 'w' in
        tensorflow model 'tf_model' of the BG 'bg' with
        normal prior of standard deviation 'std'.

        IMPORTANT NOTE: LossMLNormal(bg, tf_model, 0) has to be called if ML loss should
        no longer be used int tf_model and it has been used before. Otherwise the log(det J)
        part of the ML loss would be still applied.
        """
        self.std = std
        self.bg = bg
        self.model = tf_model

        # Currently the ML loss is implemented in that way, that the log(det J)
        # part of the loss is added using the tf.keras.Model.add_loss method and
        # the (0.5/std**2)*(F**2) part is added using call method.
        # The former part of the loss can not be removed later (as far as we
        # know), it can only be made zero by adding exactly the same loss but
        # with the opposite sign. Therefore if user wants to stop using ML loss,
        # Loss_ML_normal(bg, tf_model, 0) must be called on the boltzmann generator.
        # As we consider this error prone, the following info printing is used:
        w_add = w - self.__class__.w_last
        print(f"ML loss weight info: last: {self.__class__.w_last} added: {w_add} new: {w}")
        self.__class__.w_last = w

        if w_add:
            for layer in bg.layers:
                if hasattr(layer, 'log_det_Jxz'):
                    tf_model.add_loss((-1) * layer.log_det_Jxz * w_add)
        # Do not sum over batch and do not use any other loss reduction
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name="ML_loss")

    def call(self, z_true, z_pred):
        """ Returns the (0.5/std**2)*(F(x)**2) part of the ML loss """
        return (0.5 / (self.std**2)) * tf.reduce_sum(z_pred**2, axis=1)


class LossKL(tf.keras.losses.Loss):
    """ The Kullback–Leibler (KL) divergence loss """
    # Last weight of ML loss
    w_last = 0

    def __init__(self, bg, tf_model, w, energy_function, high_energy, max_energy, temperature=1.0):
        """ Sets log(det J) part of the KL loss to be used with weight 'w' in
        tensorflow model 'tf_model' of the BG 'bg'.

        IMPORTANT NOTE: LossKL(bg, tf_model, 0, None, None, None) has to be called if KL loss
        should no longer be used and it has been used before. Otherwise the -log(det J)
        part of the KL loss would be still applied in the tf_model.
        """
        self.energy_function = energy_function
        self.high_energy = high_energy
        self.max_energy = max_energy
        self.temperature = temperature  # kT

        from util import linlogcut
        self.linlogcut_function = linlogcut

        # See LossMLNormal for more info
        w_add = w - self.__class__.w_last
        print(f"KL loss weight info: last: {self.__class__.w_last} added: {w_add} new: {w}")
        self.__class__.w_last = w

        if w_add:
            for layer in bg.layers:
                if hasattr(layer, 'log_det_Jzx'):
                    tf_model.add_loss((-1) * layer.log_det_Jzx * w_add)
        # Do not sum over batch and do not use any other loss reduction
        super().__init__(reduction=tf.keras.losses.Reduction.NONE, name="KL_loss")

    def call(self, x_true, x_pred):
        """ Returns u(F(z)) part of the KL loss """
        # Compute dimensionless energy
        energy = self.energy_function(x_pred) / self.temperature
        # Apply linlogcut
        safe_energy = self.linlogcut_function(energy, self.high_energy, self.max_energy)
        return safe_energy
