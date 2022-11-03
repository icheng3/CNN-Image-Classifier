import math

import layers_keras
import numpy as np
import tensorflow as tf

BatchNormalization = layers_keras.BatchNormalization
Dropout = layers_keras.Dropout


class Conv2D(layers_keras.Conv2D):
    """
    Manually applies filters using the appropriate filter size and stride size
    """

    def call(self, inputs, training=False):
        ## If it's training, revert to layers implementation since this can be non-differentiable
        if training:
            return super().call(inputs, training)

        ## Otherwise, manually compute convolution at inference.
        ## Doesn't have to be differentiable. YAY!
        bn, h_in, w_in, c_in = inputs.shape  ## Batch #, height, width, # channels in input
        c_out = self.filters                 ## channels in output
        fh, fw = self.kernel_size            ## filter height & width
        sh, sw = self.strides                ## filter stride

        # Cleaning padding input.
        if self.padding == "SAME":
            if (h_in % sh == 0):
                ph = max(fh - sh, 0)
            else:
                ph = max(fh - (h_in % sh), 0)
            if (w_in % sw == 0):
                pw = max(fw - sw, 0)
            else:
                pw = max(fw - (w_in % sw), 0)
            
            pad_top = ph // 2
            pad_bottom = ph - pad_top
            pad_left = pw // 2
            pad_right = pw - pad_left

            output_height = math.ceil(h_in / sh)
            output_width  = math.ceil(w_in / sw)

            paddings = tf.constant([[0,0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]])
            inputs = tf.pad(inputs, paddings,"CONSTANT")

        elif self.padding == "VALID":
            ph, pw = 0, 0
            output_height = math.ceil((h_in - fh + 1) / sh)
            output_width  = math.ceil((w_in - fw + 1) / sw)
            inputs = np.pad(inputs, ((0,), (int(ph),), (int(pw),), (0,)), mode='constant', constant_values=0.0)
            
        else:
            raise AssertionError(f"Illegal padding type {self.padding}")
        
        output_shape = (bn, output_height, output_width, c_out)
        output = np.zeros(output_shape)  

        for i in range(0, bn):
            for y in range(0, output_height):
                for x in range(0, output_width):
                    for k in range(0, c_out):
                        currInput = inputs[i, y*sh:y*sh+fh, x*sw:x*sw+fw, :]
                        kernel = self.kernel[:, :, :, k] 
                        output[i, y, x, k] = np.tensordot(currInput, kernel, ((0,1,2), (0,1,2)))
        
        return tf.convert_to_tensor(output, tf.float32)



        
