import keras.backend as K
def custom_loss(y_true, y_pred):
  yes_obj=y_true[:,:,:,0:1]
  ''' 
  lxy=(((y_pred[:,:,:,1:2]-y_true[:,:,:,1:2])**2 + (y_pred[:,:,:,2:3]-y_true[:,:,:,2:3])**2)*yes_obj)
  lwh=(((y_pred[:,:,:,3:4]-y_true[:,:,:,3:4])**2 + (y_pred[:,:,:,4:5]-y_true[:,:,:,4:5])**2)*yes_obj)
  lp=(((y_true[:,:,:,0:1]-y_pred[:,:,:,0:1])**2)*yes_obj)
  '''
  yes_obj = y_true[...,0:1]
  p_pred = y_pred[...,0:1]

  xybox_pred = yes_obj*y_pred[...,:2]
  xybox_true = yes_obj*y_true[...,:2]
  
  epsilon = tf.fill(y_pred[..., 2:4].shape, 1e-6)
  whbox_pred = tf.math.sign(yes_obj*y_pred[..., 2:4]) * tf.math.sqrt(tf.math.abs(yes_obj*y_pred[..., 2:4] + epsilon))
  whbox_true = yes_obj*y_true[...,2:4]

  x_y_coordinate_loss = tf.keras.losses.MeanSquaredError(
          tf.reshape(xybox_pred, (-1, xybox_pred.shape[-1])),
          tf.reshape(xybox_true, (-1, xybox_true.shape[-1]))
      )
  w_h_coordinate_loss = tf.keras.losses.MeanSquaredError(
          
          tf.reshape(whbox_pred, (-1, whbox_pred.shape[-1])),
          tf.reshape(whbox_true, (-1, whbox_true.shape[-1]))
      )
  prob_loss = tf.keras.losses.MeanSquaredError(
          tf.reshape(p_pred, (-1, p_pred.shape[-1])),
          tf.reshape(yes_obj, (-1, yes_obj.shape[-1]))
      )
  
  total_loss = x_y_coordinate_loss + w_h_coordinate_loss + prob_loss
  
  return total_loss