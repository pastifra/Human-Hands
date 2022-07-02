def custom_loss(y_true, y_pred):
  
  mse = tf.keras.losses.MeanSquaredError(reduction = "sum")
  predictions = tf.reshape(y_pred,(-1,7,7,5))
  exists_box = tf.expand_dims(y_true[...,0], 3)

  #BOX LOSS
  pred_box = exists_box*predictions[...,1:5]
  target_box = exists_box*y_true[...,1:5]

  epsilon = tf.fill(pred_box[..., 2:4].shape, 1e-6)
  wh_pred = tf.math.sign(pred_box[...,3:5]) * tf.math.sqrt(tf.math.abs(pred_box[...,3:5] + epsilon))
  wh_targ = tf.math.sqrt(target_box[...,3:5] + epsilon)

  xy_pred = pred_box[...,1:3]
  xy_true = target_box[...,1:3]

  final_pred_box = tf.concat([xy_pred,wh_pred], axis = 3)
  final_true_box = tf.concat([xy_true,wh_targ], axis = 3)
  box_loss = mse(tf.reshape(final_pred_box, (-1, final_pred_box.shape[-1])),tf.reshape(final_true_box, (-1, final_true_box.shape[-1])))
  

  #OBJECT LOSS
  pred_obj = predictions[...,0:1]
  true_obj = y_true[...,0:1]

  object_loss = mse(tf.reshape(exists_box*pred_obj, (-1, )), tf.reshape(exists_box*true_obj, (-1, )) )

  #NO OBJECT LOSS
  non_exists_box = 1 - exists_box
  no_object_loss = mse(tf.reshape(non_exists_box*pred_obj, (-1, )), tf.reshape(non_exists_box*true_obj, (-1, )))

  total_loss = 5*box_loss + object_loss + 0.5*no_object_loss
  return total_loss