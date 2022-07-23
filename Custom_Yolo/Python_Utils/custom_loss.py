def custom_loss(y_true, y_pred):
  
  # y_true (Batch size, 7, 7, 5)
  # y_pred (Batch size, 7, 7, 5)

  mse = tf.keras.losses.MeanSquaredError(reduction = "sum") # Define the SUM squared error loss
  predictions = tf.reshape(y_pred,(-1,7,7,5)) # The predictions are a tensor, need some reshaping to manipulate it

  exists_box = tf.expand_dims(y_true[...,0], 3) # A box exists if the first entry of the cell is equal to 1 

  #------------#
  #| BOX LOSS |#
  #------------#

  pred_box = exists_box*predictions[...,1:5] #Calculate only loss for the cells that contain a box
  target_box = exists_box*y_true[...,1:5] #Target boxes

  epsilon = tf.fill(tf.shape(pred_box[..., 2:4]), 1e-6) #Needed to avoid divergence of square root derivatives in back propagation

  # width and height are penalyzed using the square root, however predictions can be negative so multiply by sign in order to obtain positive
  # and take absoulte value in the square root 
  wh_pred = tf.math.sign(pred_box[...,3:5]) * tf.math.sqrt(tf.math.abs(pred_box[...,3:5] + epsilon))
  wh_targ = tf.math.sqrt(target_box[...,3:5] + epsilon)

  # Get also centers
  xy_pred = pred_box[...,1:3]
  xy_true = target_box[...,1:3]

  # Concatenate the new xy and wh in order to calculate sum squared root
  final_pred_box = tf.concat([xy_pred,wh_pred], axis = 3)
  final_true_box = tf.concat([xy_true,wh_targ], axis = 3)
  box_loss = mse(tf.reshape(final_pred_box, (-1, tf.shape(final_pred_box)[-1])),tf.reshape(final_true_box, (-1, tf.shape(final_true_box)[-1])))
  

  #---------------#
  #| OBJECT LOSS |#
  #---------------#
  
  # Take only the first entry of each box corresponding to the probability that there's an object
  pred_obj = predictions[...,0:1]
  true_obj = y_true[...,0:1]

  #Calculate object loss as in the paper
  object_loss = mse(tf.reshape(exists_box*pred_obj, (-1, )), tf.reshape(exists_box*true_obj, (-1, )) )

  #------------------#
  #| NO OBJECT LOSS |#
  #------------------#

  # Calculate the loss for cells that don't have objects
  non_exists_box = 1 - exists_box
  no_object_loss = mse(tf.reshape(non_exists_box*pred_obj, (-1, )), tf.reshape(non_exists_box*true_obj, (-1, )))

  #--------------#
  #| FINAL LOSS |#
  #--------------#

  # Penalize more the box loss and less the no object loss   
  total_loss = 5*box_loss + object_loss + 0.5*no_object_loss
  
  return total_loss
