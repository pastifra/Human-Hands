class DataGenerator(tf.keras.utils.Sequence):
    'Batch data generator for Keras Yolo implementation'
    def __init__(self, path_list, bboxes_list, batch_size=25, dim=(448,448,3),
                 divisions=7, shuffle=True):
        
        self.dim = dim # image dimension to feed the network
        self.batch_size = batch_size
        self.path_list = path_list # list of all the path to the images
        self.S = divisions # grid size
        self.bboxes_list = bboxes_list # list of all the bboxes of each image (len(path_list) = len(bboxes_list))
        self.shuffle = shuffle # if true shuffle the data to generate the batches -> More robust learning
        self.on_epoch_end() # triggered at beginning and end of each epoch
        self.cell_size = dim[0]/divisions # Number of pixels in width and height of each cell

    def __len__(self):
        'Denote the number of batches per epoch'
        return int(np.floor(len(self.path_list) / self.batch_size))

    def __getitem__(self, index):
      
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # Generate indexes of the batch to extract from the two lists

        X, Y = self.__data_generation(indexes)

        return X, Y

    def on_epoch_end(self):
        
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.path_list))
        if self.shuffle == True: # For more robust data
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size,self.S,self.S,5))

        batch_num = 0
        # Generate data
        for i in indexes:
          original_img = load_img(self.path_list[i])
          width, height = original_img.size
          # load the image with the required size and calculate scale factors
          image = load_img(self.path_list[i], target_size=(448, 448))
          scale_w = 448 / width 
          scale_h = 448 / height
          image = img_to_array(image)
          # scale pixel values to [0, 1]
          image = image.astype('float32')
          image /= 255.0
          y_img = np.zeros((self.S,self.S,5))
          for box in self.bboxes_list[i]:
            # Scale the bboxes into new format
            xleft = int(box[0] * scale_w)
            yleft = int(box[1] * scale_h)
            b_width = int(box[2] * scale_w)
            b_height = int(box[3] * scale_h)
            # Get center of bbox
            ox = xleft + b_width/2
            oy = yleft + b_height/2
            # Calculate the coordinates of the cell in the grid that contains the center 
            grid_col = trunc(ox/self.cell_size)
            grid_row = trunc(oy/self.cell_size) 
            # Calculate the coordinates of the center of the bbox w.r.t the associated cell; (0,0) top left and (1,1) bottom right corners of the cell
            ox_cell = (ox - (grid_col)*self.cell_size)/self.cell_size
            oy_cell = (oy - (grid_row)*self.cell_size)/self.cell_size
            # Calculate the width and height of the bbox in terms of cell size, a bbox of width 448/S(cell size) will have grid_width = 1
            grid_width = b_width/self.cell_size
            grid_heigth = b_height/self.cell_size
            # Put the results into y; 1 represent the probability of the class
            y = [1,ox_cell,oy_cell,grid_width,grid_heigth]
            y_img[grid_row][grid_col] = y

          # Store sample
          X[batch_num,] = image

          # Store grid
          Y[batch_num,] = y_img
        
          batch_num += 1

        return X, Y
