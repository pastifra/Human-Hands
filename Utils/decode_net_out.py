#Function to transform the output of the network in the correct format
def decode_net_out(out):
  # IN : out [7x7x5]
  # RETURNS : bboxes [list] of the best training boxes
  cell_size = 448/7
  bboxes = []
  for i in range(0,7):
    for j in range(0,7):
      if out[i][j][0] > 0.7: #CONFIDENCE THRESHOLD (P of the cell)
        bbox = np.zeros((4))
        ox_cell = out[i][j][1]
        oy_cell = out[i][j][2]
        w_cell = out[i][j][3]
        h_cell = out[i][j][4]
        if(ox_cell != 0 and oy_cell !=0):
          ox = trunc(ox_cell*cell_size + cell_size*j)
          oy = trunc(oy_cell*cell_size + cell_size*i)
          w = w_cell*cell_size
          h = h_cell*cell_size
          lx = ox - w/2
          ly = oy - h/2
          bbox = [out[i][j][0], int(lx),int(ly),int(w),int(h)]
          bboxes.append(bbox)
  return bboxes
