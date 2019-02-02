import vision_functions as vis
import copy






def errorLoop(slope, up, length):

    global count_left
    global prev_area
    global slice_tot
    global last_try
    global slice_tmp
    global gm_no_tl

    difference_left = (vis.getTotalArea(count_left) - prev_area)

    # slice_tmp = copy.deepcopy(slice_tot)

    while (difference_left < (-prev_area/3) or (vis.getTotalArea(count_left) < 1000 and prev_area > 1000)):

        if last_try:
            return []

        slice_gm_tmp = copy.deepcopy(slice_gm)

        if slope < 20:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, slope, 25, 8)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, slope, -25, 8)
            slope = slope + 1

        elif up <= 5 and length <= 33:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, 25, up)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, -25, up)
            up = up + 1

        elif length <= 33:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, length, 0)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, -length, 0)
            length = length + 1

        else:
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, length, 0)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_topmost, 1, 0, -length, 0)
            slice_gm_tmp = drawLine(slice_gm_tmp, prev_top, 1, 5, 50, 8)

            last_try = True

        # vis.show_img(slice_gm_tmp)

        slice_gm_tmp, contours_tmp, hierarchy_tmp = processFrame(slice_gm_tmp)

        slice_tmp = copy.deepcopy(slice_tot)

        count_left = []
        gm_no_tl = []

        slice_tmp = drawFrame(contours_tmp, hierarchy_tmp, slice_gm_tmp, slice_tmp, (0, 255, 0), 0)
        print ("len " + str(len(count_left)))

        difference_left = (vis.getTotalArea(count_left) - prev_area)


    return slice_tmp