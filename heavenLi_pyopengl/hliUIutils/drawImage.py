# Helper to simplify drawing images from disk 

def drawImage(
        imagePath,
        gx,
        gy,
        ao,
        scale,
        w2h,
        shape,
        color,
        refresh
        ):

    flat_arr_list = []
    xRes = 0
    yRes = 0

    # Avoid unneeded conversion computation
    if (    not doesDrawCallExist(imagePath)
            or
            refresh):
        img = Image.open(imagePath).convert('RGBA')
        arr = np.array(img)
        flat_arr = arr.ravel()
        flat_arr_list = flat_arr.tolist()
        xRes, yRes = img.size

    if (shape == "square"):
        drawImageSquare(
                imagePath,
                flat_arr_list,
                0.0, 0.0,
                0.0,
                0.75,
                xRes, yRes,
                w2h,
                color
                )
    elif (shape == "circle"):
        drawImageCircle(
                imagePath,
                flat_arr_list,
                0.0, 0.0,
                0.0,
                0.75,
                xRes, yRes,
                w2h,
                color
                )
    else:
        drawImageCircle(
                imagePath,
                flat_arr_list,
                0.0, 0.0,
                0.0,
                0.75,
                xRes, yRes,
                w2h,
                color
                )
    return

