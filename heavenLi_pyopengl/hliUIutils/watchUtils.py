#
# utilities for checking whether the user
# cursor is contained in a given area
#

# Check if user is clicking in arbitrary polygon defined by list of tuples of points
def watchPolygon(cxgl, cygl, polygon, w2h, drawInfo):#, point):
    tmx = cxgl
    tmy = cygl

    for i in range(len(polygon)):
        tmx1 = polygon[i][0]
        tmy1 = polygon[i][1]
        if w2h < 1.0:
            tmy1 /= w2h
        polygon[i] = (tmx1, tmy1)

    if (drawInfo):
        for i in range(len(polygon)):
            tmx1 = polygon[i-1][0]
            tmy1 = polygon[i-1][1]
            tmx2 = polygon[i+0][0]
            tmy2 = polygon[i+0][1]

            drawPill(
                    tmx1, tmy1,
                    tmx2, tmy2,
                    0.002,
                    w2h,
                    (1.0, 0.0, 1.0, 1.0),
                    (1.0, 0.0, 1.0, 1.0)
                    )

    n = len(polygon)
    inside = False

    p1x,p1y = polygon[0]
    for i in range(n+1):
        p2x,p2y = polygon[i % n]
        if tmy > min(p1y,p2y):
            if tmy <= max(p1y,p2y):
                if tmx <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (tmy-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or tmx <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

# Check if user cursor is over a box
def watchBox(px, py, qx, qy, cxgl, cygl, w2h, drawInfo):
    col = (1.0, 0.0, 1.0, 1.0)
    if (drawInfo and abs(qx-px) > 0.0 and abs(qy-py)):
        drawPill(
                px, py,
                px, qy,
                0.002,
                w2h,
                col,
                col
                )
        drawPill(
                qx, py,
                qx, qy,
                0.002,
                w2h,
                col,
                col
                )
        drawPill(
                px, py,
                qx, py,
                0.002,
                w2h,
                col,
                col
                )
        drawPill(
                px, qy,
                qx, qy,
                0.002,
                w2h,
                col,
                col
                )

    withinX = False
    withinY = False

    if (px > qx) and (cxgl <= px) and (cxgl >= qx):
        withinX = True
    if (px < qx) and (cxgl >= px) and (cxgl <= qx):
        withinX = True
    if (py > qy) and (cygl <= py) and (cygl >= qy):
        withinY = True
    if (py < qy) and (cygl >= py) and (cygl <= qy):
        withinY = True

    if (withinX and withinY):
        return True
    else:
        return False

# Check if user is clicking in circle
def watchDot(px, py, pr, cxgl, cygl, w2h, drawInfo):
    if w2h < 1.0:
        px /= w2h
        py /= w2h
        pr /= w2h
    if (drawInfo and abs(pr) > 0.0):
        drawArch(
                px,
                py,
                pr-0.002,
                pr-0.002,
                0.0,
                360.0,
                0.002,
                w2h,
                (1.0, 0.0, 1.0, 1.0)
                )

    if (abs(pr) == 0.0):
        return False
    elif (pr >= hypot(cxgl-px, cygl-py)):
        return True
    else:
        return False
