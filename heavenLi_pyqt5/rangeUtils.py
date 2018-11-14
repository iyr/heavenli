def mapRanges(val, iMin, iMax, fMin, fMax):
    iRange = iMax - iMin
    fRange = fMax - fMax

    oVal = float(val - iMin) / float(iRange)
    oVal = fMin + (oVal*fRange)

    return oVal

def constrain(val, min_val, max_val):
    if (val <= min_val):
        return min_val
    elif (val >= max_val):
        return max_val
    else:
        return val
    #return min(max_val, max(min_val, val))

