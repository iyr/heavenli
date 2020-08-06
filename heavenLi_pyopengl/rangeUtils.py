from math import ceil, floor

# Maps a value from one range to another
def mapRanges(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

# Limits a value to a range
def constrain(val, min_val, max_val):
    if (val <= min_val):
        return min_val
    elif (val >= max_val):
        return max_val
    else:
        return val

# Used to safely index list objects with supercritical index values
def rollover(index, length):
    if (length <= 0):
        return 0
    elif (index >= 0) and (index < length):
        return index
    elif (index >= length):
        while (index >= length):
            index -= length
        return index
    elif (index < 0):
        while (index < 0):
            index += length
        return index

# Leaves just the fractional of a float (-1.0 to 1.0)
def normalizeCursor(prevC, curC):
    if (prevC > curC):
        return curC - ceil(curC)
    elif (prevC < curC):
        return curC - floor(curC)
    else:
        return 0.0

# Copy a dictionary, keeping only selected keys from a list of keys
def filterKeys(dictionary, keys):
    tmp = {}
    tmk = dictionary.keys()
    for i in range(len(keys)):
        if (keys[i] in tmk):
            tmp[keys[i]] = dictionary[keys[i]]

    return tmp
