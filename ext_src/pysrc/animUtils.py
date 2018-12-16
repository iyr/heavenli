def animCurve(c):
    return -2.25*pow(float(c)/(1.5), 2) + 1.0

def animCurveBounce(c):
    if (c >= 1.0):
        return 0
    else:
        return -3.0*pow((float(c)-(0.14167))/(1.5), 2)+1.02675926

