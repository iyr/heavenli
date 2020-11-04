
# helper class for smoothly transitioning variable state changes
class UIparam:
    def __init__(self):
        self.currentVal     = 0.0   # True value of the parameter
        self.previousVal    = 0.0   # used for calculating frame-to-frame changes
        self.targetVal      = 0.0   # Target Value that current value will 'drift' to
        self.targetReached  = True  # True if current == target
        self.tDiff          = 1.0   # Time-slice for adjusting animation speed
        self.accel          = 1.0   # temporarily adjust tDiff until animation finishes
        self.curveBias      = 0.5
        self.prevDeltaSign  = None
        self.cursor         = 1.0
        self.curve          = "Default"
        self._easings = {
                "Default": pytweening.easeOutQuint,
                "easeInQuad": pytweening.easeInQuad,
                "easeOutQuad": pytweening.easeOutQuad,
                "easeInOutQuad": pytweening.easeInOutQuad,
                "easeInQuart": pytweening.easeInQuart,
                "easeOutQuart": pytweening.easeOutQuart,
                "easeInOutQuart": pytweening.easeInOutQuart,
                "easeInQuint": pytweening.easeInQuint,
                "easeOutQuint": pytweening.easeOutQuint,
                "easeInOutQuint": pytweening.easeInOutQuint,
                "easeInSine": pytweening.easeInSine,
                "easeOutSine": pytweening.easeOutSine,
                "easeInOutSine": pytweening.easeInOutSine,
                "easeInExpo": pytweening.easeInExpo,
                "easeOutExpo": pytweening.easeOutExpo,
                "easeInOutExpo": pytweening.easeInOutExpo,
                "easeInCirc": pytweening.easeInCirc,
                "easeOutCirc": pytweening.easeOutCirc,
                "easeInOutCirc": pytweening.easeInOutCirc,
                "easeInElastic": pytweening.easeInElastic,
                "easeOutElastic": pytweening.easeOutElastic,
                "easeInOutElastic": pytweening.easeInOutElastic,
                "easeInBack": pytweening.easeInBack,
                "easeOutBack": pytweening.easeOutBack,
                "easeInOutBack": pytweening.easeInOutBack,
                "easeInBounce": pytweening.easeInBounce,
                "easeOutBounce": pytweening.easeOutBounce,
                "easeInOutBounce": pytweening.easeInOutBounce,
                "easeInCubic": pytweening.easeInCubic,
                "easeOutCubic": pytweening.easeOutCubic,
                "easeInOutCubic": pytweening.easeInOutCubic
                }
        return

    # Returns true if values are animating
    def isAnimating(self):
        return not self.targetReached

    # Used for tuning animation speed
    def setTimeSlice(self, tDiff):
        self.tDiff = tDiff
        return

    # Returns whether the target value is reached
    def isTargetReached(self):
        return self.targetReached

    # Set Target Value for the Current Value to transition to
    def setTarget(self, target):
        return self.setTargetVal(target)

    def setTargetVal(self, target):
        self.cursor         = 0.0
        self.previousVal    = self.currentVal
        self.accel          = 1.0
        self.targetVal      = target
        self.targetReached  = False

        delta = self.targetVal - self.currentVal

        # Avoid divide by zero
        if (delta == 0.0 or delta == -0.0):
            deltaSign = 1.0
        else:
            deltaSign = delta/abs(delta)
        self.prevDeltaSign = deltaSign
        return

    # Target Value Getter
    def getTar(self):
        return self.targetVal

    # True Value of the parameter
    def getVal(self):
        return self.currentVal

    # Used for making real-time changes, corrections.
    # Use sparringly
    def setValue(self, value):
        self.currentVal = float(value)
        self.previousVal = float(value)
        return

    # Use to temporarily tune animation speed (multiplies tDiff)
    # Resets to 1.0 (default) after animation completes
    def setAccel(self, value):
        self.accel = float(value)
        return

    # Choose which animation curve to use
    # For a good refernce on different animation curves, visit:
    # https://easings.net/
    def setCurve(self, curve):
        self.curve = curve
        return

    # Calculate next val if current != taget
    def updateVal(self):
        if (self.cursor < 1.0):
            if (self.cursor < 0.0):
                self.cursor = 0.0
            delta           = self.targetVal - self.previousVal
            self.currentVal = delta*self._curve(self.cursor) + self.previousVal
            self.cursor     += self.tDiff*self.accel*0.25
        else:
            self.currentVal = self.targetVal
            self.cursor     = 1.0
            self.targetReached = True

        return

    # Calculate value of selected animation curve
    def _curve(self, x):
        return self._easings[self.curve](x)
