from hliGLutils import *
#import freetype
from freetype import *

def makeFont(fontFile="fonts/Barlow-Regular.ttf", numChars=128, size=48):
#def makeFont(fontFile="fonts/copperplatedecolightpdf.ttf", numChars=128, size=48):
    face = freetype.Face(fontFile)
    face.set_char_size(size*64)
    Characters = []
    faceName=fontFile[6:20]
    print(faceName)
    print("Generating Glyph Set for "+ faceName +" at size: "+ str(size))

    class Character:
        def __init__(self):
            self.advanceX = 0
            self.advanceY = 0

            self.bearingX = 0
            self.bearingY = 0

            self.bearingTop = 0
            self.bearingLeft = 0

            self.bitmap = []

    for c in range(numChars-32):
        face.load_char(chr(c+32), FT_LOAD_RENDER)
        fglyph = face.glyph
        cglyph = Character()
        cglyph.advanceX = fglyph.advance.x
        cglyph.advanceY = fglyph.advance.y
        cglyph.bearingX = fglyph.bitmap.width
        cglyph.bearingY = fglyph.bitmap.rows
        cglyph.bearingTop = fglyph.bitmap_top
        cglyph.bearingLeft = fglyph.bitmap_left

        #print(
                #"Wrapping Glyph Data for Character " + str(chr(c+32)) + 
                #", AdvanceX: " + str(cglyph.advanceX) + 
                #", AdvanceY: " + str(cglyph.advanceY) + 
                #", bearingX: " + str(cglyph.bearingX) + 
                #", bearingY: " + str(cglyph.bearingY) + 
                #", bearingTop: " + str(cglyph.bearingTop) + 
                #", bearingLeft: " + str(cglyph.bearingLeft)
                #)
        cglyph.bitmap = fglyph.bitmap.buffer
        cglyph.binChar = c+32

        Characters.append(cglyph)

    buildAtlas(faceName, Characters, numChars-32)
