from hliGLutils import *
import freetype

def makeFont(fontFile="fonts/Barlow-Regular.ttf", numChars=128, size=48):
    face = freetype.Face(fontFile)
    face.set_char_size(size*64)
    Characters = []
    faceName=fontFile[6:20]
    print(faceName)
    print("Generating Glyph Set for "+faceName+" at size: "+str(size))

    class Character:
        def __init__(self):
            self.advanceX = 0
            self.advanceY = 0

            self.bearingX = 0
            self.bearingY = 0

            self.bearingTop = 0
            self.bearingLeft = 0

            self.bitmap = []
            self.binChar = 0

    for c in range(numChars):
        face.load_char(chr(c))
        fglyph = face.glyph
        cglyph = Character()
        cglyph.advanceX = fglyph.linearHoriAdvance
        cglyph.advanceY = fglyph.linearVertAdvance
        cglyph.bearingX = fglyph.bitmap.width
        cglyph.bearingY = fglyph.bitmap.rows
        cglyph.bearingTop = fglyph.bitmap_top
        cglyph.bitmapLeft = fglyph.bitmap_left
        cglyph.bitmap = fglyph.bitmap.buffer
        cglyph.binChar = c

        Characters.append(cglyph)

    buildAtlas(faceName, Characters, numChars)
