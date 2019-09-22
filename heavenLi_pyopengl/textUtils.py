from fontUtils import *
import freetype

#def initChars(fontFile="fonts/Barlow-Regular.ttf"):
    #face = freetype.Face(fontFile)
    #face.set_char_size(48*64)

    #print("Generating font glyphs...")
    #for c in range(128):
        #face.load_char(chr(c))
        ##bitmap = face.glyph.bitmap

        ##print("Generating glyph for "+"\""+chr(c)+"\" "+"("+str(c)+")")
        #loadChar(chr(c), 
                #int(face.glyph.bitmap.width),
                #int(face.glyph.bitmap.rows),
                #int(face.glyph.bitmap_left),
                #int(face.glyph.bitmap_top),
                #int(face.glyph.linearHoriAdvance),
                #int(face.glyph.linearVertAdvance),
                #face.glyph.bitmap.buffer)
    #print("Done!")

    #return

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
