from fontUtils import *
import freetype

def initChars(fontFile):
    face = freetype.Face("fonts/Barlow-Regular.ttf")
    face.set_char_size(48*64)

    print("Generating font glyphs...")
    for c in range(128):
        face.load_char(chr(c))
        #bitmap = face.glyph.bitmap

        #print("Generating glyph for "+"\""+chr(c)+"\" "+"("+str(c)+")")
        loadChar(chr(c), 
                int(face.glyph.bitmap.width),
                int(face.glyph.bitmap.rows),
                int(face.glyph.bitmap_left),
                int(face.glyph.bitmap_top),
                int(face.glyph.linearHoriAdvance),
                face.glyph.bitmap.buffer)
    print("Done!")
    return
