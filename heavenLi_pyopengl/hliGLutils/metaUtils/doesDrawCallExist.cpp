using namespace std;

extern map<string, drawCall> drawCalls;

PyObject* doesDrawCallExist_hliGLutils(PyObject* self, PyObject* args) {
   PyObject* drawcallPyString;

   // Parse Inputs
   if ( !PyArg_ParseTuple(args,
            "O",
            &drawcallPyString
            ) )
   {
      Py_RETURN_NONE;
   }

   // Parse image name
   const char* NameChars = PyUnicode_AsUTF8(drawcallPyString);
   string NameString = NameChars;

   if (doesDrawCallExist(NameString))
      Py_RETURN_TRUE;
   else
      Py_RETURN_FALSE;

}

bool doesDrawCallExist(string drawcall){
   if (drawCalls.count(drawcall) > 0) return true;
   else return false;
}
