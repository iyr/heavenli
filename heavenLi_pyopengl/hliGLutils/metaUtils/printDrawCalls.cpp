using namespace std;

extern map<string, drawCall> drawCalls;
#include <iostream>

PyObject* printDrawCalls_hliGLutils(PyObject* self, PyObject* args){
   printDrawCalls();
   Py_RETURN_NONE;
}

void printDrawCalls(void){
   //map<string, drawCall>::iterator i;
   printf("%d current drawCall keys: \n", (unsigned int)drawCalls.size());
   for (auto i = drawCalls.cbegin(); i != drawCalls.cend(); i++)
      cout << (*i).first << "\n";
      //printf("%s\n", i->first.c_str());
   return;
}
