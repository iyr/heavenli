//#include "pch.h"
#include <Python.h>
#include <math.h>

/*
static char module_docstring[] =
"The module implements hard-coded animation curves for heavenLi in C";
static char animCurveBounce_docstring[] =
"Quadratic curve with a bounce (0-1)";
static char animCurve_docstring[] =
"Quadratic curve with no bounce (0-1)";
*/
PyObject* animCurveBounce_animUtils(PyObject *self, PyObject *o)
{
	float c = PyFloat_AsDouble(o);
	if (c >= 1.0) {
		return PyFloat_FromDouble(0);
	}
	else {
		return PyFloat_FromDouble(-3.0*pow((c - 0.14167) / 1.5, 2) + 1.02675926);
	}
}

PyObject* animCurve_animUtils(PyObject *self, PyObject *o)
{
	float c = PyFloat_AsDouble(o);
	return PyFloat_FromDouble(-2.25*pow(c / 1.5, 2) + 1.0);
}

static PyMethodDef animUtils_methods[] = {
	{ "animCurveBounce", (PyCFunction)animCurveBounce_animUtils, METH_O },
	{ "animCurve", (PyCFunction)animCurve_animUtils, METH_O },
	{ NULL, NULL, 0, NULL }
};

static PyModuleDef animUtils_module = {
	PyModuleDef_HEAD_INIT,
	"animUtils",
	"Some hard coded animation curves",
	0,
	animUtils_methods
};

PyMODINIT_FUNC PyInit_animUtils() {
	return PyModule_Create(&animUtils_module);
}

