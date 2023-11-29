#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API

#include <boost/python.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include <AKAZE.h>

namespace libakaze_pybindings {

    using namespace libAKAZECU;
    using namespace boost::python;
    
    
    static void init_ar(){
	Py_Initialize();
	import_array();
    }

    
    BOOST_PYTHON_MODULE(libakaze_pybindings)
    {
	init_ar();
	
	to_python_converter<cv::Mat,pbcvt::matToNDArrayBoostConverter>();
	pbcvt::matFromNDArrayBoostConverter();
	
	class_<AKAZEOptions>("AKAZEOptions")
	    .def("setWidth",&AKAZEOptions::setWidth)
	    .def("setHeight",&AKAZEOptions::setHeight)
	    ;
	
	class_<AKAZE>("AKAZE", init<AKAZEOptions>())
	    .def("Create_Nonlinear_Scale_Space", &AKAZE::Create_Nonlinear_Scale_Space)
	    .def("Feature_Detection",&AKAZE::Feature_Detection_)
	    .def("Compute_Descriptors",&AKAZE::Compute_Descriptors_)
	    ;

	class_<Matcher>("Matcher")
	    .def("BFMatch",&Matcher::bfmatch_)
	    ;
	
	
    }

}
