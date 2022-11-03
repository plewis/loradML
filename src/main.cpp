//
//  main.cpp
//  loradML
//
//  Created by Paul O. Lewis on 11/1/22.
//

#include <iostream>
#include "output_manager.hpp"

using namespace std;
using namespace loradML;

OutputManager om;

#include "loradML.hpp"
#include "xloradML.hpp"

// static data member initializations
std::string  LoRaDML::_program_name        = "loradML";
unsigned     LoRaDML::_major_version       = 1;
unsigned     LoRaDML::_minor_version       = 0;

int main(int argc, const char * argv[]) {

    LoRaDML loradML;
    try {
        loradML.processCommandLineOptions(argc, argv);
        loradML.run();
    }
    catch(std::exception & x) {
        cerr << "Exception: " << x.what() << endl;
        cerr << "Aborted." << endl;
    }
    catch(...) {
        cerr << "Exception of unknown type!\n";
    }

    return 0;
}
