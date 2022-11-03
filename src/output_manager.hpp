#pragma once

#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

using namespace std;
using namespace boost;

namespace loradML {

    class OutputManager {
        public:
            typedef std::shared_ptr< OutputManager > SharedPtr;

                  OutputManager();
                  ~OutputManager();
            
            void   outputConsole() const;
            void   outputConsole(const string & s) const;
            void   outputConsole(const format & fmt) const;
            void   outputConsole(const program_options::options_description & description) const;
    };
    
    inline OutputManager::OutputManager() {
    }

    inline OutputManager::~OutputManager() {
    }

    inline void OutputManager::outputConsole() const {
        cout << endl;
    }
    
    inline void OutputManager::outputConsole(const string & s) const {
        cout << s;
    }
    
    inline void OutputManager::outputConsole(const format & fmt) const {
        cout << str(fmt);
    }
    
    inline void OutputManager::outputConsole(const program_options::options_description & description) const {
        cout << description << endl;
    }
    
}
