#pragma once

using namespace std;
using namespace boost;

#include <boost/format.hpp>

namespace loradML {

    class XLoRaDML : public std::exception {
        public:
                                XLoRaDML() throw() {}
                                XLoRaDML(const string s) throw() : _msg() {_msg = s;}
                                XLoRaDML(const format & f) throw() : _msg() {_msg = str(f);}
            virtual             ~XLoRaDML() throw() {}
            const char *        what() const throw() {return _msg.c_str();}

        private:

            string         _msg;
    };

}
