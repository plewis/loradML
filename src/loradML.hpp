#pragma once

#include <chrono>
#include <stdexcept>
#include <vector>
#include <map>
#include <fstream>
#include <numeric>
#include <Eigen/Dense>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/quadrature/trapezoidal.hpp>
#include "conditionals.hpp"
#include "output_manager.hpp"
#include "xloradML.hpp"

using namespace std;
using namespace boost;
//using math::quadrature::trapezoidal;

namespace loradML {

#if defined(GHM)
    struct RefDist {
        string         _name;             // parameter name
        unsigned       _dim;              // dimension of workspace
        unsigned       _n;                // number of parameters (or parameter vectors) processed
        unsigned       _i;                // index of next value to add
        int            _first;            // index of first column associated with this RefDist
        int            _last;             // index of last column associated with this RefDist
        vector<double> _sums;             // summary across rows of data matrix
        vector<double> _sum_of_squares;   // summary across rows of data matrix
        vector<double> _parameters;       // parameters of the reference distribution
                                          // If _dim == 1, holds mean and variance of Gamma reference distribution
                                          // If _dim in [3,4,6,12,61], holds parameters of a Dirichlet reference distribution
        string         _parameter_string; // comma-separated list of parameters in form of a string for reporting purposes

        RefDist() : _dim(0), _n(0), _i(0), _first(-1), _last(-1) {}
        
        void setDimension(unsigned dim) {
            assert(dim > 0 && dim <= 61);
            _dim = dim;
            _sums.resize(_dim);
            _sum_of_squares.resize(_dim);
            if (dim == 1)
                _parameters.resize(2);
            else
                _parameters.resize(dim);
            resetAll();
        }
        
        void addCol(double c) {
            if (_first < 0 || c < _first)
                _first = c;
            if (_last < 0 || c > _last)
                _last = c;
        }
        
        void addValue(double x) {
            assert(_i < _dim);
            _sums[_i] += x;
            _sum_of_squares[_i] += x*x;
            _i++;
        }
        
        void processSampledParameter() {
            _n++;
            _i = 0;
        }
                
        void resetAll() {
            assert(_dim > 0);
            _i = 0;
            _sums.assign(_dim, 0.0);
            _sum_of_squares.assign(_dim, 0.0);
            if (_dim == 1)
                _parameters.assign(2, 0.0);
            else
                _parameters.resize(_dim, 0.0);
        }
        
        void calcParams() {
            assert(_dim > 0);
            assert(_n > 1);
            if (_dim == 1) {
                // Gamma reference distribution
                double mean     = _sums[0]/_n;
                double variance = (_sum_of_squares[0] - mean*mean*_n)/(_n-1);
                // variance = shape*scale*scale
                // mean     = shape*scale
                double scale = variance/mean;
                double shape = mean/scale;
                assert(_parameters.size() == 2);
                _parameters[0] = shape;
                _parameters[1] = scale;
                _parameter_string = str(format("%.5f,%.5f") % shape % scale);
            }
            else {
                // Dirichlet reference distribution
                // Ming-Hui Chen's method of matching component variances
                // mu_i = phi_i/phi is mean of component i (estimate using sample mean)
                // s_i^2 is sample variance of component i
                //
                //       sum_i mu_i^2 (1 - mu_i)^2
                // phi = --------------------------- - 1
                //       sum_i s_i^2 mu_i (1 - mu_i)
                //
                // phi_i = phi mu_i
                assert(_n > 1);
                vector<double> mean(_dim, 0.0);
                double numerator = 0.0;
                double denominator = 0.0;
                for (unsigned i = 0; i < _dim; i++) {
                    double m = _sums[i]/_n;
                    double v = (_sum_of_squares[i] - m*m*_n)/(_n-1);
                    numerator   += m*m*(1.0 - m)*(1.0 - m);
                    denominator += v*m*(1.0 - m);
                    mean[i] = m;
                }
                double phi = numerator/denominator - 1.0;
                assert(_parameters.size() == _dim);
                vector<string> _param_str_vect(_dim);
                for (unsigned i = 0; i < _dim; i++) {
                    _parameters[i] = mean[i]*phi;
                    _param_str_vect[i] = str(format("%.5f") % _parameters[i]);
                }
                _parameter_string = algorithm::join(_param_str_vect, ",");
            }
        }
        
        double calcLogProbDensity(double v) {
            // Return log of the Gamma(shape, scale) probability density for the supplied value v
            assert(_dim == 1);
            assert(_parameters.size() == 2);
            double a = _parameters[0];
            double b = _parameters[1];
            double logp = (a - 1.0)*log(v) - v/b - a*log(b) -lgamma(a);
            return logp;
        }
        
        double calcLogProbDensity(vector<double> & v) {
            // Return log of the Dirichlet probability density for the supplied vector v
            assert(_dim == v.size());
            double logp = 0.0;
            double param_sum = 0.0;
            for (unsigned i = 0; i < _dim; i++) {
                param_sum += _parameters[i];
                logp += (_parameters[i] - 1.0)*log(v[i]) - lgamma(_parameters[i]);
            }
            logp += lgamma(param_sum);
            return logp;
        }
        
    };
#endif

    struct ParameterSample {
        unsigned         _iteration;    // original iteration that produced this sample point
        unsigned         _index;        // index before sorting by norm
        double           _kernel;       // posterior kernel
        double           _norm;         // Euclidean distance from mean
        Eigen::VectorXd  _param_vect;   // vector of sampled parameter values
        
        void init(unsigned it, unsigned idx, double k, double n, vector<double> & v) {
            _iteration = it;
            _index = idx;
            _kernel = k;
            _norm = n;
            // Use Eigen::Map to map a vector of doubles onto an Eigen::VectorXd
            _param_vect = Eigen::Map<Eigen::VectorXd>(&v[0], (unsigned)v.size());
        }

        // Define less-than operator so that a vector of ParameterSample objects can be sorted
        // from smallest to largest norm
        bool operator<(const ParameterSample & other) const {
            return _norm < other._norm;
        }

        // Define greater-than operator so that a vector of ParameterSample objects can be sorted
        // from largest to smallest norm
        bool operator>(const ParameterSample & other) const {
            return _norm > other._norm;
        }
    };

    struct ColumnSpec {
            enum column_t {       //   dec              bin
                ignore              =    0,  // 0 0000 0000
                iteration           =    1,  // 0 0000 0001
                posterior           =    2,  // 0 0000 0010
                unconstrained       =    4,  // 0 0000 0100 (counts as parameter)
                positive            =    8,  // 0 0000 1000 (counts as parameter)
                correlation         =   16,  // 0 0001 0000 (counts as parameter)
                proportion          =   32,  // 0 0010 0000 (counts as parameter)
                simplex             =   64,  // 0 0100 0000 (counts as parameter)
                simplexfinal        =  128,  // 0 1000 0000
                unknown             =  256,  // 1 0000 0000
                parameter           =  124   // 0 0111 1100
            };
            
            ColumnSpec() {
                _coltype = unknown;
            }
            
            ColumnSpec(string t, string nm) {
                _name = nm;
                if (t == "ignore") {
                    _coltype = ignore;
                }
                else if (t == "iteration") {
                    _coltype = iteration;
                }
                else if (t == "posterior") {
                    _coltype = posterior;
                }
                else if (t == "unconstrained") {
                    _coltype = unconstrained;
                }
                else if (t == "positive") {
                    _coltype = positive;
                }
                else if (t == "proportion") {
                    _coltype = proportion;
                }
                else if (t == "correlation") {
                    _coltype = correlation;
                }
                else if (t == "simplex") {
                    _coltype = simplex;
                }
                else if (t == "simplexfinal") {
                    _coltype = simplexfinal;
                }
                else {
                    _coltype = unknown;
                }
            }
            
            bool isParameter() {
                int x = _coltype & parameter;
                bool is_param = (x > 0);
                return is_param;
            }
            
            unsigned _coltype;
            string _name;
    };

    class LoRaDML {
        public:
                                        LoRaDML();
                                        ~LoRaDML();

            void                        clear();
            void                        processCommandLineOptions(int argc, const char * argv[]);
            void                        run();
            
        private:
        
            void                        setup();
            double                      calcLogSum(const std::vector<double> & logx_vect);
            void                        readParamFile();
            void                        handleColSpecs();
            void                        partitionSample(unsigned start_index, unsigned end_index);
            void                        standardizeParameters();
            double                      calcMarginalLikelihood(bool verbose = true);
            
#if defined(GHM)
            double                      doGHM();
#endif
            
            bool                        _quiet;
            bool                        _debug_transformations;
                        
            // Starting and ending samples (first sample is 1). Used for overlapping batch
            // mean (OBM) estimation of Monte Carlo standard error (MCSE).
            // Specify 0 for both if entire sample to be used (i.e. not computing OBM)
            unsigned                    _starting_sample;   // startsample in loradML.conf
            unsigned                    _ending_sample;     // endsample in loradML.conf
            
            // Fraction of sample to be used for training (i.e. estimating mean vector and
            // covariance matrix for standardization as well as rmax)
            double                      _training_fraction; // trainingfrac in loradML.conf
            
            // Fraction of (sorted by norm) training sample to use in determining rmax (the
            // outer limit radius (i.e. euclidean norm) defining the working parameter space)
            double                      _coverage; // coverage in loradML.conf
            
            // Other quantities used for LoRaD marginal likelihood estimation
            double                      _rmax;              // radial limit of working parameter space
            
            // Parameter vectors
            unsigned                        _nsamples_total; // number of samples stored in parameter file
            unsigned                        _nparameters;    // number of free parameters
            std::vector< ParameterSample >  _training_sample;
            std::vector< ParameterSample >  _estimation_sample;

            unsigned                        _first_index;   // index of first parameter sample used
            unsigned                        _last_index;    // one beyond index of last parameter sample
                                                            
            // Standardization
            typedef Eigen::VectorXd eigenVectorXd_t;
            typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenMatrixXd_t;
            eigenVectorXd_t                 _mean_transformed;
            eigenMatrixXd_t                 _S;
            eigenMatrixXd_t                 _sqrtS;
            eigenMatrixXd_t                 _invSqrtS;
            double                          _logDetSqrtS;
            
            // MCSE calculation
            bool                        _mcse;     // determines whether to estimate MCSE
            unsigned                    _T;        // sample size
            unsigned                    _B;        // batch size
            unsigned                    _nbatches; // number of batches
            double                      _TBratio;  // ratio of sample size (T) to batch size (B)
            unsigned                    _minimum_batch_size;

            // Related to reading in the parameter file
            string                      _param_file;            // paramfile in loradML.conf
            vector<string>              _colspecs;              // vector of colspec entries from loradML.conf
            vector<ColumnSpec>          _column_specifications; // vector of ColumnSpec objects
            vector<string>              _column_names;          // vector column headers (from input file)
            vector<unsigned>            _orig_iter;         // element i is the original MCMC iteration for the ith sample
            vector<double>              _kernel_values;     // element i is the log-kernel of the ith sample
            vector< vector<double> >    _parameter_vectors; // element i is the ith sampled parameter vector
            
#if defined(GHM)
            bool                        _ghm;   // estimate marginal likelihood using GHM
            unsigned                    _ghm_nparameters;
            vector< string >            _ghm_parameter_names;
            vector<double>              _ghm_kernel_values; // untransformed log posterior kernel
            vector< vector<double> >    _ghm_parameter_vectors; // untransformed parameter vectors
#endif

            // Program name and version
            static string               _program_name;
            static unsigned             _major_version;
            static unsigned             _minor_version;
    };
    
    inline LoRaDML::LoRaDML() {
        clear();
    }

    inline LoRaDML::~LoRaDML() {
    }

    inline void LoRaDML::clear() {
        _nsamples_total = 0;
        _nparameters = 0;
        _first_index = 0;
        _last_index = 0;
        _rmax = 0.0;
        _quiet = false;
        _mcse = false;
        _TBratio = 10.0;
        _debug_transformations = false;
        _minimum_batch_size = 200;
    }
    
    inline void LoRaDML::processCommandLineOptions(int argc, const char * argv[]) {
        program_options::variables_map vm;
        program_options::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("version,v", "show program version")
            ("debugtransformations,d", program_options::bool_switch(&_debug_transformations), "if supplied, provides detailed output for first processed sample line and then quits")
            ("quiet,q", program_options::bool_switch(&_quiet), "if supplied, the only output will be the estimated log marginal likelihood (and possibly MCSE)")
            ("paramfile", program_options::value(&_param_file)->required(), "name of file containing sampled parameter values")
            ("colspec", program_options::value(&_colspecs), "column specification (provide one for each column in paramfile)")
            ("startsample", program_options::value(&_starting_sample)->default_value(0), "first sample to consider (starting with 1; specify 0 to consider all samples)")
            ("endsample", program_options::value(&_ending_sample)->default_value(0), "last sample to consider (starting with 1; specify 0 to consider all samples)")
            ("trainingfrac,t", program_options::value(&_training_fraction)->default_value(0.5), "fraction of sample to use for training (determining rmax and mean vector and covariance matrix for standardization)")
            ("coverage,c", program_options::value(&_coverage)->default_value(0.25), "fraction of training sample to use for determining rmax")
            ("mcse,m", program_options::bool_switch(&_mcse), "if specified, use overlapping batch statistics to estimate Monte Carlo standard error.")
            ("tbratio", program_options::value(&_TBratio)->default_value(10.0), "ratio of total sample size to (overlapping) batch sample size (e.g. 10.0) for estimating Monte Carlo standard error.")
#if defined(GHM)
            ("ghm", program_options::bool_switch(&_ghm), "if specified, estimate marginal likelihood using Generalized Harmonic Mean method (Note: ad hoc).")
#endif
        ;
        program_options::store(program_options::parse_command_line(argc, argv, desc), vm);
        try {
            const program_options::parsed_options & parsed = program_options::parse_config_file< char >("loradml.conf", desc, false);
            program_options::store(parsed, vm);
        }
        catch(program_options::reading_file & x) {
            ::om.outputConsole("Note: configuration file (loradml.conf) not found\n");
        }
        program_options::notify(vm);

        // If user specified --help on command line, output usage summary and quit
        if (vm.count("help") > 0) {
            ::om.outputConsole(desc);
            ::om.outputNewline();
            exit(1);
        }

        // If user specified --version on command line, output version and quit
        if (vm.count("version") > 0) {
            ::om.outputConsole(format("This is %s version %d.%d\n") % _program_name % _major_version % _minor_version);
            exit(1);
        }
        
        // Bail out if user did not specify colspec entries
        if (vm.count("colspec") > 0) {
            handleColSpecs();
        }
        else {
            ::om.outputConsole("Please provide one colspec entry for each column in paramfile");
            exit(1);
        }
        
        // Bail out if user specified a crazy value for mcseratio
        if (vm.count("mcse") > 0) {
            if (_TBratio < 5.0) {
                ::om.outputConsole(format("Warning: recommended tbratio is 10-20; you specified %g\n") % _TBratio);
            }
            else if (_TBratio < 1.0) {
                throw XLoRaDML("Cannot specify tbratio less than 1 (10-20 is recommended)");
            }
        }
    }

    inline void LoRaDML::handleColSpecs() {
        _column_specifications.resize(_colspecs.size());
        vector<string> parts;
        unsigned i = 0;
        for (auto spec : _colspecs) {
            trim(spec);
            split(parts, spec, boost::is_any_of(" "), token_compress_on);
            unsigned nparts = (unsigned)parts.size();
            if (nparts != 2) {
                throw XLoRaDML(format("Expecting colspec to comprise 2 parts (column, type, and name), but found %d parts: \"%s\"") % nparts % spec);
            }
            assert(_column_specifications[i]._coltype == ColumnSpec::unknown);
            _column_specifications[i] = ColumnSpec(parts[0], parts[1]);
            if (_column_specifications[i]._coltype == ColumnSpec::unknown)
                throw XLoRaDML(format("colspec named \"%s\" specified an unknown column type (%s)") % parts[1] % parts[0]);
            ++i;
        }
    }

    inline double LoRaDML::calcLogSum(const std::vector<double> & logx_vect) {
        double max_logx = *(max_element(logx_vect.begin(), logx_vect.end()));
        double sum_terms = 0.0;
        for (auto logx : logx_vect) {
            sum_terms += exp(logx - max_logx);
        }
        double logsum = max_logx + log(sum_terms);
        return logsum;
    }
    
    inline void LoRaDML::readParamFile() {
        bool param_file_exists = boost::filesystem::exists(_param_file);
        if (!param_file_exists)
            throw XLoRaDML(format("Could not find the specified paramfile (\"%s\")") % _param_file);
            
        if (!_quiet) {
            ::om.outputConsole("\nReading parameter sample file...\n");
        }

        // Initializations
        _orig_iter.clear();
        _kernel_values.clear();
        _parameter_vectors.clear();
        
#if defined(GHM)
        _ghm_parameter_names.clear();
        _ghm_kernel_values.clear();
        _ghm_parameter_vectors.clear();
#endif
        
        // Read the file
        ifstream inf(_param_file);
        string line;
        vector<string> tmp;
        vector<double> tmp_param_vect;
#if defined(GHM)
        vector<double> ghm_param_vect;
#endif
        vector<double> simplex_workspace;
        unsigned i = 0;
        while (getline(inf, line)) {
            if (i == 0) {
                // Store column headers in _column_names data member
                split(_column_names, line, boost::is_any_of("\t"));
                
                // Determine number of parameters by examining _column_specifications
                _nparameters = 0;
                for (auto & cspec : _column_specifications) {
                    if (cspec.isParameter())
                        ++_nparameters;
                }
                tmp_param_vect.resize(_nparameters);

#if defined(GHM)
                _ghm_nparameters = 0;
                for (auto & cspec : _column_specifications) {
                    if (cspec.isParameter() || cspec._coltype == ColumnSpec::simplexfinal) {
                        ++_ghm_nparameters;
                        _ghm_parameter_names.push_back(cspec._name);
                    }
                }
                ghm_param_vect.resize(_ghm_nparameters);
#endif
                
                // Ensure that there is a simplexfinal colspec ending each run of simplex colspecs
                bool in_simplex = false;
                for (auto & cspec : _column_specifications) {
                    if (cspec._coltype == ColumnSpec::simplex)
                        in_simplex = true;
                    else if (cspec._coltype == ColumnSpec::simplexfinal) {
                        in_simplex = false;
                    }
                    else if (in_simplex) {
                        // If in_simplex is true, abort because every run of simplex colspecs must end
                        // with a simplexfinal colspec
                        throw XLoRaDML(format("A simplexfinal colspec must end every run of simplex colspecs, but one simplex run ended with a colspec named %s") % cspec._name);
                    }
                    else {
                        in_simplex = false;
                    }
                }
            }
            else {
                // store parameter vector and posterior kernel for this sample
                
                // Split line from parameter file into tab-delimited columns
                trim(line);
                split(tmp, line, boost::is_any_of("\t"));
                
                if (tmp.size() != _column_specifications.size()) {
                    throw XLoRaDML(format("Number of columns (%d) does not match number of colspec entries (%d)") % tmp.size() % _column_specifications.size());
                }
                
                // Initializations
                unsigned col      = 0;    // index of column in parameter file
                unsigned param    = 0;    // index of parameter in parameter vector
                unsigned origiter = 0;    // original iteration of sample
                double   kernel   = 0.0;  // posterior kernel value from parameter file
                
#if defined(GHM)
                double   ghm_kernel = 0.0; // untransformed posterior kernel value from parameter file
                unsigned ghm_param = 0;
#endif
                                        
                // Go through each column spec and column, pulling out information for this sample
                for (auto & cspec : _column_specifications) {
                    if (cspec._coltype == ColumnSpec::iteration) {
                        unsigned it = 0;
                        try {
                            it = stoi(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to an integer representing the MCMC iteration for this sample") % (i+1) % tmp[col] % col);
                        }
                        origiter = it;
                        
                        if (_debug_transformations)
                            ::om.outputConsole(format("%12d iteration\n") % it);
                    }
                    else if (cspec._coltype == ColumnSpec::posterior) {
                        double v = 0.0;
                        try {
                            v = stod(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point posterior kernel compnent") % (i+1) % tmp[col] % col);
                        }
                        kernel += v;

#if defined(GHM)
                        ghm_kernel += v;
#endif

                        if (_debug_transformations)
                            ::om.outputConsole(format("%12.5f %12s %12s log-posterior (kernel is now %g)\n") % v % "" % "" % kernel);
                    }
                    else if (cspec._coltype == ColumnSpec::unconstrained) {
                        double v = 0.0;
                        try {
                            v = stod(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point parameter value") % (i+1) % tmp[col] % col);
                        }
                        tmp_param_vect[param++] = v;
#if defined(GHM)
                        ghm_param_vect[ghm_param++] = v;
#endif
                        
                        if (_debug_transformations) {
                            ::om.outputConsole(format("%12.5f %12.5f %12.5f unconstrained (\"%s\") (kernel is now %g)\n") % 0.0 % v % v % cspec._name % kernel);
                        }
                    }
                    else if (cspec._coltype == ColumnSpec::positive) {
                        double v = 0.0;
                        try {
                            v = stod(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point parameter value") % (i+1) % tmp[col] % col);
                        }
                        if (v <= 0.0) {
                            throw XLoRaDML(format("in line %d, was expecting strictly positive parameter value in column \"%s\", but found %g instead") % (i+1) % cspec._name % v);
                        }
                        
#if defined(GHM)
                        ghm_param_vect[ghm_param++] = v;
#endif

                        // Peform log transformation
                        double logv = log(v);
                        
                        // Add the log-Jacobian to the kernel
                        kernel += logv;
                        
                        tmp_param_vect[param++] = logv;

                        if (_debug_transformations)
                            ::om.outputConsole(format("%12.5f %12.5f %12.5f positive (\"%s\") (kernel is now %g)\n") % logv % v % logv % cspec._name % kernel);
                    }
                    else if (cspec._coltype == ColumnSpec::proportion) {
                        double p = 0.0;
                        try {
                            p = stod(tmp[col]);
                            if (p <= 0.0)
                                throw runtime_error("expecting parameter value in interval (0,1)");
                            if (p >= 1.0)
                                throw runtime_error("expecting parameter value in interval (0,1)");
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point parameter value") % (i+1) % tmp[col] % col);
                        }
                        
#if defined(GHM)
                        ghm_param_vect[ghm_param++] = p;
#endif

                        // Perform logit transformation
                        double logitp = log(p) - log(1.0 - p);
                        
                        // Add the log-Jacobian to the kernel
                        kernel += log(p);
                        kernel += log(1.0 - p);
                        
                        tmp_param_vect[param++] = logitp;

                        if (_debug_transformations)
                            ::om.outputConsole(format("%12.5f %12.5f %12.5f proportion (\"%s\") (kernel is now %g)\n") % (log(p) + log(1.0 - p)) % p % logitp % cspec._name % kernel);
                    }
                    else if (cspec._coltype == ColumnSpec::correlation) {
                        double r = 0.0;
                        try {
                            r = stod(tmp[col]);
                            if (r <= -1.0)
                                throw runtime_error("expecting parameter value in interval (-1,1)");
                            if (r >= 1.0)
                                throw runtime_error("expecting parameter value in interval (-1,1)");
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point parameter value") % (i+1) % tmp[col] % col);
                        }
                        
#if defined(GHM)
                        ghm_param_vect[ghm_param++] = r;
#endif

                        // Perform transformation (-1,1) -> (-infty,+infty)
                        double logrstar = log(1.0 + r) - log(1.0 - r);
                        
                        // Add the log-Jacobian to the kernel
                        kernel += log(1.0 + r);
                        kernel += log(1.0 - r);
                        kernel -= log(2.0);
                        
                        tmp_param_vect[param++] = logrstar;

                        if (_debug_transformations)
                            ::om.outputConsole(format("%12.5f %12.5f %12.5f correlation (\"%s\") (kernel is now %g)\n") % (log(1.0 + r) + log(1.0 - r) - log(2.0)) % r % logrstar % cspec._name % kernel);
                    }
                    else if (cspec._coltype == ColumnSpec::simplex) {
                        double v = 0.0;
                        try {
                            v = stod(tmp[col]);
                            if (v <= 0.0)
                                throw runtime_error("expecting parameter value in interval (0,1)");
                            if (v >= 1.0)
                                throw runtime_error("expecting parameter value in interval (0,1)");
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point parameter value") % (i+1) % tmp[col] % col);
                        }

#if defined(GHM)
                        ghm_param_vect[ghm_param++] = v;
#endif

                        // Add value to simplex_workspace vector having key equal to cspec._name
                        //simplex_workspace[cspec._name].push_back(v);
                        simplex_workspace.push_back(v);
                    }
                    else if (cspec._coltype == ColumnSpec::simplexfinal) {
                        double finalv = 0.0;
                        try {
                            finalv = stod(tmp[col]);
                            if (finalv < 0.0)
                                throw runtime_error("expecting parameter value in interval [0,1]");
                            if (finalv > 1.0)
                                throw runtime_error("expecting parameter value in interval [0,1]");
                        }
                        catch(...) {
                            throw XLoRaDML(format("in line %d, could not convert string \"%s\" in column %d to a floating-point parameter value") % (i+1) % tmp[col] % col);
                        }
                        
#if defined(GHM)
                        ghm_param_vect[ghm_param++] = finalv;
#endif

                        // Ensure that sum is within 0.0001 of 1.0
                        //vector<double> & w = simplex_workspace[cspec._name];
                        //double simplex_sum = finalv + accumulate(w.begin(), w.end(), 0.0);
                        double simplex_sum = finalv + accumulate(simplex_workspace.begin(), simplex_workspace.end(), 0.0);
                        if (fabs(simplex_sum - 1.0) > 0.0001)
                            throw XLoRaDML(format("in line %d, simplex components for \"%s\" sum to %g but should sum to 1.0") % (i+1) % cspec._name % simplex_sum);
                            
                        // Use final element as reference to perform log-ratio transformation
                        double logfinalv = log(finalv);

                        for (auto & v : simplex_workspace) {
                            // Peform log transformation
                            double logv = log(v);
                        
                            // Add the log-Jacobian to the kernel
                            kernel += logv;
                        
                            tmp_param_vect[param++] = logv - logfinalv;

                            if (_debug_transformations) {
                                ::om.outputConsole(format("%12.5f %12.5f %12.5f simplex (\"%s\") (kernel is now %g)\n") % logv % v % (logv - logfinalv) % cspec._name % kernel);
                            }
                        }
                        
                        kernel += logfinalv;
                        if (_debug_transformations) {
                            ::om.outputConsole(format("%12.5f %12.5f %12.5f simplexfinal (\"%s\") (kernel is now %g)\n") % logfinalv % finalv %  logfinalv % cspec._name % kernel);
                        }

                        // Clean out workspace vector so that next iteration does not add to it
                        //w.clear();
                        simplex_workspace.resize(0);
                    }
                    ++col;
                }
                
                if (_debug_transformations) {
                    ::om.outputConsole("\nParameter vector:\n");
                    unsigned j = 0;
                    for (auto x : tmp_param_vect) {
                        ::om.outputConsole(format("%12d %12.5f\n") % (++j) % x);
                    }
                    cerr << "\nAborted (set debugtransformations = no in loradml.conf file to avoid this)\n" << endl;
                    exit(0);
                }
                    
                // Can now store origiter, kernel, and the parameter vector
                _orig_iter.push_back(origiter);
                _kernel_values.push_back(kernel);
                _parameter_vectors.push_back(tmp_param_vect);
#if defined(GHM)
                _ghm_kernel_values.push_back(ghm_kernel);
                _ghm_parameter_vectors.push_back(ghm_param_vect);
#endif
                
            }
            
            ++i;
        }
        
        // Provide a summary
        if (_column_names.size() == 0) {
            throw XLoRaDML("  File seems to be empty or does not comprise columns of numbers");
        }
        if (!_quiet) {
            ::om.outputConsole(format("  Processed %d column specification%s\n") % _colspecs.size() % (_colspecs.size() == 1 ? "s" : ""));
            ::om.outputConsole(format("  Found %d parameter%s\n") % _nparameters % (_nparameters == 1 ? "s" : ""));
            ::om.outputConsole(format("  Found %d column%s\n") % _column_names.size() % (_column_names.size() == 1 ? "s" : ""));
            ::om.outputConsole(format("  File has %d line%s\n") % i % (i == 1 ? "s" : ""));
            ::om.outputConsole(format("  Found %d value%s for each column\n") % _orig_iter.size() % (_orig_iter.size() == 1 ? "s" : ""));
        }
    }
    
    inline void LoRaDML::partitionSample(unsigned start_index, unsigned end_index) {
        if (!_quiet) {
            ::om.outputConsole("\nPartitioning samples into training and estimation fraction...\n");
        }

        // Allocate _training_sample and _estimation_sample
        //
        //   Sample size is nsamples = 9
        //   1    2    3    4    5    6    7    8    9
        //
        //   0    1    2    3    4    5    6    7    8    9
        //   |                        |                   |
        //   start_index              boundary            end_index
        //
        unsigned nsamples    = end_index - start_index;
        unsigned boundary    = start_index + (unsigned)(ceil(_training_fraction*nsamples));
        unsigned ntraining   = boundary - start_index;
        unsigned nestimation = end_index - boundary;
            
        // Store training sample
        _training_sample.resize(ntraining);
        unsigned j = 0;
        for (unsigned i = start_index; i < boundary; i++) {
            _training_sample[j++].init(_orig_iter[i], i, _kernel_values[i], 0.0, _parameter_vectors[i]);
        }
        
        // Store estimation sample
        _estimation_sample.resize(nestimation);
        j = 0;
        for (unsigned i = boundary; i < end_index; i++) {
            _estimation_sample[j++].init(_orig_iter[i], i, _kernel_values[i], 0.0, _parameter_vectors[i]);
        }
        
        if (!_quiet) {
            ::om.outputConsole(format("  Sample size is %d\n") % nsamples);
            ::om.outputConsole(format("  Training sample size is %d\n") % ntraining);
            ::om.outputConsole(format("  Estimation sample size is %d\n") % nestimation);
        }
    }
    
    inline void LoRaDML::standardizeParameters() {
        if (!_quiet) {
            ::om.outputConsole("\nProcessing training sample...\n");
        }
                
        // Zero the mean vector (_mean_transformed)
        assert(_nparameters > 0);
        _mean_transformed = eigenVectorXd_t::Zero(_nparameters);
        
        // Zero the variance-covariance matrix (_S)
        _S.resize(_nparameters, _nparameters);
        _S = eigenMatrixXd_t::Zero(_nparameters, _nparameters);
        
        // Calculate mean vector _mean_transformed
        unsigned ntraining = (unsigned)_training_sample.size();
        for (auto & v : _training_sample) {
            // Add v._param_vect elementwise to _mean_transformed
            _mean_transformed += v._param_vect;
        }
        _mean_transformed /= ntraining;
        
        // Sanity check
        assert(_mean_transformed.rows() == _nparameters);

        // Calculate variance-covariance matrix _S
        for (auto & v : _training_sample) {
            eigenVectorXd_t x = v._param_vect - _mean_transformed;
            _S += x*x.transpose();
        }
        _S /= ntraining - 1;
        
        // Compute eigenvalues and eigenvectors of _S. Can use an efficient
        // eigensystem solver because S is positive definite and symmetric
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(_S);
        if (solver.info() != Eigen::Success) {
            throw XLoRaDML("Error in the calculation of eigenvectors and eigenvalues of the variance-covariance matrix");
        }
        
        // Extract the eigenvalues into vector L
        Eigen::VectorXd L = solver.eigenvalues();
        
        // In case some eigenvalues are "negative zero"
        L = L.array().abs();
        
        // Perform component-wise square root
        L = L.array().sqrt();
        
        // Extract the eigenvectors into matrix V
        Eigen::MatrixXd V = solver.eigenvectors();
        
        // Compute the square root of the covariance matrix
        _sqrtS = V*L.asDiagonal()*V.transpose();
        
        // Compute the inverse square root of the covariance matrix
        _invSqrtS = _sqrtS.inverse();
        
        // Compute the log determinant of the square root of the covariance matrix
        // This is the log of the Jacobian for the standardization transformation
        _logDetSqrtS = log(_sqrtS.determinant());
        
        //::om.outputConsole(boost::format("  _logDetSqrtS = %.5f\n") % _logDetSqrtS);

        // Standardize the training sample
        for (auto & v : _training_sample) {
            v._param_vect = _invSqrtS*(v._param_vect - _mean_transformed);
            v._norm       = v._param_vect.norm();
            v._kernel     = v._kernel + _logDetSqrtS;
        }
        
        // Sort training sample from smallest to largest norm
        std::sort(_training_sample.begin(), _training_sample.end(), std::less<ParameterSample>());
        
        // Standardize the estimation sample
        for (auto & v : _estimation_sample) {
            v._param_vect = _invSqrtS*(v._param_vect - _mean_transformed);
            v._norm       = v._param_vect.norm();
            v._kernel     = v._kernel + _logDetSqrtS;
        }
        
        // Sort estimation sample from smallest to largest norm
        std::sort(_estimation_sample.begin(), _estimation_sample.end(), std::less<ParameterSample>());
    }
    
    inline double LoRaDML::calcMarginalLikelihood(bool verbose) {
        // Determine how many sample vectors to use for working parameter space
        unsigned ntraining = (unsigned)_training_sample.size();
        unsigned nretained = (unsigned)floor(_coverage*ntraining);
        assert(nretained > 1);
        
        double _rmax = _training_sample[nretained]._norm;
        if (!_quiet) {
            ::om.outputConsole(boost::format("  Lowest radial distance is %.5f\n") % _rmax);
            ::om.outputConsole("\nProcessing estimation sample...\n");
        }
        
        // Determine Delta, the integral from 0.0 to _rmax of the
        // marginal distribution of radial vector lengths of a multivariate
        // standard normal distribution with dimension _nparameters
        double p = _nparameters;
        double s = p/2.0;
        double sigma_squared = 1.0; // use sigma_squared = 1.0 for standard normal
        double sigma = sqrt(sigma_squared);
        double t = _rmax*_rmax/(2.0*sigma_squared);
        double log_Delta = log(boost::math::gamma_p(s, t));
        
        //::om.outputConsole(boost::format("  log_Delta = %.5f\n") % log_Delta);

        // Calculate the sum of ratios in the PWK method, using the multivariate
        // standard normal density as the reference
        double log_mvnorm_constant = 0.5*p*log(2.*M_PI) + 1.0*p*log(sigma);
        std::vector<double> log_ratios;
        std::vector<double> log_inv_ratios;
        unsigned nestimation = (unsigned)_estimation_sample.size();
        for (unsigned i = 0; i < nestimation; ++i) {
            double norm = _estimation_sample[i]._norm;
            if (norm > _rmax)
                break;
            double log_kernel = _estimation_sample[i]._kernel;
            double log_reference = -0.5*sigma_squared*pow(norm,2.0) - log_mvnorm_constant;
            double log_ratio = log_reference - log_kernel;
            log_ratios.push_back(log_ratio);
            if (verbose)
                log_inv_ratios.push_back(-1.0*log_ratio);
        }
        
        if (log_ratios.size() == 0) {
            throw XLoRaDML("Zero samples fall inside working parameter space; try increasing coverage fraction");
        }
        double log_sum_ratios = calcLogSum(log_ratios);
        double log_marginal_likelihood = log_Delta - (log_sum_ratios - log(nestimation));

        if (verbose) {
            double log_sum_inv_ratios = calcLogSum(log_inv_ratios);
            double ghmstar = log_sum_inv_ratios - log(nestimation);
            ::om.outputConsole(format("  GHM* estimator is %.5f\n") % ghmstar);
        }
        
        if (!_quiet) {
            ::om.outputConsole(format("  Number of samples used is %d\n") % log_ratios.size());
            ::om.outputConsole(format("  Nominal coverage is %.5f\n") % _coverage);
            ::om.outputConsole(format("  Actual coverage is %.5f\n") % (1.0*log_ratios.size()/nestimation));
        }
        
        if (verbose) {
            ::om.outputConsole(format("  Log marginal likelihood is %.5f\n") % log_marginal_likelihood);
        }
            
        return log_marginal_likelihood;
    }
    
    inline void LoRaDML::setup() {
        // Example:
        // T/B       = 5
        // T         = 11.0
        // B         = floor(2.2) = 2.0
        // T - B + 1 = 10 batches
        //
        //      0  1  2  3  4  5  6  7  8  9  10
        //  1   0  1
        //  2      1  2
        //  3         2  3
        //  4            3  4
        //  5               4  5
        //  6                  5  6
        //  7                     6  7
        //  8                        7  8
        //  9                           8  9
        // 10                              9  10
        //
        _nsamples_total = (unsigned)_orig_iter.size();
        if (_starting_sample > _nsamples_total) {
            throw XLoRaDML(format("you specified startsample = %d but there are only %d samplea in the file") % _starting_sample % _nsamples_total);
        }
        if (_ending_sample > _nsamples_total) {
            throw XLoRaDML(format("you specified endsample = %d but there are only %d samplea in the file") % _ending_sample % _nsamples_total);
        }
        _first_index = (_starting_sample == 0 ? 0 : _starting_sample - 1);
        _last_index  = (_ending_sample == 0 ? _nsamples_total : _ending_sample);
        _T = _last_index - _first_index;
        if (_mcse) {
            _B = (unsigned)(floor((float)_T/_TBratio));
        
            // sanity check
            if (_B < _minimum_batch_size)
                throw XLoRaDML(format("batch size < %d for T = %d and specified T/B ratio (%.3f)") % _minimum_batch_size % _T % _TBratio);
                
            _nbatches = _T - _B + 1;
        }
    }
    
#if defined(GHM)
    inline double LoRaDML::doGHM() {
        // This function is ad hoc and the presence of unconstrained colspecs means it is not being used
        // for the purpose for which it was designed
        for (auto & cspec : _column_specifications) {
            if (cspec._coltype == ColumnSpec::unconstrained)
                throw XLoRaDML("ghm cannot be computed if there are unconstrained parameters");
        }
    
        // Create reference distribution objects for all parameters that could be present in the four partitioning
        // models (unpart, bygene, bycodon, and byboth) for the 4-gene cicada example used in the Wang et al. paper
        RefDist         edgeprop;
        RefDist         treelen;
        RefDist         relrate;
        vector<RefDist> basefreq(12);
        vector<RefDist> exchange(12);
        vector<RefDist> ratevar(12);
        unsigned nrows = (unsigned)_ghm_parameter_vectors.size();
        unsigned ncols = (unsigned)_ghm_parameter_names.size();

        // Create a vector of references to RefDist objects so that we don't have to examine parameter names
        // to parse each row of the data matrix
        vector<RefDist *> refdists;
        set<RefDist *> refdists_used;
        vector<string> parts;
        for (unsigned col = 0; col < ncols; col++) {
            string nm = _ghm_parameter_names[col];
            if (nm == "TL") {
                assert(treelen._dim == 0 || treelen._dim == 1);
                if (treelen._dim == 0) {
                    treelen._name = "tree_length";
                    treelen.setDimension(1);
                }
                treelen.addCol(col);
                refdists.push_back(&treelen);
                refdists_used.insert(&treelen);
            }
            else if (algorithm::contains(nm, "edgeProp") || algorithm::contains(nm, "edgeLen")) {
                // Note: "edgeLen" changed to "edgeProp" but need to handle legacy cases
                assert(edgeprop._dim == 0 || edgeprop._dim == 61);
                if (edgeprop._dim == 0) {
                    edgeprop._name = "edge_proportions";
                    edgeprop.setDimension(61);
                }
                edgeprop.addCol(col);
                refdists.push_back(&edgeprop);
                refdists_used.insert(&edgeprop);
            }
            else if (algorithm::contains(nm, "rAC")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(exchange[which]._dim == 0 || exchange[which]._dim == 6);
                if (exchange[which]._dim == 0) {
                    exchange[which]._name = str(format("exchangeabilities-%d") % which);
                    exchange[which].setDimension(6);
                }
                exchange[which].addCol(col);
                refdists.push_back(&exchange[which]);
                refdists_used.insert(&exchange[which]);
            }
            else if (algorithm::contains(nm, "rAG")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(exchange[which]._dim == 0 || exchange[which]._dim == 6);
                if (exchange[which]._dim == 0) {
                    exchange[which]._name = str(format("exchangeabilities-%d") % which);
                    exchange[which].setDimension(6);
                }
                exchange[which].addCol(col);
                refdists.push_back(&exchange[which]);
                refdists_used.insert(&exchange[which]);
            }
            else if (algorithm::contains(nm, "rAT")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(exchange[which]._dim == 0 || exchange[which]._dim == 6);
                if (exchange[which]._dim == 0) {
                    exchange[which]._name = str(format("exchangeabilities-%d") % which);
                    exchange[which].setDimension(6);
                }
                exchange[which].addCol(col);
                refdists.push_back(&exchange[which]);
                refdists_used.insert(&exchange[which]);
            }
            else if (algorithm::contains(nm, "rCG")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(exchange[which]._dim == 0 || exchange[which]._dim == 6);
                if (exchange[which]._dim == 0) {
                    exchange[which]._name = str(format("exchangeabilities-%d") % which);
                    exchange[which].setDimension(6);
                }
                exchange[which].addCol(col);
                refdists.push_back(&exchange[which]);
                refdists_used.insert(&exchange[which]);
            }
            else if (algorithm::contains(nm, "rCT")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(exchange[which]._dim == 0 || exchange[which]._dim == 6);
                if (exchange[which]._dim == 0) {
                    exchange[which]._name = str(format("exchangeabilities-%d") % which);
                    exchange[which].setDimension(6);
                }
                exchange[which].addCol(col);
                refdists.push_back(&exchange[which]);
                refdists_used.insert(&exchange[which]);
            }
            else if (algorithm::contains(nm, "rGT")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(exchange[which]._dim == 0 || exchange[which]._dim == 6);
                if (exchange[which]._dim == 0) {
                    exchange[which]._name = str(format("exchangeabilities-%d") % which);
                    exchange[which].setDimension(6);
                }
                exchange[which].addCol(col);
                refdists.push_back(&exchange[which]);
                refdists_used.insert(&exchange[which]);
            }
            else if (algorithm::contains(nm, "piA")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(basefreq[which]._dim == 0 || basefreq[which]._dim == 4);
                if (basefreq[which]._dim == 0) {
                    basefreq[which]._name = str(format("basefreqs-%d") % which);
                    basefreq[which].setDimension(4);
                }
                basefreq[which].addCol(col);
                refdists.push_back(&basefreq[which]);
                refdists_used.insert(&basefreq[which]);
            }
            else if (algorithm::contains(nm, "piC")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(basefreq[which]._dim == 0 || basefreq[which]._dim == 4);
                if (basefreq[which]._dim == 0) {
                    basefreq[which]._name = str(format("basefreqs-%d") % which);
                    basefreq[which].setDimension(4);
                }
                basefreq[which].addCol(col);
                refdists.push_back(&basefreq[which]);
                refdists_used.insert(&basefreq[which]);
            }
            else if (algorithm::contains(nm, "piG")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(basefreq[which]._dim == 0 || basefreq[which]._dim == 4);
                if (basefreq[which]._dim == 0) {
                    basefreq[which]._name = str(format("basefreqs-%d") % which);
                    basefreq[which].setDimension(4);
                }
                basefreq[which].addCol(col);
                refdists.push_back(&basefreq[which]);
                refdists_used.insert(&basefreq[which]);
            }
            else if (algorithm::contains(nm, "piT")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(basefreq[which]._dim == 0 || basefreq[which]._dim == 4);
                if (basefreq[which]._dim == 0) {
                    basefreq[which]._name = str(format("basefreqs-%d") % which);
                    basefreq[which].setDimension(4);
                }
                basefreq[which].addCol(col);
                refdists.push_back(&basefreq[which]);
                refdists_used.insert(&basefreq[which]);
            }
            else if (algorithm::contains(nm, "ratevar")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(ratevar[which]._dim == 0 || ratevar[which]._dim == 1);
                if (ratevar[which]._dim == 0) {
                    ratevar[which]._name = str(format("ratevar-%d") % which);
                    ratevar[which].setDimension(1);
                }
                ratevar[which].addCol(col);
                refdists.push_back(&ratevar[which]);
                refdists_used.insert(&ratevar[which]);
            }
            else if (algorithm::contains(nm, "relrate")) {
                split(parts, nm, boost::is_any_of("-"));
                unsigned which = stoi(parts[1]);
                assert(relrate._dim <= 12);
                if (relrate._dim == 0)
                    relrate._name = "subset_relrates";
                if (relrate._dim < which)
                    relrate.setDimension(which);
                relrate.addCol(col);
                refdists.push_back(&relrate);
                refdists_used.insert(&relrate);
            }
            else {
                throw XLoRaDML(format("GHM cannot be calculated because column name \"%s\" was unexpected") % nm);
            }
        }
        
        // Accumulate sums and sum of squares over all rows of the data matrix
        for (unsigned row = 0; row < nrows; row++) {
            // Increment _n and reset the element index _i to zero for each RefDist object used
            for (auto refdist : refdists_used) {
                refdist->processSampledParameter();
            }
            
            // Add parameter values to the appropriate RefDist object
            for (unsigned col = 0; col < ncols; col++) {
                double x = _ghm_parameter_vectors[row][col];
                refdists[col]->addValue(x);
            }
        }
        
        if (!_quiet) {
            ::om.outputConsole("\nReference distributions:\n");
        }
        
        for (auto refdist : refdists_used) {
            refdist->calcParams();
            if (!_quiet) {
                ::om.outputConsole(format("  %s (%d-%d): %s\n") % refdist->_name % refdist->_first % refdist->_last % refdist->_parameter_string);
            }
        }
        
        if (!_quiet) {
            ::om.outputConsole("\nComputing GHM estimate...\n");
        }
        
        vector<double> log_ratios;
        for (unsigned row = 0; row < nrows; row++) {
            double logK = _ghm_kernel_values[row];
            
            // Compute log of the joint reference density
            double logR = 0.0;
            for (auto refdist : refdists_used) {
                if (refdist->_dim == 1) {
                    assert(refdist->_first == refdist->_last && refdist->_first != -1);
                    logR += refdist->calcLogProbDensity(_ghm_parameter_vectors[row][refdist->_first]);
                }
                else {
                    assert(refdist->_first != -1 && refdist->_last != -1 && refdist->_first < refdist->_last);
                    unsigned from = (unsigned)refdist->_first;
                    unsigned to   = (unsigned)refdist->_last;
                    assert(refdist->_dim == to - from + 1);
                    vector<double> v(refdist->_dim);
                    for (unsigned c = from; c <= to; c++) {
                        v[c - from] = _ghm_parameter_vectors[row][c];
                    }
                    logR += refdist->calcLogProbDensity(v);
                }
            }
            
            double log_ratio = logR - logK; //@@@
            log_ratios.push_back(log_ratio);
        }
        
        // Compute the log of the sum of the saved log ratios (using floating point control)
        unsigned n = (unsigned)log_ratios.size();
        assert(n > 0);
        double logmaxr = *std::max_element(log_ratios.begin(), log_ratios.end());
        double sumexp = 0.0;
        std::for_each(log_ratios.begin(), log_ratios.end(), [logmaxr,&sumexp](double logr){sumexp += exp(logr - logmaxr);});
        assert(sumexp > 0.0);
        
        // Compute the GHM estimate
        double log_inverse_marginal_likelihood = logmaxr + log(sumexp) - log(n);
        double log_marginal_likelihood = -log_inverse_marginal_likelihood;

        return log_marginal_likelihood;
    }
#endif

    inline void LoRaDML::run() {
        if (!_quiet) {
            ::om.outputConsole(format("This is %s (ver. %d.%d)\n") % _program_name % _major_version % _minor_version);
            ::om.outputConsole(format("  Parameter sample file is \"%s\"\n") % _param_file);
            ::om.outputConsole(format("  Training fraction is %.5f\n") % _training_fraction);
            ::om.outputConsole(format("  Coverage specified is %.5f\n") % _coverage);
            ::om.outputConsole(format("  Starting sample is %d\n") % _starting_sample);
            ::om.outputConsole(format("  Ending sample is %d\n") % _ending_sample, true);
            ::om.outputConsole(format("  MCSE calculation requested: %s\n") % (_mcse ? "yes" : "no"));
            if (_mcse)
                ::om.outputConsole(format("    T/B ratio requested: %.1f\n") % _TBratio);
        }
        
        // Read parameter file
        readParamFile();
        
        // Populate _training_sample and _estimation_sample
        setup();
        partitionSample(_first_index, _last_index);
        
        // Use _training_sample to determine _rmax, _mean_vector, and _covariance_matrix
        // and then use those to standardize and filter the _estimation_sample
        standardizeParameters();
        
        // Use _standardized_sample to estimate the marginal likelihood
        calcMarginalLikelihood();

        if (_mcse) {
            chrono::steady_clock::time_point mcse_begin = chrono::steady_clock::now();

            if (!_quiet) {
                double fT = (float)_T;
                double fB = (float)_B;
                double TBratio = fT/fB;
                ::om.outputConsole("\nEstimating MCSE...\n");
                ::om.outputConsole(format("  Total sample size T is %d\n") % _T);
                ::om.outputConsole(format("  Batch size B is %d\n") % _B);
                ::om.outputConsole(format("  Realized T/B ratio is %.1f\n") % TBratio);
                ::om.outputConsole(format("  Number of batches is %d\n") % _nbatches);
            }
            _quiet = true;
            
            vector<double> eta(_nbatches, 0.0);
            
            bool progress_header_shown = false;
            for (unsigned b = 0; b < _nbatches; b++) {
                chrono::steady_clock::time_point mcse_now = chrono::steady_clock::now();
                unsigned secs = (unsigned)chrono::duration_cast<chrono::seconds>(mcse_now - mcse_begin).count();
                if (secs > 30) {
                    if (!progress_header_shown)
                        ::om.outputConsole("\nProgress...\n", true);
                    double bpct = 100.0*(b + 1)/_nbatches;
                    ::om.outputConsole(format("%12d of %d (%.1f%%)\n") % (b + 1) % _nbatches % bpct, true);
                    mcse_begin = chrono::steady_clock::now();
                    progress_header_shown = true;
                }
                unsigned start_at = _first_index + b;
                unsigned end_at = start_at + _B;
                
                // Populate _training_sample and _estimation_sample
                partitionSample(start_at, end_at);
                
                // Use _training_sample to determine _rmax, _mean_vector, and _covariance_matrix
                // and then use those to standardize and filter the _estimation_sample
                standardizeParameters();
                
                // Use _standardized_sample to estimate the marginal likelihood
                double log_marg_like = calcMarginalLikelihood(false);
                eta[b] = log_marg_like;
            }
            
            double mean_eta = accumulate(eta.begin(), eta.end(), 0.0)/_nbatches;
            double mean_square = 0.0;
            for (auto x : eta) {
                mean_square += pow(x - mean_eta, 2.);
            }
            mean_square *= float(_B);
            mean_square /= float(_T - _B);
            mean_square /= float(_T - _B + 1);
            double MCSE = sqrt(mean_square);
            ::om.outputConsole(format("  MCSE is %.5f\n") % MCSE);
        }
        
#if defined(GHM)
        if (_ghm) {
            double logLghm = doGHM();
            ::om.outputConsole(format("  GHM estimate is %.5f\n") % logLghm);
        }
#endif
    }

}   // namespace loradML

