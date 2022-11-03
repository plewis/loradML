#pragma once

#include <vector>
#include <fstream>
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
            enum column_t {
                ignore              = 0,
                unconstrained       = 1,
                posterior           = 2,
                iteration           = 3,
                unknown             = 4
            };
            
            ColumnSpec() {
                _coltype = unknown;
            }
            
            ColumnSpec(string t, string nm) {
                _name = nm;
                if (t == "unconstrained") {
                    _coltype = unconstrained;
                }
                else if (t == "ignore") {
                    _coltype = ignore;
                }
                else if (t == "posterior") {
                    _coltype = posterior;
                }
                else if (t == "iteration") {
                    _coltype = iteration;
                }
                else {
                    _coltype = unknown;
                }
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
        
            double                      calcLogSum(const std::vector<double> & logx_vect);
            void                        readParamFile();
            void                        handleColSpecs();
            void                        partitionSample();
            void                        standardizeParameters();
            void                        calcMarginalLikelihood();
            
            bool                        _quiet;
                        
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
            unsigned                        _ntraining;      // number of samples used for training
            unsigned                        _nestimation;    // number of samples used for estimation
            unsigned                        _nsamples;       // _ntraining + _nestimation
            unsigned                        _nparameters;    // number of free parameters
            std::vector< ParameterSample >  _training_sample;
            std::vector< ParameterSample >  _estimation_sample;

            unsigned                        _first_index;   // index of first parameter sample used
            unsigned                        _last_index;    // one beyond index of last parameter sample
            unsigned                        _boundary;      // index of parameter sample at boundary between
                                                            // training and estimation samples
                                                            
            // Standardization
            typedef Eigen::VectorXd eigenVectorXd_t;
            typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigenMatrixXd_t;
            eigenVectorXd_t                 _mean_transformed;
            eigenMatrixXd_t                 _S;
            eigenMatrixXd_t                 _sqrtS;
            eigenMatrixXd_t                 _invSqrtS;
            double                          _logDetSqrtS;
        
            // Related to reading in the parameter file
            string                      _param_file;            // paramfile in loradML.conf
            vector<string>              _colspecs;              // vector of colspec entries from loradML.conf
            vector<ColumnSpec>          _column_specifications; // vector of ColumnSpec objects
            vector<string>              _column_names;          // vector column headers
            vector<unsigned>            _orig_iter;         // element i is the original MCMC iteration for the ith sample
            vector<double>              _kernel_values;     // element i is the log-kernel of the ith sample
            vector< vector<double> >    _parameter_vectors; // element i is the ith sampled parameter vector
            
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
        _ntraining = 0;
        _nestimation = 0;
        _nsamples = 0;
        _nparameters = 0;
        _first_index = 0;
        _last_index = 0;
        _boundary = 0;
        _rmax = 0.0;
    }
    
    inline void LoRaDML::processCommandLineOptions(int argc, const char * argv[]) {
        program_options::variables_map vm;
        program_options::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("version,v", "show program version")
            ("quiet,q", program_options::value(&_quiet)->default_value(false), "if yes, the only output will be the estimated log marginal likelihood")
            ("paramfile", program_options::value(&_param_file)->required(), "name of file containing sampled parameter values")
            ("colspec", program_options::value(&_colspecs), "column specification (provide one for each column in paramfile)")
            ("startsample", program_options::value(&_starting_sample)->default_value(0), "first sample to consider (starting with 1; specify 0 to consider all samples)")
            ("endsample", program_options::value(&_ending_sample)->default_value(0), "last sample to consider (starting with 1; specify 0 to consider all samples)")
            ("trainingfrac", program_options::value(&_training_fraction)->default_value(0.5), "fraction of sample to use for training (determining rmax and mean vector and covariance matrix for standardization)")
            ("coverage", program_options::value(&_coverage)->default_value(0.25), "fraction of training sample to use for determining rmax")
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
            ::om.outputConsole();
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
    }
    
    inline void LoRaDML::handleColSpecs() {
        _column_specifications.resize(_colspecs.size());
        vector<string> parts;
        for (auto spec : _colspecs) {
            trim(spec);
            split(parts, spec, boost::is_any_of(" "), token_compress_on);
            unsigned nparts = (unsigned)parts.size();
            if (nparts != 3) {
                throw XLoRaDML(format("Expecting colspec to comprise 3 parts (column, type, and name), but found %d parts: \"%s\"") % nparts % spec);
            }
            unsigned i = stoi(parts[0]) - 1;
            assert(_column_specifications[i]._coltype == ColumnSpec::unknown);
            _column_specifications[i] = ColumnSpec(parts[1], parts[2]);
        }
        if (!_quiet) {
            ::om.outputConsole(format("\nProcessed %d column specifications\n") % _colspecs.size());
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
        if (!_quiet) {
            ::om.outputConsole(format("Reading sampled parameters from file \"%s\"\n") % _param_file);
        }

        // Initializations
        _orig_iter.clear();
        _kernel_values.clear();
        _parameter_vectors.clear();
        
        // Read the file
        ifstream inf(_param_file);
        string line;
        vector<string> tmp;
        vector<double> tmp_param_vect;
        unsigned i = 0;
        while (getline(inf, line)) {
            if (i == 0) {
                // Store column headers in _column_names data member
                split(_column_names, line, boost::is_any_of("\t"));
                
                // Determine number of parameters by examining _column_specifications
                _nparameters = 0;
                for (auto & cspec : _column_specifications) {
                    if (cspec._coltype == ColumnSpec::unconstrained)
                        ++_nparameters;
                }
                tmp_param_vect.resize(_nparameters);
            }
            else {
                // store parameter vector and posterior kernel for this sample
                
                // Split line from parameter file into tab-delimited columns
                trim(line);
                split(tmp, line, boost::is_any_of("\t"));
                
                // Initializations
                unsigned col      = 0;    // index of column in parameter file
                unsigned param    = 0;    // index of parameter in parameter vector
                unsigned origiter = 0;    // original iteration of sample
                double   kernel   = 0.0;  // posterior kernel value from parameter file
                
                // Go through each column spec and column, pulling out information for this sample
                for (auto & cspec : _column_specifications) {
                    if (cspec._coltype == ColumnSpec::iteration) {
                        unsigned it = 0;
                        try {
                            it = stoi(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("Could not convert string \"%s\" in column %d to an integer representing the MCMC iteration for this sample") % tmp[col] % col);
                        }
                        origiter = it;
                    }
                    else if (cspec._coltype == ColumnSpec::posterior) {
                        double v = 0.0;
                        try {
                            v = stod(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("Could not convert string \"%s\" in column %d to a floating-point posterior kernel compnent") % tmp[col] % col);
                        }
                        kernel += v;
                    }
                    else if (cspec._coltype == ColumnSpec::unconstrained) {
                        double v = 0.0;
                        try {
                            v = stod(tmp[col]);
                        }
                        catch(...) {
                            throw XLoRaDML(format("Could not convert string \"%s\" in column %d to a floating-point parameter value") % tmp[col] % col);
                        }
                        tmp_param_vect[param++] = v;
                    }
                    ++col;
                }
                
                // Can now store origiter, kernel, and the parameter vector
                _orig_iter.push_back(origiter);
                _kernel_values.push_back(kernel);
                _parameter_vectors.push_back(tmp_param_vect);
            }
            ++i;
        }
        
        // Provide a summary
        if (_column_names.size() == 0) {
            throw XLoRaDML("  File seems to be empty or does not comprise columns of numbers");
        }
        if (!_quiet) {
            ::om.outputConsole(format("  Found %d columns\n") % _column_names.size());
            ::om.outputConsole(format("  Found %d values for each column\n") % _orig_iter.size());
        }
    }
    
    inline void LoRaDML::partitionSample() {
        // Determine total number of samples
        _nsamples_total = (unsigned)_orig_iter.size();
        assert(_nsamples_total > 0);
        
        // Allocate _training_sample and _estimation_sample
        //
        //   1    2    3    4    5    6    7    8    9
        //   |                                       |
        //   _starting_sample                        _ending_sample
        //
        //   0    1    2    3    4    5    6    7    8    9
        //   |                        |                   |
        //   _first_index             _boundary           _last_index
        //
        //   _trainingfrac = 0.5
        //   _boundary = ceiling(0.5*9)) = 5
        //   _ntraining   = 5 - 0 = 5
        //   _nestimation = 9 - 5 = 4
        //   _nsamples    = 9 - 0 = 9
        _first_index = (_starting_sample == 0 ? 0 : _starting_sample - 1);
        _last_index  = (_ending_sample == 0 ? _nsamples_total : _ending_sample);
        _nsamples    = _last_index - _first_index;
        _boundary    = _first_index + int(ceil(_training_fraction*_nsamples));
        _ntraining   = _boundary - _first_index;
        _nestimation = _last_index - _boundary;
        
        // cerr << "debugging: _starting_sample   = " << _starting_sample << endl;
        // cerr << "debugging: _ending_sample     = " << _ending_sample << endl;
        // cerr << "debugging: _training_fraction = " << _training_fraction << endl;
        // cerr << "debugging: _first_index       = " << _first_index << endl;
        // cerr << "debugging: _last_index        = " << _last_index << endl;
        // cerr << "debugging: _nsamples          = " << _nsamples << endl;
        // cerr << "debugging: _boundary          = " << _boundary << endl;
        // cerr << "debugging: _ntraining         = " << _ntraining << endl;
        // cerr << "debugging: _nestimation       = " << _nestimation << endl;
    
        // Store training sample
        _training_sample.resize(_ntraining);
        unsigned j = 0;
        for (unsigned i = _first_index; i < _boundary; i++) {
            _training_sample[j++].init(_orig_iter[i], i, _kernel_values[i], 0.0, _parameter_vectors[i]);
        }
        
        // Store estimation sample
        _estimation_sample.resize(_nestimation);
        j = 0;
        for (unsigned i = _boundary; i < _last_index; i++) {
            _estimation_sample[j++].init(_orig_iter[i], i, _kernel_values[i], 0.0, _parameter_vectors[i]);
        }
        
        if (!_quiet) {
            ::om.outputConsole(format("  Sample (%d) partitioned into training (%d) and estimation sets (%d)\n") % _nsamples % _ntraining % _nestimation);
        }
    }
    
    inline void LoRaDML::standardizeParameters() {
        if (!_quiet)
            ::om.outputConsole("  Standardizing parameters...\n");
                
        // Zero the mean vector (_mean_transformed)
        assert(_nparameters > 0);
        _mean_transformed = eigenVectorXd_t::Zero(_nparameters);
        
        // Zero the variance-covariance matrix (_S)
        _S.resize(_nparameters, _nparameters);
        _S = eigenMatrixXd_t::Zero(_nparameters, _nparameters);
        
        // Calculate mean vector _mean_transformed
        assert(_ntraining == (unsigned)_training_sample.size());
        assert(_ntraining > 1);
        for (auto & v : _training_sample) {
            // Add v._param_vect elementwise to _mean_transformed
            _mean_transformed += v._param_vect;
        }
        _mean_transformed /= _ntraining;
        
        // Sanity check
        assert(_mean_transformed.rows() == _nparameters);

        // Calculate variance-covariance matrix _S
        for (auto & v : _training_sample) {
            eigenVectorXd_t x = v._param_vect - _mean_transformed;
            _S += x*x.transpose();
        }
        _S /= _ntraining - 1;
        
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
    
    inline void LoRaDML::calcMarginalLikelihood() {
        // Determine how many sample vectors to use for working parameter space
        unsigned nretained = (unsigned)floor(_coverage*_ntraining);
        assert(nretained > 1);
        
        double _rmax = _training_sample[nretained]._norm;

        //::om.outputConsole(boost::format("  _rmax = %.5f\n") % _rmax);
        
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
        for (unsigned i = 0; i < _nestimation; ++i) {
            double norm = _estimation_sample[i]._norm;
            if (norm > _rmax)
                break;
            double log_kernel = _estimation_sample[i]._kernel;
            double log_reference = -0.5*sigma_squared*pow(norm,2.0) - log_mvnorm_constant;
            double log_ratio = log_reference - log_kernel;
            log_ratios.push_back(log_ratio);
        }
        
        double log_sum_ratios = calcLogSum(log_ratios);
        double log_marginal_likelihood = log_Delta - (log_sum_ratios - log(_nestimation));
        
        //::om.outputConsole(format("  no. samples used = %d\n") % log_ratios.size());
        //::om.outputConsole(format("  fraction used = %.5f\n") % (1.0*log_ratios.size()/_nestimation));
        ::om.outputConsole(format("log marginal likelihood = %.5f\n") % log_marginal_likelihood);
    }
    
    inline void LoRaDML::run() {
        if (!_quiet) {
            ::om.outputConsole(format("This is %s (ver. %d.%d)\n") % _program_name % _major_version % _minor_version);
        }
        
        // Read parameter file
        readParamFile();
        
        // Populate _training_sample and _estimation_sample
        partitionSample();
        
        // Use _training_sample to determine _rmax, _mean_vector, and _covariance_matrix
        // and then use those to standardize and filter the _estimation_sample
        standardizeParameters();
        
        // Use _standardized_sample to estimate the marginal likelihood
        calcMarginalLikelihood();
    }

}   // namespace loradML

