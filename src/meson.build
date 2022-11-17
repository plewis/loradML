project('loradML', 'cpp',
	default_options : ['cpp_std=c++11','buildtype=release','prefix=/home/pol02003/yubo'],
	version : '1.0')
cpp = meson.get_compiler('cpp')

# These lines specify the locations of the NCL, Boost, and BeagleLib compiled libraries
lib_system = cpp.find_library('boost_system', dirs: ['/home/pol02003/lib/static'], required: true)
lib_filesystem = cpp.find_library('boost_filesystem', dirs: ['/home/pol02003/lib/static'], required: true)
lib_program_options = cpp.find_library('boost_program_options', dirs: ['/home/pol02003/lib/static'], required: true)

# These lines specify the locations of header files for the NCL, Boost, BeagleLib, and Eigen library
incl_boost = include_directories('/home/pol02003/boost_1_72_0')
incl_eigen = include_directories('/home/pol02003/eigen-eigen-323c052e1731')

# This line creates the executable file
executable('loradML', 'main.cpp', install: true, install_dir: '.', dependencies: [lib_program_options,lib_system,lib_filesystem], include_directories: [incl_boost,incl_eigen])

# These lines just copy files to the install directory
#install_data('loradml.conf', install_dir: '.')

