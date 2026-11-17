# The engine's old name beside the new one (v1.3.164: `strata` became `abyss`): configs, scripts and other checkouts
# that start <build>/strata keep working.  A symbolic link on Linux, a copy on Windows (no links without developer
# mode there).  Run after each link of the engine (cmake -P, ENGINE = the file just built); never fails the build -
# a strata.exe still running keeps the old copy until the next build.
get_filename_component(_dir "${ENGINE}" DIRECTORY)
get_filename_component(_name "${ENGINE}" NAME)
get_filename_component(_ext "${ENGINE}" LAST_EXT)
set(_alias "${_dir}/strata${_ext}")
if(WIN32 OR _ext STREQUAL ".exe")
  file(COPY_FILE "${ENGINE}" "${_alias}" RESULT _rc ONLY_IF_DIFFERENT)
else()
  file(REMOVE "${_alias}")
  file(CREATE_LINK "${_name}" "${_alias}" RESULT _rc SYMBOLIC)
endif()
if(NOT _rc EQUAL 0)
  message(WARNING "could not write ${_alias} (the engine's old name): ${_rc}")
endif()
