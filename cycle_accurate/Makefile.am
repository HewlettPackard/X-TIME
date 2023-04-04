AM_CXXFLAGS = -Wall -Wno-deprecated-declarations -Wno-uninitialized -Wno-unused-local-typedefs -Wno-sign-compare -std=c++11 -fopenmp
AM_CPPFLAGS = 	-I$(top_srcdir)/src \
	$(MPI_CPPFLAGS) \
	$(PYTHON_CPPFLAGS) $(CPPFLAGS)
  
compdir = $(pkglibdir)
#compdir = $(cwd)
comp_LTLIBRARIES = libxtime.la
#sstdir = $(includedir)/sst/elements/supersim

libxtime_la_SOURCES = \
	pyxtime.py \
	src/pymodule.cc \
	src/acam.cc \
	src/accumulator.cc \
	src/adder.cc \
	src/control.cc \
	src/demux.cc \
	src/driver.cc \
	src/memory.cc \
	src/mmr.cc \
	src/mpe.cc
	
libxtime_la_LDFLAGS = -module -avoid-version -L$(pkglibdir)

BUILT_SOURCES = pyxtime.inc

# This sed script converts 'od' output to a comma-separated list of byte-
# values, suitable for #include'ing into an array definition.
# This can be done much more simply with xxd or hexdump, but those tools
# are not installed by default on all supported platforms.
#
# od:	-v:		Print all data
#		-t x1:	Print as byte-values, in hex
# sed:	Script 1:  Remove base-address column from od output
# 		Script 2:  Remove trailing blank line resulting from script 1
# 		Script 3:  Add '0x' prefix, and ',' suffix to each value
%.inc: %.py
	od -v -t x1 < $< | sed -e 's/^[^ ]*[ ]*//g' -e '/^\s*$$/d' -e 's/\([0-9a-f]*\)[ $$]*/0x\1,/g' > $@

install-exec-hook:
	$(SST_REGISTER_TOOL) SST_ARO_LIBRARY SST_ARO_LIBRARY_LIBDIR=$(pkglibdir)


clean-local: clean-local-check
clean-local-check:
	-rm -rf pyxtime.inc
